# Adapted from Coconut (https://github.com/facebookresearch/coconut).
# Original code Copyright (c) Meta Platforms, Inc. and affiliates., MIT License.

import itertools
import json
import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning


def get_dataset(path, tokenizer, max_size=1000000000):

    def tokenize_sample(sample):

        question_tokenized = tokenizer.encode(
            sample["question"] + "\n", add_special_tokens=True
        )
        steps_tokenized = [
            tokenizer.encode(s + "\n", add_special_tokens=False)
            for s in sample["steps"]
        ]
        answer_tokenized = tokenizer.encode(
            "### " + sample["answer"], add_special_tokens=False
        ) + [tokenizer.eos_token_id]

        sample = {
            "question_tokenized": question_tokenized,
            "steps_tokenized": steps_tokenized,
            "answer_tokenized": answer_tokenized,
            "idx": sample["idx"],
        }
        return sample

    data = json.load(open(path))[:max_size]
    data = [{**d, "idx": idx} for idx, d in enumerate(data)]

    keys = data[0].keys()
    dataset = Dataset.from_dict({k: [d[k] for d in data] for k in keys})

    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            processed_dataset = [
                dataset.map(
                    tokenize_sample, remove_columns=list(dataset.features), num_proc=32
                )
            ]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]

    else:
        dataset = dataset.map(
            tokenize_sample, remove_columns=list(dataset.features), num_proc=32
        )

    # verify
    d = data[0]
    complete = d["question"] + "\n" + "\n".join(d["steps"]) + "\n### " + d["answer"]
    complete_tokenized = tokenizer.encode(complete, add_special_tokens=True) + [
        tokenizer.eos_token_id
    ]
    assert (
        complete_tokenized
        == dataset[0]["question_tokenized"]
        + list(itertools.chain.from_iterable(dataset[0]["steps_tokenized"]))
        + dataset[0]["answer_tokenized"]
    )

    return dataset


@dataclass
class MyCollator:

    tokenizer: PreTrainedTokenizerBase
    latent_id: Optional[int] = None
    label_pad_token_id: Optional[int] = -100

    def __call__(self, features, return_tensors=None):

        assert self.tokenizer.padding_side == "right"

        """
        Pad the batch like this to maximize the reuse of kv cache.
        E.g.,
        
        xxxxxxxxxx<latent><latent>xxxxx--
        -----xxxxx<latent>xxxxxxxx-------
        ---xxxxxxx<latent><latent>xxxxxxx


        ("x" is word token, "-" is pad token)
        """

        earliest_latent = [
            feature["input_ids"].index(self.latent_id)
            for feature in features
            if self.latent_id in feature["input_ids"]
        ]

        if len(earliest_latent) > 0:  # if there are continuous thoughts in the sequence
            latest_earliest_latent = max(earliest_latent)
            for feature in features:
                if self.latent_id in feature["input_ids"]:
                    n_tok_pad = latest_earliest_latent - feature["input_ids"].index(
                        self.latent_id
                    )
                else:
                    n_tok_pad = 0
                feature["position_ids"] = [0] * n_tok_pad + list(
                    range(len(feature["input_ids"]))
                )
                feature["input_ids"] = [
                    self.tokenizer.pad_token_id
                ] * n_tok_pad + feature["input_ids"]
                if "labels" in feature:
                    feature["labels"] = [self.label_pad_token_id] * n_tok_pad + feature[
                        "labels"
                    ]
                if "answer_labels" in feature:
                    feature["answer_labels"] = [self.label_pad_token_id] * n_tok_pad + feature[
                        "answer_labels"
                    ]
                feature["attention_mask"] = [0] * n_tok_pad + feature["attention_mask"]

        return_tensors = "pt"

        label_name = "label" if "label" in features[0].keys() else "labels"

        # Fields that need special handling (not passed to tokenizer's pad)
        special_fields = {label_name, "position_ids", "replaced_cot_steps", "answer_labels"}

        non_label_position_features = [
            {
                k: v
                for k, v in feature.items()
                if k not in special_fields
            }
            for feature in features
        ]

        # run through tokenizer without labels to ensure no side effects
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_label_position_features,
            padding=True,
            pad_to_multiple_of=None,
            return_tensors=return_tensors,
        )

        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )
        if labels is not None and all(label is None for label in labels):
            labels = None
        position_ids = (
            [feature["position_ids"] for feature in features]
            if "position_ids" in features[0].keys()
            else None
        )
        # we have to pad the labels and position_ids manually as we cannot rely on `tokenizer.pad`

        if labels is not None:
            max_label_length = max(len(l) for l in labels)

            batch["labels"] = [
                label + [self.label_pad_token_id] * (max_label_length - len(label))
                for label in labels
            ]
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)

        if position_ids is not None:
            max_pos_length = max(len(l) for l in position_ids)

            batch["position_ids"] = [
                position_id + [0] * (max_pos_length - len(position_id))
                for position_id in position_ids
            ]
            batch["position_ids"] = torch.tensor(
                batch["position_ids"], dtype=torch.int64
            )

        # Handle answer_labels for answer-only loss computation
        if "answer_labels" in features[0]:
            answer_labels = [feature["answer_labels"] for feature in features]
            max_answer_label_length = max(len(l) for l in answer_labels)
            batch["answer_labels"] = [
                al + [self.label_pad_token_id] * (max_answer_label_length - len(al))
                for al in answer_labels
            ]
            batch["answer_labels"] = torch.tensor(batch["answer_labels"], dtype=torch.int64)

        # Handle replaced_cot_steps: (batch, max_n_steps, max_step_len)
        if "replaced_cot_steps" in features[0]:
            all_steps = [feature["replaced_cot_steps"] for feature in features]
            max_n_steps = max(len(steps) for steps in all_steps) if all_steps else 0
            max_step_len = max(
                len(step) for steps in all_steps for step in steps
            ) if any(all_steps) else 0
            
            if max_n_steps > 0 and max_step_len > 0:
                padded_steps = []
                for steps in all_steps:
                    # Pad each step to max_step_len
                    padded = [
                        step + [self.tokenizer.pad_token_id] * (max_step_len - len(step))
                        for step in steps
                    ]
                    # Pad to max_n_steps with empty (all-pad) steps
                    while len(padded) < max_n_steps:
                        padded.append([self.tokenizer.pad_token_id] * max_step_len)
                    padded_steps.append(padded)
                
                batch["replaced_cot_steps"] = torch.tensor(
                    padded_steps, dtype=torch.int64
                )

        # Handle per-sample stage for cumulative training
        if "stage" in features[0]:
            batch["sample_stages"] = torch.tensor(
                [feature["stage"] for feature in features], dtype=torch.int64
            )

        return batch


def get_question_latent_dataset(
    scheduled_stage,
    base_dataset_valid,
    configs,
    start_id,
    latent_id,
    end_id,
    no_special_marker=False,
    single_latent_token=False,
):

    def process_dataset(sample):

        if configs.pad_latent_to_max:
            max_latent_stage = configs.max_latent_stage
        else:
            max_latent_stage = min(
                configs.max_latent_stage, len(sample["steps_tokenized"])
            )

        k = min(max_latent_stage, scheduled_stage)

        k *= configs.c_thought

        if single_latent_token:
            k = 1
        elif hasattr(configs, 'fixed_latent_tokens') and configs.fixed_latent_tokens > 0:
            k = configs.fixed_latent_tokens

        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * k
            + ([] if no_special_marker else [end_id])
        )

        return {
            "input_ids": tokens,
            "idx": sample["idx"],
            "attention_mask": [1] * len(tokens),
            "position_ids": list(range(len(tokens))),
        }

    return base_dataset_valid.map(
        process_dataset, remove_columns=list(base_dataset_valid.features), num_proc=32
    )


def get_cot_latent_dataset(
    scheduled_stage,
    base_dataset,
    configs,
    start_id,
    latent_id,
    end_id,
    no_special_marker=False,
    shuffle=False,
    single_latent_token=False,
    shuffle_seed=None,
):

    n_additional_tokens = 0 if no_special_marker else 2

    def process_dataset(sample):

        if (
            random.random() < configs.uniform_prob
        ):  # with some prob, randomly sample stage
            scheduled_stage_to_train = random.choice(
                list(range(len(sample["steps_tokenized"]) + 1))
            )
        else:
            scheduled_stage_to_train = scheduled_stage

        if scheduled_stage_to_train > configs.max_latent_stage:
            n_skip_steps = configs.max_latent_stage  # Cap at max, rest will be visible CoT
            n_latent_tokens = configs.max_latent_stage

        else:
            n_skip_steps, n_latent_tokens = (
                scheduled_stage_to_train,
                scheduled_stage_to_train,
            )

        if configs.no_cot:
            n_skip_steps = 100  # skip all step
            n_latent_tokens = 0

        n_latent_tokens *= configs.c_thought

        if single_latent_token:
            n_latent_tokens = 1
        elif hasattr(configs, 'fixed_latent_tokens') and configs.fixed_latent_tokens > 0:
            n_latent_tokens = configs.fixed_latent_tokens

        # CoT steps that are being replaced by latent tokens (list of lists)
        replaced_cot_steps = sample["steps_tokenized"][:n_skip_steps]

        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * n_latent_tokens
            + ([] if no_special_marker else [end_id])
            + list(
                itertools.chain.from_iterable(sample["steps_tokenized"][n_skip_steps:])
            )
            + sample["answer_tokenized"]
        )

        # Answer-only labels: mask everything except answer tokens
        answer_len = len(sample["answer_tokenized"])
        answer_labels = [-100] * (len(tokens) - answer_len) + tokens[-answer_len:]

        return {
            "input_ids": tokens,
            "labels": [-100]
            * (
                len(sample["question_tokenized"])
                + n_latent_tokens
                + n_additional_tokens
            )
            + tokens[
                n_latent_tokens
                + n_additional_tokens
                + len(sample["question_tokenized"]) :
            ],
            "answer_labels": answer_labels,  # For answer-only loss computation
            "attention_mask": [1] * len(tokens),
            "idx": sample["idx"],
            "position_ids": list(range(len(tokens))),
            "replaced_cot_steps": replaced_cot_steps,  # List of token lists, one per replaced step
        }

    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            processed_dataset = base_dataset.map(
                process_dataset, remove_columns=list(base_dataset.features), num_proc=32
            )
            if shuffle:
                processed_dataset = processed_dataset.shuffle(seed=shuffle_seed)
            processed_dataset = [processed_dataset]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]

    else:
        processed_dataset = base_dataset.map(
            process_dataset, remove_columns=list(base_dataset.features), num_proc=32
        )
        if shuffle:
            processed_dataset = processed_dataset.shuffle(seed=shuffle_seed)
        dataset = processed_dataset

    return dataset


def get_cumulative_cot_latent_dataset(
    scheduled_stage,
    base_dataset,
    configs,
    start_id,
    latent_id,
    end_id,
    no_special_marker=False,
    shuffle=False,
    single_latent_token=False,
    shuffle_seed=None,
):
    """
    Multi-task dataset where each sample is assigned a random stage from 0..scheduled_stage.
    This provides indirect supervision: each block gets signal from samples where
    the corresponding CoT step is visible.
    
    Same epoch size as regular training, but mixed stages.
    """
    n_additional_tokens = 0 if no_special_marker else 2
    
    def process_dataset(sample):
        # Randomly sample a stage for this sample
        target_stage = random.randint(0, scheduled_stage)
        
        n_skip_steps = target_stage
        n_latent_tokens = target_stage
        
        if target_stage > configs.max_latent_stage:
            n_skip_steps = configs.max_latent_stage  # Cap at max, rest will be visible CoT
            n_latent_tokens = configs.max_latent_stage
        
        if configs.no_cot:
            n_skip_steps = 100  # skip all step
            n_latent_tokens = 0
        
        n_latent_tokens *= configs.c_thought
        
        if single_latent_token:
            n_latent_tokens = 1
        elif hasattr(configs, 'fixed_latent_tokens') and configs.fixed_latent_tokens > 0:
            n_latent_tokens = configs.fixed_latent_tokens
        
        # CoT steps that are being replaced by latent tokens (list of lists)
        replaced_cot_steps = sample["steps_tokenized"][:n_skip_steps]
        
        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * n_latent_tokens
            + ([] if no_special_marker else [end_id])
            + list(
                itertools.chain.from_iterable(sample["steps_tokenized"][n_skip_steps:])
            )
            + sample["answer_tokenized"]
        )
        
        # Answer-only labels: mask everything except answer tokens
        answer_len = len(sample["answer_tokenized"])
        answer_labels = [-100] * (len(tokens) - answer_len) + tokens[-answer_len:]
        
        return {
            "input_ids": tokens,
            "labels": [-100]
            * (
                len(sample["question_tokenized"])
                + n_latent_tokens
                + n_additional_tokens
            )
            + tokens[
                n_latent_tokens
                + n_additional_tokens
                + len(sample["question_tokenized"]) :
            ],
            "answer_labels": answer_labels,  # For answer-only loss computation
            "attention_mask": [1] * len(tokens),
            "idx": sample["idx"],
            "position_ids": list(range(len(tokens))),
            "replaced_cot_steps": replaced_cot_steps,
            "stage": target_stage,  # Track which stage this sample is from
        }
    
    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            dataset = base_dataset.map(
                process_dataset, 
                remove_columns=list(base_dataset.features), 
                num_proc=32
            )
            if shuffle:
                dataset = dataset.shuffle(seed=shuffle_seed)
            processed_dataset = [dataset]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]
    else:
        dataset = base_dataset.map(
            process_dataset, 
            remove_columns=list(base_dataset.features), 
            num_proc=32
        )
        if shuffle:
            dataset = dataset.shuffle(seed=shuffle_seed)
    
    return dataset
