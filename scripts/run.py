# Adapted from Coconut (https://github.com/facebookresearch/coconut).
# Original code Copyright (c) Meta Platforms, Inc. and affiliates., MIT License.

import argparse
import functools
import gc
import itertools
import json
import os
import random
import sys
from copy import copy
from pathlib import Path

import numpy as np
import shutil

import torch
import torch.distributed
import torch.distributed as dist
import torch.optim as optim
import wandb
import yaml
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullOptimStateDictConfig, FullStateDictConfig, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# Ensure sibling modules (dataset, lotus, utils) are importable when this file
# is launched from the repo root, e.g. `torchrun scripts/run.py ...`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import (
    MyCollator,
    get_cot_latent_dataset,
    get_cumulative_cot_latent_dataset,
    get_dataset,
    get_question_latent_dataset,
)
from lotus import Lotus
from utils import Config, set_seed


def load_hierarchical_yaml(config_path):
    """
    Load a YAML config that may inherit from base configs.
    Supports 'base' key with a single file path or list of file paths.
    Configs are merged in order, with later configs overriding earlier ones.
    """
    config_path = Path(config_path)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # If no base configs, return as-is
    if "base" not in config:
        return config

    # Get base config paths
    base_paths = config.pop("base")
    if isinstance(base_paths, str):
        base_paths = [base_paths]

    # Load and merge base configs
    merged_config = {}
    for base_path in base_paths:
        # Resolve relative to args directory
        if not base_path.startswith("/"):
            base_path = config_path.parent / base_path
        else:
            base_path = Path(base_path)

        # Recursively load base config (supports multi-level inheritance)
        base_config = load_hierarchical_yaml(base_path)

        # Merge base config into merged_config
        merged_config.update(base_config)

    # Override with current config values
    merged_config.update(config)

    return merged_config


def main():

    parser = argparse.ArgumentParser(description="looped")
    parser.add_argument("config_file")
    parser.add_argument("--name", type=str, default=None, help="Override run name")
    args = parser.parse_args()

    # init distributed environment
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    # load the configuration file with hierarchical support
    config_dict = load_hierarchical_yaml(args.config_file)

    # Apply CLI overrides
    if args.name is not None:
        config_dict["name"] = args.name

    if rank == 0:
        print("Config:", config_dict)

    configs = Config(config_dict)
    set_seed(configs.seed)
    save_dir = os.path.join(configs.save_path, configs.name)

    if not os.path.exists(save_dir) and rank == 0:
        os.makedirs(save_dir)

    torch.distributed.barrier()
    cur_ckpts = os.listdir(save_dir)

    # check if the job is preempted and resumed.

    if configs.resume != 0:
        # If resume is explicitly set in config, prioritize it over automatic checkpoint detection
        if configs.load_model_path is None or configs.load_model_path == "None":
            # User specified resume but no checkpoint path - try to find checkpoint in save_dir
            checkpoint_name = f"checkpoint_{configs.resume}"
            potential_checkpoint = os.path.join(
                configs.save_path, configs.name, checkpoint_name
            )
            if os.path.exists(potential_checkpoint):
                configs.load_model_path = potential_checkpoint
                if rank == 0:
                    print(
                        f"Resume set to {configs.resume}. Found checkpoint at {potential_checkpoint}"
                    )
            else:
                if rank == 0:
                    print(
                        f"Warning: you want to skip the first {configs.resume} epochs but checkpoint {checkpoint_name} not found in {save_dir}!"
                    )
        else:
            if rank == 0:
                print(
                    f"Loading from {configs.load_model_path} and resuming from epoch {configs.resume}"
                )

    elif (
        len(cur_ckpts) > 0
        and not configs.only_eval
        and getattr(configs, "allow_resume_from_checkpoint", False)
    ):
        # if there are previous checkpoints, only_eval is False, resumption is allowed,
        # and resume was NOT explicitly set (resume == 0), then auto-resume from latest checkpoint.
        # This handles preemption recovery.

        checkpoints = [f for f in cur_ckpts if f.startswith("checkpoint_") and f.split("_")[1].isdigit()]
        checkpoints.sort(key=lambda x: int(x.split("_")[1]))

        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            configs.resume = int(latest_checkpoint.split("_")[1])
            load_dir = os.path.join(configs.save_path, configs.name, latest_checkpoint)
            configs.load_model_path = load_dir
            if rank == 0:
                print(f"Auto-resuming from {load_dir} (epoch {configs.resume})")
        else:
            if rank == 0:
                print("Warning: found files in save_dir but no valid checkpoint_N directories. Starting from scratch.")

    elif (
        len(cur_ckpts) > 0
        and not configs.only_eval
        and not getattr(configs, "allow_resume_from_checkpoint", False)
    ):
        # if there are previous checkpoints but resumption is disabled, warn the user
        checkpoints = [f for f in cur_ckpts if f.startswith("checkpoint_")]
        if rank == 0 and len(checkpoints) > 0:
            print(
                f"Warning: found {len(checkpoints)} checkpoint(s) in {save_dir}, "
                f"but allow_resume_from_checkpoint is set to False. Starting training from scratch."
            )

    model = AutoModelForCausalLM.from_pretrained(configs.model_id)
    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    loaded = False
    saved_weights = None  # Initialize early so we can check vocab size

    if configs.load_model_path is not None and configs.load_model_path != "None":
        # Check if it's a Hugging Face repo ID (doesn't start with / or .)
        if not configs.load_model_path.startswith(('/', '.', '~')):
            # Load from Hugging Face Hub (model-agnostic: GPT-2, Llama, etc.)
            print(f"Loading model from Hugging Face: {configs.load_model_path}")
            hf_model = AutoModelForCausalLM.from_pretrained(configs.load_model_path)
            saved_weights = hf_model.state_dict()
        else:
            # Load from local path (supports both file and directory checkpoints)
            model_load_path = configs.load_model_path
            if os.path.isdir(model_load_path):
                model_file = os.path.join(model_load_path, "model.pt")
                if os.path.exists(model_file):
                    model_load_path = model_file
            saved_weights = torch.load(
                model_load_path, map_location=torch.device(rank), weights_only=False
            )

        # Check if checkpoint has larger vocab (includes latent tokens)
        # and resize model BEFORE loading to avoid shape mismatch
        if saved_weights is not None:
            # Find embedding key in checkpoint
            emb_key = None
            for k in saved_weights.keys():
                if 'wte.weight' in k or 'embed_tokens.weight' in k:
                    emb_key = k
                    break
            
            if emb_key is not None:
                ckpt_vocab_size = saved_weights[emb_key].shape[0]
                model_vocab_size = model.get_input_embeddings().weight.shape[0]
                
                if ckpt_vocab_size > model_vocab_size:
                    if rank == 0:
                        print(f"Resizing model vocab from {model_vocab_size} to {ckpt_vocab_size} to match checkpoint")
                    model.resize_token_embeddings(ckpt_vocab_size)

        if configs.looped and not any(
            [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            # we are loading a base model into the latent (looped) model
            # e.g., for GSM8k, we used a SFTed model to skip the stage 0
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

        elif not configs.looped and any(
            [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            raise ValueError("Cannot load latent-model weights into a causallm model")

        elif configs.looped and any(
            [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            # loading from preempted run
            # will handle later
            pass

        else:
            # resume or evaluate sft model
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

    if not (configs.cot or configs.no_thoughts or configs.no_cot):
        # if we need new tokens, initialize their embeddings and lm heads
        model.resize_token_embeddings(len(tokenizer))
        
        # Only initialize with "<<" if NOT loading a checkpoint that already has trained latent tokens
        # A trained checkpoint with latent tokens would have the latent_id embedding already learned
        skip_latent_init = False
        if saved_weights is not None:
            # Check if checkpoint has base_causallm keys (looped checkpoint) 
            # or has same vocab size (trained model with latent tokens)
            has_looped_keys = any(k.startswith("base_causallm") for k in saved_weights.keys())
            
            # Check vocab size in checkpoint
            emb_key = None
            for k in saved_weights.keys():
                if 'wte.weight' in k or 'embed_tokens.weight' in k:
                    emb_key = k
                    break
            ckpt_has_latent_tokens = emb_key is not None and saved_weights[emb_key].shape[0] >= len(tokenizer)
            
            skip_latent_init = has_looped_keys or ckpt_has_latent_tokens
            if skip_latent_init and rank == 0:
                print("Skipping latent token initialization (using trained embeddings from checkpoint)")
        
        if not skip_latent_init:
            embeddings = model.get_input_embeddings()
            target_id = tokenizer.convert_tokens_to_ids("<<")
            # initialize the new token embeddings with a known token
            # it helps stablize the training
            for token_id in [latent_id, start_id, end_id]:
                target_embedding = embeddings.weight.data[target_id]
                embeddings.weight.data[token_id] = target_embedding
                # The input embeddings and lm heads are tied in GPT2. So the code below is not necessary
                lm_head = model.lm_head
                lm_head.weight.data[token_id] = lm_head.weight.data[target_id]

    if configs.no_thoughts:
        configs.c_thought = 0
        configs.looped = False

    if configs.looped:
        # Check if KV cache should be used (default: True so latent tokens attend to prefix)
        use_kv_cache = getattr(configs, "use_kv_cache", True)
        intermediate_loss_weight = getattr(configs, "intermediate_loss_weight", 0.0)
        freeze_supervised_blocks = getattr(configs, "freeze_supervised_blocks", False)
        intermediate_loss_type = getattr(configs, "intermediate_loss_type", "crossentropy")
        intermediate_label_smoothing = getattr(configs, "intermediate_label_smoothing", 0.0)

        # Teacher distillation settings
        teacher_model = None
        codi_loss_weight = getattr(configs, "codi_loss_weight", 0.0)
        codi_loss_type = getattr(configs, "codi_loss_type", "smooth_l1")
        teacher_model_path = getattr(configs, "teacher_model_path", None)

        # Load teacher model if any distillation loss needs it
        _need_teacher = (codi_loss_weight > 0)
        if teacher_model_path is not None and _need_teacher:
            if rank == 0:
                print(f"Loading teacher model from {teacher_model_path}")

            teacher_model = AutoModelForCausalLM.from_pretrained(
                configs.model_id,
                torch_dtype=torch.bfloat16 if configs.bf16 else torch.float32,
            )

            # Resolve teacher weights. Supports a Hugging Face repo id (same
            # convention as load_model_path: no leading '/', '.', or '~') or a
            # local checkpoint file/dir.
            teacher_load_path = teacher_model_path
            if not teacher_load_path.startswith(("/", ".", "~")):
                # Hugging Face repo id
                hf_teacher = AutoModelForCausalLM.from_pretrained(teacher_load_path)
                teacher_model.load_state_dict(hf_teacher.state_dict(), strict=False)
                del hf_teacher
            else:
                if os.path.isdir(teacher_load_path):
                    model_file = os.path.join(teacher_load_path, "model.pt")
                    if os.path.exists(model_file):
                        teacher_load_path = model_file
                if teacher_load_path not in ("None", "none", "") and os.path.exists(teacher_load_path):
                    teacher_weights = torch.load(teacher_load_path, map_location=torch.device(rank), weights_only=False)

                    # Handle base_causallm keys if it's a looped checkpoint
                    if any(k.startswith("base_causallm") for k in teacher_weights.keys()):
                        teacher_weights = {
                            k.replace("base_causallm.", ""): v
                            for k, v in teacher_weights.items()
                            if k.startswith("base_causallm")
                        }

                    teacher_model.load_state_dict(teacher_weights, strict=False)
                elif rank == 0:
                    print(f"No teacher checkpoint file found at '{teacher_model_path}', using base HF model as teacher")

            teacher_model = teacher_model.to(rank)

            if rank == 0:
                print(f"Teacher model loaded, codi_loss_weight={codi_loss_weight}, codi_loss_type={codi_loss_type}")

        model = Lotus(
            model,
            latent_id,
            start_id,
            end_id,
            tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            c_thought=configs.c_thought,
            use_kv_cache=use_kv_cache,
            intermediate_loss_weight=intermediate_loss_weight,
            freeze_supervised_blocks=freeze_supervised_blocks,
            intermediate_loss_type=intermediate_loss_type,
            intermediate_label_smoothing=intermediate_label_smoothing,
            teacher_model=teacher_model,
            codi_loss_weight=codi_loss_weight,
            codi_loss_type=codi_loss_type,
            inter_answer_loss_weight=getattr(configs, "inter_answer_loss_weight", 0.0),
            ia_no_prefix=getattr(configs, "ia_no_prefix", False),
            ia_decoder_layers=getattr(configs, "ia_decoder_layers", 0),
            ia_decoder_use_base_copy=getattr(configs, "ia_decoder_use_base_copy", False),
            intermediate_loss_after_loop=getattr(configs, "intermediate_loss_after_loop", False),
            flat_intermediate_supervision=getattr(configs, "flat_intermediate_supervision", False),
            ia_loss_after_loop=getattr(configs, "ia_loss_after_loop", False),
            latent_injection_mode=getattr(configs, "latent_injection_mode", "add"),
        )

    # If resuming AND the checkpoint contains ia_decoder weights, we need the
    # ia_decoder module to exist BEFORE load_state_dict so the keys have
    # somewhere to land. Create it now; load_state_dict will overwrite the
    # fresh weights with the trained ones from the checkpoint.
    _has_ia_keys = saved_weights is not None and any(
        k.startswith("ia_decoder") for k in saved_weights
    )
    _want_base_copy = (
        configs.looped
        and getattr(configs, "ia_decoder_use_base_copy", False)
        and hasattr(model, "init_ia_decoder_base_copy")
    )
    _want_external = (
        configs.looped
        and getattr(configs, "ia_decoder_init_path", None) is not None
        and hasattr(model, "init_ia_decoder_from_path")
    )
    _ia_pre_created = False
    if _has_ia_keys and (_want_base_copy or _want_external):
        if rank == 0:
            print(
                "Checkpoint contains ia_decoder.* keys — pre-creating IA decoder module "
                "so load_state_dict can populate it with trained weights."
            )
        if _want_base_copy:
            model.init_ia_decoder_base_copy()
        elif _want_external:
            model.init_ia_decoder_from_path(getattr(configs, "ia_decoder_init_path"))
        _ia_pre_created = True

    if configs.load_model_path is not None and configs.load_model_path != "None" and not loaded:
        print(model.load_state_dict(saved_weights, strict=False))

    # Initialize the base-copy IA decoder AFTER load_state_dict so the deep
    # copy inherits the loaded checkpoint weights. Skip if already pre-created
    # above (resume case where checkpoint had trained ia_decoder weights).
    if (
        configs.looped
        and getattr(configs, "ia_decoder_use_base_copy", False)
        and hasattr(model, "init_ia_decoder_base_copy")
    ):
        if _ia_pre_created:
            if rank == 0:
                print(
                    "IA decoder was pre-created and populated from checkpoint — "
                    "skipping re-initialization."
                )
        else:
            if rank == 0:
                print(
                    "Initializing IA decoder as a deep copy of the loaded base model "
                    "(ia_decoder_use_base_copy=True): same size, trainable, independent params."
                )
            model.init_ia_decoder_base_copy()

    # Initialize an external IA decoder (loaded from a separate HF path)
    # AFTER load_state_dict so it doesn't get clobbered. Skip if pre-created.
    _ia_init_path = getattr(configs, "ia_decoder_init_path", None)
    if (
        configs.looped
        and _ia_init_path is not None
        and hasattr(model, "init_ia_decoder_from_path")
    ):
        if _ia_pre_created:
            if rank == 0:
                print(
                    "IA decoder was pre-created and populated from checkpoint — "
                    "skipping re-initialization from external path."
                )
        else:
            if rank == 0:
                print(
                    f"Initializing IA decoder from external pretrained checkpoint: {_ia_init_path} "
                    f"(separate from main model; learnable input projection inserted if hidden sizes differ)."
                )
            model.init_ia_decoder_from_path(_ia_init_path)

    print(f"Running FSDP on rank = {rank}, world size = {world_size}")
    model = model.to(rank)

    llama_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            # GPT2Block,       # for GPT2, we don't need to shard layers (it becomes DDP)
            LlamaDecoderLayer  # only shard llama's layers.
        },
    )

    if configs.bf16:
        model.to(torch.bfloat16)

    # if only eval, use ddp (to avoid bugs in fsdp)
    if configs.only_eval:
        parallel_model = DDP(model, device_ids=[rank])

    else:
        parallel_model = FSDP(
            model, auto_wrap_policy=llama_auto_wrap_policy, device_id=rank,
            use_orig_params=False,
        )

    del model

    if rank == 0:
        print(parallel_model)

    # prepare the ground truth answer and cot for evaluation
    question_val = [d["question"] for d in json.load(open(configs.val_path))]
    answers_val = [
        d["answer"].replace(",", "").strip() for d in json.load(open(configs.val_path))
    ]
    cot_val = ["\n".join(d["steps"]) for d in json.load(open(configs.val_path))]

    base_dataset_valid = get_dataset(
        configs.val_path, tokenizer, max_size=32 if configs.debug else 100000000
    )

    if not configs.only_eval:
        base_dataset_train = get_dataset(
            configs.train_path, tokenizer, max_size=5000 if configs.debug else 100000000
        )

        # Automatically set c_thought to max CoT step length if auto_c_thought is enabled
        if getattr(configs, "auto_c_thought", False):
            max_step_len = 0
            for sample in base_dataset_train:
                for step_tokens in sample["steps_tokenized"]:
                    max_step_len = max(max_step_len, len(step_tokens))
            # Apply max_c_thought cap if specified
            max_c_thought = getattr(configs, "max_c_thought", None)
            if max_c_thought is not None and max_c_thought > 0:
                original_max = max_step_len
                max_step_len = min(max_step_len, max_c_thought)
                if rank == 0:
                    print(f"[auto_c_thought] Max step length in dataset: {original_max}, capped to max_c_thought: {max_c_thought}")
            configs.c_thought = max_step_len
            if rank == 0:
                print(f"[auto_c_thought] Set c_thought to {max_step_len}")

    if "gsm" in configs.val_path:
        max_new_tokens = 64
    else:
        max_new_tokens = 128

    total_train_steps = 0

    if not configs.debug and not configs.only_eval and rank == 0:
        wandb_entity = getattr(configs, 'entity', None)
        wandb_run = wandb.init(project=configs.project, name=configs.name, entity=wandb_entity, save_code=True)
        wandb_run.config.update(configs, allow_val_change=True)
        # Upload code files (.py and .yaml)
        wandb_run.log_code(".", include_fn=lambda path: path.endswith(".py") or path.endswith(".yaml"))
        # text_table = wandb.Table(columns=["step", "text"])

    else:
        wandb_run = None

    if configs.reset_optimizer:
        optimizer = None

    else:
        optimizer = optim.AdamW(
            parallel_model.parameters(),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )

    # LR scheduler (cosine with warmup, matching CODI/SIM-CoT)
    lr_scheduler = None
    warmup_ratio = getattr(configs, "warmup_ratio", 0.0)
    lr_schedule = getattr(configs, "lr_schedule", "constant")
    # For "curriculum" schedule: warmup LR during stage ramp-up, decay after max stage
    lr_min_ratio = getattr(configs, "lr_min_ratio", 0.1)  # min LR = lr * lr_min_ratio

    best_acc = 0

    # Determine if we're resuming from a full checkpoint (directory with optimizer/scheduler state)
    resume_checkpoint_dir = None
    if configs.resume != 0 and configs.load_model_path and configs.load_model_path != "None":
        # Check if the load_model_path (or its parent for file-based) is a directory checkpoint
        ckpt_path = configs.load_model_path
        if os.path.isdir(ckpt_path) and os.path.exists(os.path.join(ckpt_path, "optimizer.pt")):
            resume_checkpoint_dir = ckpt_path
        elif os.path.isfile(ckpt_path):
            # Old format: single file checkpoint — no optimizer state to load
            pass

    # Load optimizer state from checkpoint if available
    if resume_checkpoint_dir is not None and optimizer is not None:
        optim_path = os.path.join(resume_checkpoint_dir, "optimizer.pt")
        if rank == 0:
            print(f"Loading optimizer state from {optim_path}")
            full_osd = torch.load(optim_path, map_location="cpu", weights_only=False)
        else:
            full_osd = {}
        optim_save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(parallel_model, StateDictType.FULL_STATE_DICT, save_policy, optim_save_policy):
            optim_state_to_load = FSDP.optim_state_dict_to_load(parallel_model, optimizer, full_osd)
        optimizer.load_state_dict(optim_state_to_load)
        del full_osd, optim_state_to_load
        gc.collect()
        if rank == 0:
            print("Optimizer state loaded successfully")

    # Load training state (total_train_steps, best_acc, RNG states) if available
    if resume_checkpoint_dir is not None:
        training_state_path = os.path.join(resume_checkpoint_dir, "training_state.pt")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location="cpu", weights_only=False)
            total_train_steps = training_state.get("total_train_steps", 0)
            best_acc = training_state.get("best_acc", 0)
            # Restore RNG states for reproducibility
            if "rng_state" in training_state:
                random.setstate(training_state["rng_state"]["python"])
                np.random.set_state(training_state["rng_state"]["numpy"])
                torch.random.set_rng_state(training_state["rng_state"]["torch"])
                torch.cuda.set_rng_state(training_state["rng_state"]["cuda"])
            if rank == 0:
                print(f"Restored training state: total_train_steps={total_train_steps}, best_acc={best_acc}")
            del training_state

    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)

    def _make_loader(dataset, batch_size, sampler):
        # Shared DataLoader construction: num_workers/pin_memory/collate_fn are
        # identical across the train + eval loaders; shuffle is sampler-driven.
        return torch.utils.data.DataLoader(
            dataset,
            num_workers=1,
            pin_memory=True,
            batch_size=batch_size,
            collate_fn=collator,
            sampler=sampler,
        )

    for epoch in range(configs.resume, configs.num_epochs):

        scheduled_stage = (
            0 if (configs.cot or configs.no_cot) else min(epoch // configs.epochs_per_stage, configs.max_latent_stage)
        )

        # Choose dataset function based on whether we're using looped version
        use_looped = hasattr(configs, "looped") and configs.looped

        dataset_gen_val = get_question_latent_dataset(
            scheduled_stage,
            base_dataset_valid,
            configs,
            start_id,
            latent_id,
            end_id,
            no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
            single_latent_token=getattr(configs, "single_latent_token", False),
        )

        valid_gen_dataloader = _make_loader(
            dataset_gen_val,
            batch_size=1,
            sampler=DistributedSampler(dataset_gen_val, shuffle=False),
        )

        if not configs.only_eval:

            # Choose between cumulative (multi-task) or single-stage dataset
            use_cumulative = getattr(configs, "cumulative_stages", False)
            dataset_fn = get_cumulative_cot_latent_dataset if use_cumulative else get_cot_latent_dataset

            base_dataset_epoch = base_dataset_train

            dataset_train = dataset_fn(
                scheduled_stage,
                base_dataset_epoch,
                configs,
                start_id,
                latent_id,
                end_id,
                no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
                shuffle=True,
                single_latent_token=getattr(configs, "single_latent_token", False),
                shuffle_seed=configs.seed + 100003 * epoch,
            )

            train_sampler = DistributedSampler(dataset_train, shuffle=True, seed=configs.seed, drop_last=True)
            train_sampler.set_epoch(epoch)
            train_dataloader = _make_loader(
                dataset_train,
                batch_size=configs.batch_size_training,
                sampler=train_sampler,
            )

            # the sampler is deterministic even if shuffle is set to True
            # so we have shuffled the dataset when it's constructed (at every epoch).
            dataset_loss_val = get_cot_latent_dataset(
                scheduled_stage,
                base_dataset_valid,
                configs,
                start_id,
                latent_id,
                end_id,
                no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
                single_latent_token=getattr(configs, "single_latent_token", False),
            )

            valid_loss_dataloader = _make_loader(
                dataset_loss_val,
                batch_size=configs.batch_size_training,
                sampler=DistributedSampler(dataset_loss_val, shuffle=False),
            )

            if configs.reset_optimizer:
                del optimizer
                lr_scheduler = None

                optimizer = optim.AdamW(
                    parallel_model.parameters(),
                    lr=configs.lr,
                    weight_decay=configs.weight_decay,
                )

            # (Re-)create LR scheduler at the start of each epoch/stage
            # ── constant LR with linear warmup ──
            # When lr_schedule == "constant" and warmup_ratio > 0, ramp LR
            # from 0 to configs.lr over the first warmup_ratio fraction of
            # the *first* stage's steps, then hold constant. The scheduler
            # is created only once (epoch 0 / fresh) so warmup does NOT
            # re-fire at every stage transition. After warmup the multiplier
            # stays at 1.0 forever, so subsequent epochs see configs.lr.
            if lr_schedule == "constant" and warmup_ratio > 0 and lr_scheduler is None:
                steps_per_epoch_const = len(train_dataloader)
                # Anchor warmup length to the *first* epoch's step count so
                # warmup duration is independent of which stage we're in.
                warmup_steps = max(1, int(steps_per_epoch_const * warmup_ratio))
                from torch.optim.lr_scheduler import LambdaLR
                def _constant_with_warmup(current_step, warmup=warmup_steps):
                    if current_step < warmup:
                        return float(current_step) / float(warmup)
                    return 1.0
                lr_scheduler = LambdaLR(optimizer, _constant_with_warmup)
                if rank == 0:
                    print(f"Constant LR with warmup: warmup_steps={warmup_steps} (ratio={warmup_ratio})")

            if lr_schedule == "curriculum" and lr_scheduler is None:
                from torch.optim.lr_scheduler import LambdaLR
                import math
                # Phase 1: constant LR until max_latent_stage is reached
                # Phase 2: cosine decay from lr to lr * lr_min_ratio
                warmup_epochs = configs.max_latent_stage * configs.epochs_per_stage
                total_epochs = configs.num_epochs
                steps_per_epoch = len(train_dataloader)
                total_steps = steps_per_epoch * total_epochs
                decay_start_step = steps_per_epoch * warmup_epochs
                _curriculum_step_offset = epoch * steps_per_epoch

                def _curriculum_lr(current_step, _offset=_curriculum_step_offset,
                                   _decay_start=decay_start_step, _total=total_steps,
                                   _min_ratio=lr_min_ratio):
                    global_step = _offset + current_step
                    if global_step < _decay_start:
                        # Constant LR during stage ramp-up
                        return 1.0
                    else:
                        # Cosine decay: 1.0 → lr_min_ratio
                        progress = float(global_step - _decay_start) / float(max(1, _total - _decay_start))
                        return _min_ratio + (1.0 - _min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

                lr_scheduler = LambdaLR(optimizer, _curriculum_lr)
                if rank == 0:
                    print(f"Curriculum LR: constant for {warmup_epochs} epochs, "
                          f"then cosine decay to lr_min_ratio={lr_min_ratio}")

            # Load LR scheduler state from checkpoint on first resume epoch
            if resume_checkpoint_dir is not None and lr_scheduler is not None:
                training_state_path = os.path.join(resume_checkpoint_dir, "training_state.pt")
                if os.path.exists(training_state_path):
                    training_state = torch.load(training_state_path, map_location="cpu", weights_only=False)
                    if training_state.get("lr_scheduler") is not None:
                        lr_scheduler.load_state_dict(training_state["lr_scheduler"])
                        if rank == 0:
                            print(f"Restored LR scheduler state (last_epoch={lr_scheduler.last_epoch})")
                    del training_state
                # Only load once
                resume_checkpoint_dir = None

            parallel_model.module.train()

            total_length = len(train_dataloader)
            pbar = tqdm(
                colour="blue",
                desc=f"Training Epoch: {epoch+1}",
                total=total_length,
                dynamic_ncols=True,
            )

            for step, batch in enumerate(train_dataloader):

                # Logging commented out — skip the expensive token-by-token decode
                # to avoid rank 0 falling behind other ranks in FSDP collectives.

                batch = {
                    key: batch[key].to(rank) for key in batch.keys() if key not in ["idx", "answer_labels"]
                }

                # Add n_looped_iters parameter if using Lotus
                if use_looped:
                    # Use max of sample_stages if available (cumulative mode), else scheduled_stage
                    if "sample_stages" in batch:
                        batch["n_looped_iters"] = batch["sample_stages"].max().item()
                    else:
                        batch["n_looped_iters"] = scheduled_stage

                # For CoT with FSDP, don't pass labels to HF model — we compute
                # CE from logits ourselves, so avoid paying for it twice.
                if not configs.looped and dist.is_initialized():
                    train_labels = batch.pop("labels")
                else:
                    train_labels = None

                outputs = parallel_model(**batch)

                # ── FSDP-correct loss normalization ──────────────────────────────
                # Models return raw loss_sum (CE with reduction='sum') and n_valid
                # counts.  We normalize here, OUTSIDE the FSDP-wrapped forward, so
                # the all_reduce doesn't create a sync-point that breaks FSDP's
                # overlapped communication.
                #
                # Formula: loss = (W * loss_sum) / n_global
                #   where n_global = all_reduce(n_local) and W = world_size.
                # FSDP backward divides grad by W, yielding grad(loss_sum)/n_global
                # which is the correct per-token gradient.

                if not configs.looped:
                    # ── CoT training (plain HuggingFace model) ──
                    if dist.is_initialized():
                        logits = outputs.logits
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = train_labels[..., 1:].contiguous()
                        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
                        loss_sum = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        n_valid = (shift_labels.view(-1) != -100).sum().detach().clone().float()
                        dist.all_reduce(n_valid, op=dist.ReduceOp.SUM)
                        w = float(dist.get_world_size())
                        loss = (loss_sum * w) / n_valid.clamp(min=1)
                    else:
                        loss = outputs.loss

                elif dist.is_initialized():
                    # ── Looped / Lotus under FSDP ──
                    w = float(dist.get_world_size())

                    # Main LM loss
                    if hasattr(outputs, 'main_loss_sum'):
                        # Lotus
                        main_sum = outputs.main_loss_sum
                        n_main = outputs.n_main_valid.detach().clone().float()
                    else:
                        # Plain looped
                        main_sum = outputs.loss_sum
                        n_main = outputs.n_valid_tokens.detach().clone().float()

                    dist.all_reduce(n_main, op=dist.ReduceOp.SUM)
                    # Theory-ablation toggle: when use_answer_loss=False, the answer-token
                    # next-token-prediction CE is dropped entirely (only L_step / intermediate
                    # loss drives the model). Default True preserves the standard objective.
                    _use_ans_loss = getattr(configs, "use_answer_loss", True)
                    if _use_ans_loss:
                        loss = (main_sum * w) / n_main.clamp(min=1)
                    else:
                        # Graph-connected zero: stays in the autograd graph so backward()
                        # succeeds even when no other loss term contributes (e.g., stage 0
                        # before any latent slots are introduced). All gradients from this
                        # path are 0; later loss terms (intermediate, ...) add their own.
                        loss = main_sum * 0.0

                    # Intermediate loss (Lotus)
                    if hasattr(outputs, 'inter_loss_sum') and hasattr(outputs, 'n_inter_valid'):
                        n_inter = outputs.n_inter_valid.detach().clone().float()
                        dist.all_reduce(n_inter, op=dist.ReduceOp.SUM)
                        if n_inter > 0:
                            inter_loss = (outputs.inter_loss_sum * w) / n_inter.clamp(min=1)
                            inter_weight = getattr(configs, "intermediate_loss_weight", 0.0)
                            loss = loss + inter_weight * inter_loss

                    # CODI distillation loss (Lotus with teacher)
                    if hasattr(outputs, 'codi_loss_sum') and hasattr(outputs, 'n_codi_valid'):
                        n_codi = outputs.n_codi_valid.detach().clone().float()
                        dist.all_reduce(n_codi, op=dist.ReduceOp.SUM)
                        if n_codi > 0:
                            codi_loss = (outputs.codi_loss_sum * w) / n_codi.clamp(min=1)
                            codi_weight = getattr(configs, "codi_loss_weight", 0.0)
                            loss = loss + codi_weight * codi_loss

                    # Intermediate answer supervision (suffix forward at each loop step)
                    if hasattr(outputs, 'inter_answer_loss_sum') and hasattr(outputs, 'n_inter_answer_valid'):
                        n_inter_ans = outputs.n_inter_answer_valid.detach().clone().float()
                        dist.all_reduce(n_inter_ans, op=dist.ReduceOp.SUM)
                        if n_inter_ans > 0:
                            inter_ans_loss = (outputs.inter_answer_loss_sum * w) / n_inter_ans.clamp(min=1)
                            inter_ans_weight = getattr(configs, "inter_answer_loss_weight", 0.0)
                            loss = loss + inter_ans_weight * inter_ans_loss

                else:
                    # Single GPU — local mean is correct
                    loss = outputs.loss

                loss.backward()

                # Gradient norm clipping.
                # IMPORTANT: FSDP shards parameters across ranks, so each rank's
                # .grad is only a SHARD of the full gradient. Using
                # torch.nn.utils.clip_grad_norm_ would compute only the local-shard
                # norm (off by ~sqrt(W)) and clip inconsistently across ranks.
                # Use the FSDP module's .clip_grad_norm_() which all-reduces the
                # squared norm across shards to get the true global grad norm.
                grad_norm_clip = getattr(configs, "grad_norm_clip", 1.0)
                _clip_val = grad_norm_clip if grad_norm_clip > 0 else float('inf')
                if isinstance(parallel_model, FSDP):
                    grad_norm = parallel_model.clip_grad_norm_(_clip_val)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        parallel_model.parameters(), _clip_val
                    )
                    
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer.zero_grad()
                total_train_steps += 1
                pbar.update(1)

                if wandb_run and rank == 0:
                    _loss_display = loss.detach().float()
                    log_dict = {
                        "train/epoch": epoch + 1,
                        "train/stage": scheduled_stage,
                        "train/step": total_train_steps,
                        "train/loss": _loss_display,
                        "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        "train/lr": optimizer.param_groups[0]["lr"],
                    }
                    if use_looped:
                        log_dict["train/n_looped_iters"] = batch["n_looped_iters"]
                    # Log main_loss and intermediate_loss if available
                    if hasattr(outputs, 'main_loss') and outputs.main_loss is not None:
                        if _use_ans_loss:
                            main_loss_val = outputs.main_loss.detach().float() if isinstance(outputs.main_loss, torch.Tensor) else outputs.main_loss
                        else:
                            # Answer loss disabled by config: log 0 to make this explicit.
                            main_loss_val = 0.0
                        log_dict["train/main_loss"] = main_loss_val
                    for _attr in ("intermediate_loss", "codi_loss", "inter_answer_loss"):
                        _v = getattr(outputs, _attr, None)
                        if _v is not None:
                            log_dict[f"train/{_attr}"] = _v.detach().float() if isinstance(_v, torch.Tensor) else _v
                    wandb_run.log(log_dict)

                _ia_str = ""
                if hasattr(outputs, 'inter_answer_loss') and outputs.inter_answer_loss is not None:
                    _ia_val = outputs.inter_answer_loss.detach().float() if isinstance(outputs.inter_answer_loss, torch.Tensor) else outputs.inter_answer_loss
                    _ia_str = f" ia_loss: {round(float(_ia_val), 4)}"
                _loss_val = round(float(loss.detach().float()), 4)
                # if rank == 0 and step % 10 == 0:
                #     print(f"[E{epoch+1} S{scheduled_stage} B{step}] loss={_loss_val}{_ia_str}", flush=True)
                pbar.set_description(
                    f"Training Epoch: {epoch+1}/{configs.num_epochs} (Stage {scheduled_stage}), batch {step}/{len(train_dataloader)} "
                    f"completed (loss: {_loss_val}{_ia_str}"
                )
            pbar.close()
            dist.barrier()

            # Save periodic checkpoint (model + optimizer + scheduler + RNG states)
            save_every = getattr(configs, "save_every_epochs", 1)
            if not configs.debug and (epoch + 1) % save_every == 0:
                ckpt_dir = os.path.join(save_dir, f"checkpoint_{epoch + 1}")
                if rank == 0:
                    os.makedirs(ckpt_dir, exist_ok=True)

                save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                optim_save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(parallel_model, StateDictType.FULL_STATE_DICT, save_policy, optim_save_policy):
                    model_state = parallel_model.state_dict()
                    optim_state = FSDP.optim_state_dict(parallel_model, optimizer) if optimizer is not None else None

                if rank == 0:
                    torch.save(model_state, os.path.join(ckpt_dir, "model.pt"))
                    if optim_state is not None:
                        torch.save(optim_state, os.path.join(ckpt_dir, "optimizer.pt"))
                    training_state = {
                        "epoch": epoch + 1,
                        "total_train_steps": total_train_steps,
                        "best_acc": best_acc,
                        "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
                        "rng_state": {
                            "python": random.getstate(),
                            "numpy": np.random.get_state(),
                            "torch": torch.random.get_rng_state(),
                            "cuda": torch.cuda.get_rng_state(),
                        },
                    }
                    torch.save(training_state, os.path.join(ckpt_dir, "training_state.pt"))
                    print(f"Saved checkpoint to {ckpt_dir}")

                    # Remove previous checkpoint to save disk (keep only latest)
                    prev_ckpts = [
                        d for d in os.listdir(save_dir)
                        if d.startswith("checkpoint_") and d.split("_")[1].isdigit()
                        and d != f"checkpoint_{epoch + 1}"
                    ]
                    for old_ckpt in prev_ckpts:
                        old_path = os.path.join(save_dir, old_ckpt)
                        if os.path.isdir(old_path):
                            shutil.rmtree(old_path)
                            print(f"Removed old checkpoint {old_path}")

                dist.barrier()
                del model_state
                if optim_state is not None:
                    del optim_state
                gc.collect()
                torch.cuda.empty_cache()

            # Eval interval: run eval every N epochs (default 1 = every epoch)
            eval_interval = getattr(configs, "eval_interval", 1)
            skip_eval = not (
                (epoch + 1) % eval_interval == 0
                or (epoch + 1) == configs.num_epochs  # always eval on last epoch
                or configs.only_eval
            )

            if skip_eval:
                # Skip eval this epoch — just flush logs and continue
                if wandb_run and rank == 0:
                    wandb_run.log({}, commit=True)
                dist.barrier()
                gc.collect()
                torch.cuda.empty_cache()
                continue

            # val loss
            total_loss = 0
            total_main_loss = 0
            total_intermediate_loss = 0
            total_answer_loss = 0

            with torch.no_grad():
                parallel_model.module.eval()
                for step, batch in enumerate(valid_loss_dataloader):
                    
                    # Save answer_labels before moving to device
                    answer_labels = batch.get("answer_labels", None)

                    batch = {
                        key: batch[key].to(rank) for key in batch.keys() if key not in ["idx", "answer_labels"]
                    }

                    # Add n_looped_iters parameter if using Lotus
                    if use_looped:
                        # Use max of sample_stages if available (cumulative mode), else scheduled_stage
                        if "sample_stages" in batch:
                            batch["n_looped_iters"] = batch["sample_stages"].max().item()
                        else:
                            batch["n_looped_iters"] = scheduled_stage

                    outputs = parallel_model(**batch)
                    loss = outputs.loss
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    total_loss += loss.item() / world_size
                    
                    # Track main_loss and intermediate_loss separately
                    if hasattr(outputs, 'main_loss') and outputs.main_loss is not None:
                        main_loss = outputs.main_loss if isinstance(outputs.main_loss, torch.Tensor) else torch.tensor(outputs.main_loss, device=rank)
                        dist.all_reduce(main_loss, op=dist.ReduceOp.SUM)
                        total_main_loss += main_loss.item() / world_size
                    if hasattr(outputs, 'intermediate_loss') and outputs.intermediate_loss is not None:
                        inter_loss = outputs.intermediate_loss if isinstance(outputs.intermediate_loss, torch.Tensor) else torch.tensor(outputs.intermediate_loss, device=rank)
                        dist.all_reduce(inter_loss, op=dist.ReduceOp.SUM)
                        total_intermediate_loss += inter_loss.item() / world_size
                    
                    # Compute answer-only loss using answer_labels (for non-looped modes only)
                    # For Lotus, logits shape may not match full sequence due to KV caching
                    if answer_labels is not None and not use_looped:
                        answer_labels = answer_labels.to(rank)
                        logits = outputs.logits
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = answer_labels[..., 1:].contiguous()
                        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
                        answer_loss_sum = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        n_valid = (shift_labels.view(-1) != -100).sum().detach().clone()
                        dist.all_reduce(n_valid, op=dist.ReduceOp.SUM)
                        dist.all_reduce(answer_loss_sum, op=dist.ReduceOp.SUM)
                        total_answer_loss += (answer_loss_sum.item() / n_valid.clamp(min=1).item())

                if wandb_run and rank == 0:

                    log_dict = {
                        "eval/loss": total_loss / len(valid_loss_dataloader),
                    }
                    if total_main_loss > 0:
                        log_dict["eval/main_loss"] = total_main_loss / len(valid_loss_dataloader)
                    if total_intermediate_loss > 0:
                        log_dict["eval/intermediate_loss"] = total_intermediate_loss / len(valid_loss_dataloader)
                    if total_answer_loss > 0:
                        log_dict["eval/answer_only_loss"] = total_answer_loss / len(valid_loss_dataloader)
                    wandb_run.log(log_dict, commit=False)
                
                if rank == 0:
                    print(f"eval loss: {total_loss / len(valid_loss_dataloader):.4f}")
                    if total_answer_loss > 0:
                        print(f"eval answer-only loss: {total_answer_loss / len(valid_loss_dataloader):.4f}")

        # val generation accuracy
        total_length = len(valid_gen_dataloader)

        pbar = tqdm(
            colour="blue", desc=f"Test Accuracy", total=total_length, dynamic_ncols=True
        )
        n_test_samples = len(answers_val)
        test_results = torch.zeros(n_test_samples, device=rank)
        test_cot_results = torch.zeros(n_test_samples, device=rank)

        with torch.no_grad():
            parallel_model.module.eval()
            for idx, batch in enumerate(valid_gen_dataloader):
                test_idx = batch["idx"][0]

                batch = {
                    k: v.to(rank)
                    for k, v in batch.items()
                    if v != None and k not in ["idx", "position_ids"]
                }
                # https://github.com/huggingface/transformers/issues/32492

                assert len(batch["input_ids"]) == 1
                sample_idx = test_idx.cpu().item()
                answer = answers_val[sample_idx]
                answer_cot = cot_val[sample_idx]
                question = question_val[sample_idx]

                # synced_gpus=True in FSDP mode, as we need to keep # forward pass the same on each device
                # Add n_looped_iters parameter if using Lotus
                generate_kwargs = {
                    **batch,
                    "max_new_tokens": max_new_tokens,
                    "synced_gpus": not configs.only_eval,
                }
                if use_looped:
                    generate_kwargs["n_looped_iters"] = scheduled_stage

                if use_looped:
                    outputs, intermediate_tokens = parallel_model.module.generate(output_intermediate=True, **generate_kwargs)
                else:
                    outputs = parallel_model.module.generate(**generate_kwargs)
                    intermediate_tokens = []

                text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer_output = text_output.split("#")[-1].replace(",", "").strip()
                cot_output = (
                    ("\n".join(text_output.split("\n")[1:])).split("#")[0].strip()
                )

                if idx < 5 and rank == 0:
                    # print some examples
                    print(
                        f"Question {test_idx}: Answer = '{answer}' CoT = '{answer_cot}'"
                    )
                    n_looped_iters_used = generate_kwargs.get('n_looped_iters', 0)
                    actual_loops = getattr(parallel_model.module if hasattr(parallel_model, 'module') else parallel_model, 'actual_loops_executed', 'unknown')
                    print(f"Full output (n_looped_iters={n_looped_iters_used}, actual={actual_loops}): '{tokenizer.decode(outputs[0])}'")
                    print(f"Extracted Output: '{answer_output}'")
                    
                    # Decode intermediate tokens (thoughts at each loop iteration)
                    if intermediate_tokens:
                        print(f"Intermediate thoughts ({len(intermediate_tokens)} loops):")
                        for loop_idx, step_tokens in enumerate(intermediate_tokens):
                            decoded_thought = tokenizer.decode(step_tokens, skip_special_tokens=True)
                            print(f"  Loop {loop_idx}: '{decoded_thought}'")

                test_results[sample_idx] = float(answer_output == answer)
                test_cot_results[sample_idx] = float(cot_output == answer_cot)

                pbar.update(1)
                pbar.set_description(
                    f"Test accuracy: {round(float(test_results.sum().item() / n_test_samples), 2)}"
                )

            pbar.close()
            print(f"Device {rank}: local_cor={int(test_results.sum().item())}, local_cot={int(test_cot_results.sum().item())}")

        # MAX so duplicated samples from DistributedSampler padding are counted once
        dist.all_reduce(test_results, op=dist.ReduceOp.MAX)
        dist.all_reduce(test_cot_results, op=dist.ReduceOp.MAX)

        cor = int(test_results.sum().item())
        cor_cot = int(test_cot_results.sum().item())
        total = n_test_samples
        if rank == 0:
            print(f"Accuracy on validation set: {cor} / {total} = {cor/total}")
            print(f"CoT match on validation set: {cor_cot} / {total} = {cor_cot/total}")
        sys.stdout.flush()

        if wandb_run:
            wandb_run.log({"eval/acc": cor / total, "eval/cot_em": cor_cot / total}, commit=False)

        # Best-model selection: save checkpoint_final when validation accuracy improves.
        if not configs.only_eval:
            val_acc = cor / total
            if val_acc > best_acc and not configs.debug:
                best_acc = val_acc
                save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(parallel_model, StateDictType.FULL_STATE_DICT, save_policy):
                    states = parallel_model.state_dict()
                    if rank == 0:
                        final_ckpt_path = os.path.join(save_dir, "checkpoint_final")
                        torch.save(states, final_ckpt_path)
                        print(f"New best val acc: {val_acc:.4f} — saved checkpoint_final")
                dist.barrier()
                del states
                gc.collect()
                torch.cuda.empty_cache()

        if configs.only_eval:
            break

        # Flush all commit=False logs for this epoch
        if wandb_run and rank == 0:
            wandb_run.log({}, commit=True)

        dist.barrier()
        gc.collect()
        torch.cuda.empty_cache()

    # checkpoint_final is saved during training when val acc improves (best model selection)
    if rank == 0 and not configs.only_eval and not configs.debug:
        final_path = os.path.join(save_dir, "checkpoint_final")
        if os.path.exists(final_path):
            print(f"Best val acc: {best_acc:.4f} — checkpoint_final already saved")
        else:
            print(f"Warning: no checkpoint_final saved (val acc never improved from {best_acc})")

    dist.barrier()


if __name__ == "__main__":
    main()
