"""
Standalone evaluation script for LOTUS and CoT models.
Evaluates on GSM8K, GSM-Hard, MultiArith, SVAMP.
"""
import argparse
import os
import sys
import json
import math
import time
import torch
import torch.distributed as dist
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# scripts/ dir on the path for sibling imports; data/ lives at the repo root (its parent).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)
from lotus import Lotus


def load_ood_dataset(name):
    """Load OOD dataset, returns list of (question, answer) tuples."""
    if name == "gsm-hard":
        ds = load_dataset("reasoning-machines/gsm-hard", split="train")
        data = []
        for ex in ds:
            q = ex["input"].strip()
            a = str(ex["target"]).strip().replace(",", "")
            if a.endswith(".0"):
                a = a[:-2]
            data.append((q, a))
        return data
    elif name == "multi-arith":
        ds = load_dataset("ChilleD/MultiArith", split="test")
        data = []
        for ex in ds:
            q = ex["question"].strip()
            a = str(ex["final_ans"]).strip().replace(",", "")
            if a.endswith(".0"):
                a = a[:-2]
            data.append((q, a))
        return data
    elif name == "svamp":
        ds = load_dataset("ChilleD/SVAMP")
        combined = concatenate_datasets([ds["train"], ds["test"]])
        data = []
        for ex in combined:
            q = ex["question_concat"].strip()
            a = str(ex["Answer"]).strip().replace(",", "")
            if a.endswith(".0"):
                a = a[:-2]
            data.append((q, a))
        return data
    elif name == "gsm8k":
        json_path = os.path.join(_REPO_ROOT, "data", "gsm_test.json")
        with open(json_path) as f:
            raw = json.load(f)
        return [(d["question"].strip(), d["answer"].replace(",", "").strip()) for d in raw]
    else:
        # Try loading from local JSON file
        with open(name) as f:
            raw = json.load(f)
        return [(d["question"], d["answer"].replace(",", "").strip()) for d in raw]


def extract_answer(text):
    """Extract final answer from model output like '... ### 42'"""
    import re
    # Try to find answer after ### first
    parts = text.split("###")
    if len(parts) > 1:
        ans = parts[-1].replace(",", "").strip()
        if ans:
            return ans
    # Fallback: find the last number in the text
    numbers = re.findall(r'-?[\d,]+\.?\d*', text)
    if numbers:
        return numbers[-1].replace(",", "").strip()
    return text.split("#")[-1].replace(",", "").strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="Path to a checkpoint_final torch state dict. If omitted, weights are taken "
                             "directly from --model_id (an HF-format model saved with save_pretrained).")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--datasets", nargs="+", default=["gsm-hard", "multi-arith", "svamp"],
                        help="Dataset names to evaluate")
    parser.add_argument("--n_looped_iters", type=int, default=6, help="Number of latent loops")
    parser.add_argument("--c_thought", type=int, default=25, help="Number of thought tokens per loop")
    parser.add_argument("--n_latent_override", type=int, default=None,
                        help="If set, use this exact number of latent positions in the input prefix (overrides n_looped_iters*c_thought).")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--fp32", action="store_true", default=False,
                        help="Force fp32 inference (overrides --bf16; needed for GPT-2 trained without bf16).")
    parser.add_argument("--cot", action="store_true", default=False,
                        help="Evaluate a CoT model (plain causal LM, no latent tokens)")
    parser.add_argument("--save_preds", default=None,
                        help="Optional path to dump per-example predictions (JSON).")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if args.fp32 else (torch.bfloat16 if args.bf16 else torch.float32)

    # Load tokenizer and add special tokens
    print(f"Loading tokenizer from {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    if not args.cot:
        tokenizer.add_tokens("<|start-latent|>")
        tokenizer.add_tokens("<|end-latent|>")
        tokenizer.add_tokens("<|latent|>")
        start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
        end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
        latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")

    # Load base model
    print(f"Loading base model {args.model_id}...")
    base_model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype)

    # If a checkpoint state dict is given, load it; otherwise the weights come
    # directly from --model_id (e.g. an HF-format model saved with save_pretrained).
    saved_weights = None
    if args.checkpoint is not None:
        ckpt_path = args.checkpoint
        if os.path.isdir(ckpt_path):
            model_file = os.path.join(ckpt_path, "model.pt")
            if os.path.exists(model_file):
                ckpt_path = model_file
        print(f"Loading checkpoint from {ckpt_path}...")
        saved_weights = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Check vocab size and resize if needed
        emb_key = None
        for k in saved_weights.keys():
            if 'embed_tokens.weight' in k or 'wte.weight' in k:
                emb_key = k
                break
        if emb_key:
            ckpt_vocab = saved_weights[emb_key].shape[0]
            model_vocab = base_model.get_input_embeddings().weight.shape[0]
            if ckpt_vocab != model_vocab:
                print(f"Resizing model vocab from {model_vocab} to {ckpt_vocab}")
                base_model.resize_token_embeddings(ckpt_vocab)
    else:
        # Weights already live in base_model (loaded from --model_id). Ensure the
        # embedding matrix covers the latent tokens added to the tokenizer.
        if not args.cot and base_model.get_input_embeddings().weight.shape[0] < len(tokenizer):
            base_model.resize_token_embeddings(len(tokenizer))
        print(f"Using weights directly from {args.model_id} (no separate checkpoint).")

    if args.cot:
        # CoT model: plain causal LM, no Lotus wrapper
        if saved_weights is not None:
            info = base_model.load_state_dict(saved_weights, strict=False)
            if info.missing_keys:
                print(f"Warning: missing keys: {info.missing_keys}")
            print(f"<All keys matched successfully>" if not info.unexpected_keys else f"Unexpected: {info.unexpected_keys}")
        model = base_model.to(device).to(dtype)
    else:
        # Lotus model
        model = Lotus(
            base_model,
            latent_token_id=latent_id,
            start_latent_id=start_id,
            end_latent_id=end_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            c_thought=args.c_thought,
        )

        if saved_weights is not None:
            info = model.load_state_dict(saved_weights, strict=False)
            if info.missing_keys:
                real_missing = [k for k in info.missing_keys if not k.startswith("teacher")]
                if real_missing:
                    print(f"Warning: missing keys: {real_missing}")
            print(f"<All keys matched successfully>" if not info.unexpected_keys else f"Unexpected: {info.unexpected_keys}")
            print("Skipping latent token initialization (using trained embeddings from checkpoint)")
        model = model.to(device).to(dtype)
    model.eval()

    # Evaluate each dataset
    results = {}
    total_start_time = time.time()

    for ds_name in args.datasets:
        mode_str = "cot" if args.cot else f"n_looped_iters={args.n_looped_iters}"
        print(f"\n{'='*60}")
        print(f"Evaluating on {ds_name} with {mode_str}")
        print(f"{'='*60}")

        data = load_ood_dataset(ds_name)
        print(f"Loaded {len(data)} examples")

        correct = 0
        total = 0
        total_gen_tokens = 0
        total_query_prefill_time = 0.0
        total_thought_time = 0.0
        total_eot_marker_time = 0.0
        total_answer_time = 0.0
        ds_start_time = time.time()
        per_example_preds = []
        # Reset peak memory tracking before this dataset
        torch.cuda.reset_peak_memory_stats()

        # Token IDs for marker detection
        hash_triple_id = tokenizer.encode("###", add_special_tokens=False)[0]  # 14711
        hash_single_id = tokenizer.encode("#", add_special_tokens=False)[0]    # 2
        space_id = tokenizer.encode(" ", add_special_tokens=False)[-1]         # 220
        eot_marker_ids = {hash_triple_id, hash_single_id, space_id}

        with torch.no_grad():
            pbar = tqdm(data, desc=f"{ds_name}")
            for i, (question, answer) in enumerate(pbar):
                q_tokens = tokenizer.encode(question + "\n", add_special_tokens=True)

                if args.cot:
                    # CoT: just the question, no latent tokens
                    input_ids = torch.tensor([q_tokens], device=device)
                    attention_mask = torch.ones_like(input_ids)
                else:
                    # Lotus: question + latent tokens
                    if args.n_latent_override is not None:
                        n_latent_tokens = args.n_latent_override
                    else:
                        n_latent_tokens = args.n_looped_iters * args.c_thought
                    latent_tokens = [start_id] + [latent_id] * n_latent_tokens + [end_id]
                    input_ids = torch.tensor([q_tokens + latent_tokens], device=device)
                    attention_mask = torch.ones_like(input_ids)

                try:
                    if args.cot:
                        # Token-by-token generation with per-token timing
                        # Phases: THOUGHT -> EOT_MARKER -> ANSWER
                        # 1. Query prefill
                        torch.cuda.synchronize()
                        t0 = time.time()
                        prefill_out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
                        torch.cuda.synchronize()
                        total_query_prefill_time += time.time() - t0

                        # 2. Decode token by token with per-token timing
                        kv_cache = prefill_out.past_key_values
                        next_token = torch.argmax(prefill_out.logits[0, -1]).item()
                        gen_token_ids = [next_token]
                        seq_len = input_ids.shape[1]
                        # Per-token times: (token_id, generation_time)
                        _first_tok_time = time.time()
                        token_gen_times = [(_first_tok_time, next_token, 0.0)]  # first token is ~free (argmax)

                        for _ in range(args.max_new_tokens - 1):
                            if next_token == tokenizer.eos_token_id:
                                break
                            _iter_start = time.time()
                            new_input = torch.tensor([[next_token]], device=device)
                            out = model(input_ids=new_input, past_key_values=kv_cache,
                                        position_ids=torch.tensor([[seq_len]], device=device),
                                        use_cache=True)
                            kv_cache = out.past_key_values
                            seq_len += 1
                            next_token = torch.argmax(out.logits[0, -1]).item()  # .item() implicitly syncs
                            token_gen_times.append((_iter_start, next_token, time.time() - _iter_start))
                            gen_token_ids.append(next_token)

                        # Classify tokens into phases: thought -> eot_marker -> answer
                        # "###" (token 14711) marks the boundary
                        phase = "thought"
                        ex_thought_time = 0.0
                        ex_eot_marker_time = 0.0
                        ex_answer_time = 0.0
                        for _, tok_id, gen_time in token_gen_times:
                            if phase == "thought":
                                if tok_id == hash_triple_id or tok_id == hash_single_id:
                                    phase = "eot_marker"
                                    ex_eot_marker_time += gen_time
                                else:
                                    ex_thought_time += gen_time
                            elif phase == "eot_marker":
                                if tok_id in eot_marker_ids:
                                    ex_eot_marker_time += gen_time
                                else:
                                    phase = "answer"
                                    ex_answer_time += gen_time
                            else:  # answer
                                ex_answer_time += gen_time

                        total_thought_time += ex_thought_time
                        total_eot_marker_time += ex_eot_marker_time
                        total_answer_time += ex_answer_time

                        outputs = torch.tensor([input_ids[0].tolist() + gen_token_ids], device=device)
                    else:
                        outputs = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=args.max_new_tokens,
                            n_looped_iters=args.n_looped_iters,
                        )
                        total_query_prefill_time += model._last_timing["query_prefill"]
                        total_thought_time += model._last_timing["thought"]

                        # Split decode into eot_marker vs answer using per-token times
                        decode_token_times = model._last_decode_token_times
                        past_marker = False
                        ex_eot_marker_time = 0.0
                        ex_answer_time = 0.0
                        for tok_id, gen_time in decode_token_times:
                            if not past_marker and tok_id in eot_marker_ids:
                                ex_eot_marker_time += gen_time
                            else:
                                past_marker = True
                                ex_answer_time += gen_time
                        total_eot_marker_time += ex_eot_marker_time
                        total_answer_time += ex_answer_time

                    # Only decode the generated portion (after input)
                    input_len = input_ids.shape[1]
                    gen_tokens = outputs[0][input_len:]
                    n_gen = len(gen_tokens)
                    total_gen_tokens += n_gen
                    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                    pred = extract_answer(gen_text)

                    if i < 3:
                        print(f"\nQ: {question[:80]}...")
                        print(f"Gen: '{gen_text}'")
                        print(f"Pred: '{pred}' | GT: '{answer}'")

                    # Compare numerically
                    is_correct = False
                    try:
                        pred_num = float(pred)
                        ans_num = float(answer)
                        if pred_num == ans_num:
                            correct += 1
                            is_correct = True
                    except (ValueError, ZeroDivisionError):
                        if pred.strip() == answer.strip():
                            correct += 1
                            is_correct = True
                    per_example_preds.append({
                        "idx": i,
                        "question": question,
                        "pred": pred,
                        "answer": answer,
                        "correct": is_correct,
                    })

                except Exception as e:
                    if i < 3:
                        print(f"Error on example {i}: {e}")

                total += 1
                pbar.set_description(f"{ds_name} acc={correct/total:.3f}")

        ds_elapsed = time.time() - ds_start_time
        acc = correct / total if total > 0 else 0
        avg_gen_len = total_gen_tokens / total if total > 0 else 0
        results[ds_name] = {"accuracy": acc, "correct": correct, "total": total,
                            "avg_gen_tokens": avg_gen_len, "time": ds_elapsed,
                            "query_prefill_time": total_query_prefill_time,
                            "thought_time": total_thought_time,
                            "eot_marker_time": total_eot_marker_time,
                            "answer_time": total_answer_time,
                            "per_example": per_example_preds}
        print(f"\n{ds_name}: {correct}/{total} = {acc*100:.2f}%")
        print(f"  avg output length: {avg_gen_len:.1f} tokens")
        print(f"  time: {ds_elapsed:.1f}s ({ds_elapsed/total:.2f}s/example)")
        print(f"  query prefill: {total_query_prefill_time:.1f}s ({total_query_prefill_time/total*1000:.1f}ms/example)")
        print(f"  thought:       {total_thought_time:.1f}s ({total_thought_time/total*1000:.1f}ms/example)")
        print(f"  eot marker:    {total_eot_marker_time:.1f}s ({total_eot_marker_time/total*1000:.1f}ms/example)")
        print(f"  answer:        {total_answer_time:.1f}s ({total_answer_time/total*1000:.1f}ms/example)")
        total_inference = total_query_prefill_time + total_thought_time + total_eot_marker_time + total_answer_time
        print(f"  total infer:   {total_inference:.1f}s ({total_inference/total*1000:.1f}ms/example)")
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
        peak_mem_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)
        results[ds_name]["peak_mem_alloc_gb"] = peak_mem_gb
        results[ds_name]["peak_mem_reserved_gb"] = peak_mem_reserved_gb
        print(f"  peak GPU mem:  allocated={peak_mem_gb:.3f} GB, reserved={peak_mem_reserved_gb:.3f} GB")

    total_elapsed = time.time() - total_start_time

    # Summary
    print(f"\n{'='*60}")
    mode_str = "cot" if args.cot else f"n_looped_iters={args.n_looped_iters}, c_thought={args.c_thought}"
    print(f"SUMMARY ({mode_str})")
    print(f"{'='*60}")
    for ds_name, res in results.items():
        total_infer = res['query_prefill_time'] + res['thought_time'] + res['eot_marker_time'] + res['answer_time']
        print(f"  {ds_name:15s}: {res['correct']:4d}/{res['total']:4d} = {res['accuracy']*100:.2f}%  "
              f"avg_len={res['avg_gen_tokens']:.1f}  "
              f"q_prefill={res['query_prefill_time']:.1f}s  thought={res['thought_time']:.1f}s  "
              f"eot_marker={res['eot_marker_time']:.1f}s  answer={res['answer_time']:.1f}s  "
              f"total={total_infer:.1f}s")
    print(f"  {'total time':15s}: {total_elapsed:.1f}s")

    if args.save_preds:
        # Dump per-example predictions across all datasets
        dump = {ds: res.get("per_example", []) for ds, res in results.items()}
        with open(args.save_preds, "w") as f:
            json.dump(dump, f)
        print(f"Saved per-example predictions to {args.save_preds}")


if __name__ == "__main__":
    main()
