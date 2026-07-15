# Adapted from Coconut (https://github.com/facebookresearch/coconut).
# Original code Copyright (c) Meta Platforms, Inc. and affiliates., MIT License.

import copy
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.models.gpt2 import GPT2LMHeadModel

Outputs = namedtuple("Outputs", [
    "loss", "inputs_embeds", "logits", "intermediate_logits",
    "main_loss", "intermediate_loss",
    "main_loss_sum", "n_main_valid",
    "inter_loss_sum", "n_inter_valid",
    "codi_loss", "codi_loss_sum", "n_codi_valid",
    "inter_answer_loss", "inter_answer_loss_sum", "n_inter_answer_valid",
])


class LightweightDecoder(nn.Module):
    """Small causal transformer decoder for IA supervision.
    
    Takes [latent_emb, CoT_token_embeds] and predicts CoT tokens via NTP.
    Much cheaper than a full LLM copy — just a few transformer layers + lm_head.
    """

    def __init__(self, hidden_size, vocab_size, n_layers=2, n_heads=8, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm for stability
        )
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, inputs_embeds):
        """
        Args:
            inputs_embeds: (batch, seq_len, hidden) — [latent_emb, CoT0, CoT1, ...]
        Returns:
            logits: (batch, seq_len, vocab)
        """
        seq_len = inputs_embeds.shape[1]
        # Causal mask: prevent attending to future positions
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=inputs_embeds.device, dtype=inputs_embeds.dtype,
        )
        hidden = self.layers(inputs_embeds, mask=causal_mask, is_causal=True)
        hidden = self.ln_f(hidden)
        return self.lm_head(hidden)


class BaseCopyIADecoder(nn.Module):
    """Auxiliary IA decoder that is a full-size, trainable copy of the base CausalLM.

    Used when ``ia_decoder_use_base_copy=True``: a deep copy of the loaded base
    model is wrapped here and used as the auxiliary NTP decoder for the IA loss.
    Same architecture and size as the answer-generating model, but with
    independent parameters so gradients on the IA path do not flow into the
    main answer model.

    Exposes the same call signature as ``LightweightDecoder``: takes
    ``inputs_embeds`` of shape ``(batch, seq_len, hidden)`` and returns logits
    of shape ``(batch, seq_len, vocab)``. Internally runs a from-scratch causal
    forward (no prefix, no past KV cache, ``use_cache=False``).
    """

    def __init__(self, base_model):
        super().__init__()
        # base_model is expected to already be a deep copy (caller's
        # responsibility), so its parameters are independent of the main model.
        self.model = base_model

    def forward(self, inputs_embeds):
        batch_size, seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device
        position_ids = (
            torch.arange(0, seq_len, device=device, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        attention_mask = torch.ones(
            batch_size, seq_len, device=device, dtype=torch.long,
        )
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
        )
        return outputs.logits


class ExternalIADecoder(nn.Module):
    """Auxiliary IA decoder loaded from a separate pretrained checkpoint.

    Unlike ``BaseCopyIADecoder`` (which deep-copies the loaded main model),
    this loads an arbitrary HF causal LM (e.g., Llama-3.2-1B) as the aux
    decoder. If the aux model's hidden size differs from the main model,
    a learnable input projection is inserted so the main model's post-loop
    latents can be fed to the aux model. Vocabulary must match the main
    model's tokenizer (asserted at construction time).

    Exposes the same call signature as ``LightweightDecoder`` /
    ``BaseCopyIADecoder``: takes ``inputs_embeds`` of shape
    ``(batch, seq_len, main_hidden)`` and returns logits of shape
    ``(batch, seq_len, vocab)``.
    """

    def __init__(self, aux_model, main_hidden_size):
        super().__init__()
        self.model = aux_model
        aux_hidden_size = aux_model.config.hidden_size
        if main_hidden_size == aux_hidden_size:
            self.in_proj = nn.Identity()
        else:
            self.in_proj = nn.Linear(main_hidden_size, aux_hidden_size, bias=False)
            nn.init.normal_(self.in_proj.weight, std=0.02)
            # Match aux model dtype so projection output matches what the aux
            # model expects (e.g., bf16 for bf16 training).
            self.in_proj = self.in_proj.to(dtype=aux_model.dtype)
        self.main_hidden_size = main_hidden_size
        self.aux_hidden_size = aux_hidden_size

    def forward(self, inputs_embeds):
        batch_size, seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device
        projected = self.in_proj(inputs_embeds)
        position_ids = (
            torch.arange(0, seq_len, device=device, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        attention_mask = torch.ones(
            batch_size, seq_len, device=device, dtype=torch.long,
        )
        outputs = self.model(
            inputs_embeds=projected,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
        )
        return outputs.logits


def efficient_forward(model, inputs_embeds, attention_mask, position_ids, past_key_values=None, is_gpt2=True):
    """Memory-efficient forward returning only last hidden state + logits + KV cache.

    Avoids storing all intermediate layer hidden states (~95% activation-memory
    saving vs. output_hidden_states=True).
    """
    base_model = model

    if is_gpt2:
        # GPT2: call transformer directly
        transformer = base_model.transformer
        lm_head = base_model.lm_head
        transformer_outputs = transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        hidden_states = transformer_outputs.last_hidden_state
        logits = lm_head(hidden_states)
        past_key_values = transformer_outputs.past_key_values
    else:
        # Llama: call model.model directly
        llama_model = base_model.model
        lm_head = base_model.lm_head
        # Ensure past_key_values is a DynamicCache, not None/tuple. LlamaModel
        # converts None -> DynamicCache internally but then sets
        # return_legacy_cache=True, returning a tuple, which breaks callers that
        # pass the cache to LlamaForCausalLM (expects a Cache object).
        if past_key_values is None:
            from transformers import DynamicCache
            past_key_values = DynamicCache()
        elif isinstance(past_key_values, tuple):
            from transformers import DynamicCache
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        model_outputs = llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        hidden_states = model_outputs.last_hidden_state
        logits = lm_head(hidden_states)
        past_key_values = model_outputs.past_key_values

    class OutputWrapper:
        def __init__(self, logits, last_hidden_state, past_key_values):
            self.logits = logits
            self.last_hidden_state = last_hidden_state
            self.past_key_values = past_key_values

    return OutputWrapper(logits, hidden_states, past_key_values)


def _clone_kv_cache(cache):
    """Shallow-copy a DynamicCache so in-place .update() calls don't mutate the
    original prefix cache across loop iterations. List containers are copied;
    tensor refs are shared (cheap). tuple-based caches (GPT2) are immutable."""
    if cache is None:
        return None
    try:
        from transformers import DynamicCache as _DC
        if isinstance(cache, _DC):
            clone = _DC()
            clone.key_cache = list(cache.key_cache)
            clone.value_cache = list(cache.value_cache)
            clone._seen_tokens = cache._seen_tokens
            return clone
    except ImportError:
        pass
    return cache


class Lotus(nn.Module):

    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
        pad_token_id=None,  # For masking padding in intermediate supervision
        c_thought=1,  # Number of latent tokens per reasoning step
        use_kv_cache=True,  # Enable/disable KV cache during loops
        intermediate_loss_weight=0.0,  # Weight for intermediate supervision loss
        freeze_supervised_blocks=False,  # If True, don't update embeddings for blocks that have been supervised
        intermediate_loss_type="crossentropy",  # 'crossentropy' or 'cosine'
        intermediate_label_smoothing=0.0,  # Label smoothing for intermediate cross-entropy loss
        teacher_model=None,  # Frozen teacher model for CODI distillation
        codi_loss_weight=0.0,  # Weight for CODI-style answer-boundary distillation loss
        codi_loss_type="smooth_l1",  # 'smooth_l1', 'mse', or 'cosine' for CODI loss
        inter_answer_loss_weight=0.0,  # Weight for intermediate answer supervision (suffix forward at each loop step)
        ia_no_prefix=False,  # If True, IA forward sees only [latent_emb, CoT] with no prefix KV cache (SIM-CoT style)
        ia_decoder_layers=0,  # If > 0, use a lightweight auxiliary decoder with this many layers instead of the main model
        ia_decoder_use_base_copy=False,  # If True, IA decoder is a deep copy of the loaded base model (same size, trainable, independent params); call init_ia_decoder_base_copy() AFTER load_state_dict to populate it
        intermediate_loss_after_loop=False,  # If True, supervise all blocks after final iteration (not during)
        flat_intermediate_supervision=False,  # If True, pack CoT tokens contiguously across all latent slots
        ia_loss_after_loop=False,  # If True, compute IA loss once after all loops using all latent hidden states
        latent_injection_mode="add",  # "add" (default, current): embed_t = original + h_{t-1}; "replace": embed_t = h_{t-1} (drops residual to test recurrent variance amplification)
    ):

        super(Lotus, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.c_thought = c_thought
        self.intermediate_loss_weight = intermediate_loss_weight
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.use_kv_cache = use_kv_cache  # Control KV cache usage
        self.freeze_supervised_blocks = freeze_supervised_blocks
        self.intermediate_loss_type = intermediate_loss_type
        self.intermediate_label_smoothing = intermediate_label_smoothing
        self.codi_loss_weight = codi_loss_weight
        self.codi_loss_type = codi_loss_type
        self.inter_answer_loss_weight = inter_answer_loss_weight
        self.ia_no_prefix = ia_no_prefix
        self.ia_decoder_layers = ia_decoder_layers
        self.ia_decoder_use_base_copy = ia_decoder_use_base_copy
        self.ia_decoder = None  # initialized after model detection (lightweight) or after load_state_dict (base-copy)
        self.intermediate_loss_after_loop = intermediate_loss_after_loop
        self.flat_intermediate_supervision = flat_intermediate_supervision
        self.ia_loss_after_loop = ia_loss_after_loop
        assert latent_injection_mode in ("add", "replace"), f"unknown latent_injection_mode: {latent_injection_mode}"
        self.latent_injection_mode = latent_injection_mode
        if teacher_model is not None:
            for param in teacher_model.parameters():
                param.requires_grad = False
            teacher_model.eval()
        object.__setattr__(self, 'teacher_model', teacher_model)
        assert self.use_kv_cache is True, "Currently only support use_kv_cache=True"

        # Detect model type
        self._is_gpt2 = isinstance(self.base_causallm, GPT2LMHeadModel)
        self._is_llama = hasattr(self.base_causallm, 'model') and hasattr(self.base_causallm.model, 'layers')  # Llama structure
        
        # Get embedding layer
        if self._is_gpt2:
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

        # Initialize lightweight auxiliary decoder for IA supervision if requested
        if self.ia_decoder_layers > 0:
            assert not self.ia_decoder_use_base_copy, (
                "ia_decoder_layers > 0 (lightweight decoder) and ia_decoder_use_base_copy=True "
                "(full base-model copy) are mutually exclusive; pick one."
            )
            hidden_size = self.embedding.embedding_dim
            vocab_size = self.embedding.num_embeddings
            n_heads = 8 if hidden_size >= 512 else 4  # sensible default
            self.ia_decoder = LightweightDecoder(
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                n_layers=self.ia_decoder_layers,
                n_heads=n_heads,
            )
        # NOTE: when ia_decoder_use_base_copy=True, self.ia_decoder is left as
        # None here; the caller (e.g. run.py) must invoke
        # init_ia_decoder_base_copy() AFTER load_state_dict so the IA decoder
        # inherits the loaded checkpoint weights.
        
        # Storage for KV cache (used in generation)
        self._last_kv_cache = None
        
        # Timing attributes (populated during forward/generate)
        self._query_prefill_time = 0.0
        self._thought_time = 0.0
        self._last_timing = {}
        self._last_decode_token_times = []

    def init_ia_decoder_base_copy(self):
        """Initialize the IA decoder as a deep copy of the (already-loaded) base model.

        Call this AFTER weights from ``load_model_path`` have been loaded into
        ``self.base_causallm``. The deep copy inherits the loaded weights so the
        IA decoder starts from the same initialization as the answer-generating
        model, but with independent parameters: gradients on the IA path do not
        flow into the main answer model, and the optimizer treats them as
        separate trainable tensors.

        Idempotent only when ``self.ia_decoder`` is None or already a
        ``BaseCopyIADecoder``; raises if a ``LightweightDecoder`` is already
        installed (mutually exclusive modes).
        """
        if not self.ia_decoder_use_base_copy:
            raise RuntimeError(
                "init_ia_decoder_base_copy() called but ia_decoder_use_base_copy=False; "
                "set ia_decoder_use_base_copy=True in the config first."
            )
        if isinstance(self.ia_decoder, LightweightDecoder):
            raise RuntimeError(
                "Cannot install BaseCopyIADecoder: a LightweightDecoder is already installed. "
                "ia_decoder_layers > 0 and ia_decoder_use_base_copy=True are mutually exclusive."
            )
        # Deep-copy the base CausalLM. This snapshots its current parameters
        # (which should already include load_model_path weights) and creates a
        # fully independent module.
        copied_base = copy.deepcopy(self.base_causallm)
        # Make sure the copy is in train mode by default; the caller can flip
        # it later if needed.
        copied_base.train()
        self.ia_decoder = BaseCopyIADecoder(copied_base)
        return self.ia_decoder

    def init_ia_decoder_from_path(self, init_path):
        """Initialize the IA decoder from a separate pretrained checkpoint.

        Loads ``init_path`` via ``AutoModelForCausalLM.from_pretrained`` and
        wraps it as the auxiliary decoder. Vocab must match the main model
        (same tokenizer family). If the aux model's hidden size differs from
        the main model's, a learnable input projection is inserted so the
        main model's post-loop latents can be fed to the aux model.

        Independent parameters from the main model: gradients on the IA path
        do not flow back, and the optimizer treats them as separate tensors.
        """
        if isinstance(self.ia_decoder, (LightweightDecoder, BaseCopyIADecoder)):
            raise RuntimeError(
                "Cannot install ExternalIADecoder: another IA decoder is already installed. "
                "ia_decoder_layers, ia_decoder_use_base_copy, and ia_decoder_init_path "
                "are mutually exclusive."
            )
        from transformers import AutoModelForCausalLM
        aux_model = AutoModelForCausalLM.from_pretrained(
            init_path,
            torch_dtype=self.base_causallm.dtype,
        )
        main_vocab = self.base_causallm.config.vocab_size
        aux_vocab = aux_model.config.vocab_size
        if main_vocab != aux_vocab:
            # Common case: main was resized to add latent tokens
            # (<|start-latent|>, <|end-latent|>, <|latent|>) which the aux
            # model doesn't have. Resize the aux model to match. Mirror the
            # init pattern used in run.py for the main model: copy the
            # embedding row of "<<" into each newly-added latent token row.
            assert main_vocab > aux_vocab, (
                f"ia_decoder_init_path vocab smaller than main: main={main_vocab}, aux={aux_vocab}. "
                f"Cannot shrink aux vocab; choose an aux model with same or smaller tokenizer."
            )
            aux_model.resize_token_embeddings(main_vocab)
            try:
                aux_emb = aux_model.get_input_embeddings()
                aux_lm_head = getattr(aux_model, "lm_head", None)
                # Source row to copy from: prefer "<<" if available; fall
                # back to silently leaving the multivariate-normal init
                # (transformers' default) if not.
                from transformers import AutoTokenizer
                aux_tok = AutoTokenizer.from_pretrained(init_path)
                src_id = aux_tok.convert_tokens_to_ids("<<")
                if src_id is not None and src_id != aux_tok.unk_token_id and src_id < aux_vocab:
                    src_emb = aux_emb.weight.data[src_id].clone()
                    for new_id in (self.latent_token_id, self.start_latent_id, self.end_latent_id):
                        if new_id is not None and new_id < main_vocab:
                            aux_emb.weight.data[new_id] = src_emb
                            if aux_lm_head is not None:
                                aux_lm_head.weight.data[new_id] = src_emb
            except Exception as e:
                # Init heuristic is best-effort; if anything goes wrong just
                # keep the default init from resize_token_embeddings.
                print(f"[init_ia_decoder_from_path] latent-token init heuristic skipped: {e}")
        aux_model.train()
        main_hidden = self.base_causallm.config.hidden_size
        self.ia_decoder = ExternalIADecoder(aux_model, main_hidden)
        return self.ia_decoder

    def forward(
        self, input_ids, attention_mask, labels, position_ids, n_looped_iters=0, 
        replaced_cot_steps=None, sample_stages=None, **kwargs
    ):
        """
        Forward pass with full prefix looping and original input injection.

        At each loop iteration, ALL positions from start to end of latent tokens are:
        1. Updated with their refined hidden states from previous iteration
        2. Injected with their original embeddings (residual connection)
        3. Recomputed through the model

        Args:
            n_looped_iters: Number of times to loop over the entire prefix (question + latent tokens).
                     If 0, no latent token is used (falls back to standard forward).
            sample_stages: Optional tensor of per-sample stages (batch_size,). When provided,
                          each sample uses hidden states from its stage-th loop iteration.
        """

        # Get original embeddings (used for injection at each loop)
        original_embeddings = self.embedding(input_ids)
        inputs_embeds = original_embeddings.clone()

        # Get number of replaced CoT steps for intermediate supervision
        n_replaced_steps = replaced_cot_steps.shape[1] if replaced_cot_steps is not None else 0

        # Verify n_looped_iters >= number of replaced CoT steps (we can loop more than we have supervision for)
        if n_replaced_steps > 0 and n_looped_iters > 0:
            assert n_looped_iters >= n_replaced_steps, (
                f"n_looped_iters ({n_looped_iters}) must be >= number of replaced CoT steps ({n_replaced_steps})"
            )

        # Find latent token positions (can be multiple consecutive tokens per batch instance)
        latent_indices = (
            input_ids == self.latent_token_id
        ).nonzero()  # (num_latent_tokens_in_the_batch, 2)

        # Get latent token range for each batch instance (None if no latent tokens)
        latent_ranges = []  # List of (start_pos, end_pos) tuples or None
        for i in range(input_ids.shape[0]):
            batch_latent_indices = latent_indices[latent_indices[:, 0] == i]
            if len(batch_latent_indices) > 0:
                # Get range from first to last latent token
                start_pos = batch_latent_indices[0, 1].item()
                end_pos = batch_latent_indices[-1, 1].item()
                latent_ranges.append((start_pos, end_pos))
            else:
                latent_ranges.append(None)

        has_latent = any(r is not None for r in latent_ranges)

        # If no latent tokens or n_looped_iters is 0, do standard forward pass
        if not has_latent or n_looped_iters == 0:
            outputs = self.base_causallm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
            )
            # Store KV cache for generation
            self._last_kv_cache = outputs.past_key_values

            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='sum')
            loss_sum = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            n_valid = (shift_labels.view(-1) != -100).sum()
            loss = loss_sum / n_valid.clamp(min=1)  # local mean for logging

            self.gen_forward_cnt += 1
            _zero = torch.tensor(0.0, device=input_ids.device)
            _zero_int = torch.tensor(0, device=input_ids.device)
            return Outputs(
                loss=loss, inputs_embeds=inputs_embeds, logits=logits,
                intermediate_logits=[], main_loss=loss, intermediate_loss=_zero,
                main_loss_sum=loss_sum, n_main_valid=n_valid,
                inter_loss_sum=_zero, n_inter_valid=_zero_int,
                codi_loss=_zero, codi_loss_sum=_zero, n_codi_valid=_zero_int,
                inter_answer_loss=_zero, inter_answer_loss_sum=_zero, n_inter_answer_valid=_zero_int,
            )

        # Find the earliest and latest latent token positions across the batch
        earliest_latent_start = min(r[0] for r in latent_ranges if r is not None)
        latest_latent_end = max(r[1] for r in latent_ranges if r is not None)

        # Define regions:
        # - [0:loop_start]: prefix region (never changes, can be cached)
        # - [loop_start:loop_end]: loop region (changes during loops)
        # Include one extra token before latent tokens so logits[0] predicts first latent token
        # This makes intermediate supervision alignment clean: logits[i] predicts target[i]
        loop_start = max(0, earliest_latent_start - 1)  # Include one token before latents
        loop_end = latest_latent_end + 1
        
        # Offset for block indexing: accounts for the extra token at the start
        # block_start = loop_idx * c_thought uses logits relative to loop region
        # Since we included one extra token, block 0 starts at position 0 in logits

        # Step 1: Process prefix region (if exists) and cache it
        import time as _time_fwd
        _do_timing = getattr(self, '_in_generate', False)
        if _do_timing:
            torch.cuda.synchronize()
            _query_start = _time_fwd.time()
        prefix_kv_cache = None
        if self.use_kv_cache and loop_start > 0:
            # Process prefix once and cache it (embeddings here never change)
            prefix_outputs = self.base_causallm(
                inputs_embeds=inputs_embeds[:, :loop_start, :],
                attention_mask=attention_mask[:, :loop_start],
                position_ids=position_ids[:, :loop_start],
                use_cache=True,
            )
            prefix_kv_cache = prefix_outputs.past_key_values
            # Gradient checkpointing makes HF silently set use_cache=False,
            # producing an empty DynamicCache. Detect and fall back to None.
            if prefix_kv_cache is not None:
                _seq = getattr(prefix_kv_cache, 'get_seq_length', None)
                if _seq is not None and _seq() == 0:
                    prefix_kv_cache = None

        # Step 2: Initial forward pass through loop region (with prefix cache if available)
        if _do_timing:
            torch.cuda.synchronize()
            self._query_prefill_time = _time_fwd.time() - _query_start
            _thought_start = _time_fwd.time()
        # Clone prefix cache — Llama's DynamicCache is mutable; without cloning,
        # Step 2 would grow it from loop_start → loop_end entries, corrupting it
        # for subsequent loop iterations that also need the original prefix cache.
        step2_kv = _clone_kv_cache(prefix_kv_cache)
        # Forward through loop region
        # When prefix_kv_cache is None (e.g. gradient checkpointing), include
        # prefix tokens in the input so the loop region can still attend to them.
        if prefix_kv_cache is not None:
            _loop_embeds = inputs_embeds[:, loop_start:loop_end, :]
            _loop_pos = position_ids[:, loop_start:loop_end]
            _loop_mask = attention_mask[:, :loop_end]
        elif loop_start > 0:
            # No KV cache but prefix exists — include prefix in forward
            _loop_embeds = inputs_embeds[:, :loop_end, :]
            _loop_pos = position_ids[:, :loop_end]
            _loop_mask = attention_mask[:, :loop_end]
        else:
            _loop_embeds = inputs_embeds[:, loop_start:loop_end, :]
            _loop_pos = position_ids[:, loop_start:loop_end]
            _loop_mask = attention_mask[:, loop_start:loop_end]

        _fwd_embeds = _loop_embeds
        outputs = efficient_forward(
            self.base_causallm, _fwd_embeds, _loop_mask, _loop_pos,
            past_key_values=step2_kv, is_gpt2=self._is_gpt2,
        )

        # Extract loop-region hidden states (if prefix was included, slice it off)
        if prefix_kv_cache is None and loop_start > 0:
            hidden_states = outputs.last_hidden_state[:, loop_start:, :]
        else:
            hidden_states = outputs.last_hidden_state

        # Helper function to vectorize embedding injection
        def inject_latent_embeddings(inputs_embeds, original_embeddings, hidden_states, latent_ranges, offset=0):
            """
            Vectorized injection of hidden states into latent token positions.

            For each batch element with latent tokens, updates embeddings at latent positions:
                inputs_embeds[start:end+1] = (original_embeddings[start:end+1] + hidden_states[start:end+1]) / 2

            Args:
                offset: Position offset for converting absolute positions in latent_ranges to relative positions
                        in the sliced tensors. For example, if inputs_embeds starts at position 10 in the full
                        sequence, offset should be 10.
            """
            # Create a mask for latent token positions: shape (batch_size, seq_len)
            batch_size, seq_len, _ = inputs_embeds.shape
            latent_mask = torch.zeros(batch_size, seq_len, device=inputs_embeds.device, dtype=torch.bool)

            for i, latent_range in enumerate(latent_ranges):
                if latent_range is not None:
                    start_pos, end_pos = latent_range
                    # Convert absolute positions to relative positions in the slice
                    rel_start = start_pos - offset
                    rel_end = end_pos - offset
                    # Only set mask if the range is within the current slice
                    if rel_start < seq_len and rel_end >= 0:
                        rel_start = max(0, rel_start)
                        rel_end = min(seq_len - 1, rel_end)
                        latent_mask[i, rel_start:rel_end + 1] = True

            # Expand mask for broadcasting: (batch_size, seq_len, 1)
            latent_mask_expanded = latent_mask.unsqueeze(-1)

            # Vectorized update: only modify latent positions
            # inputs_embeds = where(is_latent, <injected>, inputs_embeds)
            # latent_injection_mode:
            #   "add"     (default): original + hidden  (residual; original embedding added every iter)
            #   "replace":            hidden          (no residual; tests recurrent variance amplifier)
            if self.latent_injection_mode == "replace":
                updated_embeds = hidden_states
            else:  # "add"
                updated_embeds = original_embeddings + hidden_states
            inputs_embeds = torch.where(latent_mask_expanded, updated_embeds, inputs_embeds)

            return inputs_embeds

        # Use c_thought from model config
        c_thought = self.c_thought
        
        # Check if we should freeze supervised blocks
        freeze_supervised_blocks = self.freeze_supervised_blocks

        # Initialize intermediate supervision loss and token count for normalization
        intermediate_loss = 0.0
        total_valid_cot_tokens = 0
        intermediate_logits = []
        
        # Skip intermediate loss computation if weight is 0
        compute_intermediate_loss = self.intermediate_loss_weight > 0 and replaced_cot_steps is not None
        # When intermediate_loss_after_loop is True, defer supervision to after all iterations
        compute_intermediate_loss_in_loop = compute_intermediate_loss and not self.intermediate_loss_after_loop

        # Build flat supervision targets: pack all CoT tokens contiguously across latent slots
        # Shape: (batch, total_slots) where total_slots = n_looped_iters * c_thought
        # Unused slots are filled with pad_token_id (ignored by CrossEntropyLoss)
        flat_cot_targets = None
        if compute_intermediate_loss and self.flat_intermediate_supervision and replaced_cot_steps is not None:
            batch_size_rc = replaced_cot_steps.shape[0]
            total_slots = n_looped_iters * c_thought
            flat_cot_targets = torch.full(
                (batch_size_rc, total_slots), self.pad_token_id,
                device=input_ids.device, dtype=input_ids.dtype,
            )
            for b in range(batch_size_rc):
                pos = 0
                for step_idx in range(replaced_cot_steps.shape[1]):
                    step_tokens = replaced_cot_steps[b, step_idx, :]
                    # Get non-padding tokens
                    valid = step_tokens[step_tokens != self.pad_token_id]
                    n_tok = valid.shape[0]
                    if n_tok == 0:
                        continue
                    end_pos = min(pos + n_tok, total_slots)
                    flat_cot_targets[b, pos:end_pos] = valid[:end_pos - pos]
                    pos = end_pos
                    if pos >= total_slots:
                        break

        # Initialize intermediate answer supervision (suffix forward at each loop step)
        inter_answer_loss_sum = torch.tensor(0.0, device=input_ids.device)
        inter_answer_n_valid = torch.tensor(0, device=input_ids.device)
        compute_inter_answer_loss = (
            self.inter_answer_loss_weight > 0
            and replaced_cot_steps is not None
            and self.training
            and loop_end < input_ids.shape[1]
            and not self.ia_loss_after_loop  # per-step mode only; after-loop handled separately
        )
        compute_ia_after_loop = (
            self.inter_answer_loss_weight > 0
            and self.ia_loss_after_loop
            and replaced_cot_steps is not None
            and self.training
            and loop_end < input_ids.shape[1]
        )
        
        # Storage for frozen hidden states (block_idx -> hidden_states for that block's positions)
        # When freeze_supervised_blocks is True, we save hidden states after supervising each block
        # and restore them after subsequent forward passes
        frozen_hidden_states = {} if freeze_supervised_blocks else None
        
        # Storage for per-loop hidden states when using per-sample stages
        # Maps loop_idx -> hidden_states for that iteration
        per_loop_hidden_states = {} if sample_stages is not None else None

        # Step 3: Loop iterations - refine loop region with prefix cache reuse
        for loop_idx in range(n_looped_iters):
            # Clone prefix KV cache each iteration — DynamicCache is mutable,
            # so forward calls append loop-region entries in-place.
            loop_kv = _clone_kv_cache(prefix_kv_cache)

            # Vectorized embedding injection for latent positions only
            inputs_embeds[:, loop_start:loop_end, :] = inject_latent_embeddings(
                inputs_embeds[:, loop_start:loop_end, :],
                original_embeddings[:, loop_start:loop_end, :],
                hidden_states,
                latent_ranges,
                offset=loop_start,  # Convert absolute to relative positions
            )

            # Forward through loop region with updated embeddings
            # Reuse prefix cache (positions before loop_start never change)
            # Same prefix-inclusion logic as Step 2:
            if prefix_kv_cache is not None:
                _loop_embeds = inputs_embeds[:, loop_start:loop_end, :]
                _loop_pos = position_ids[:, loop_start:loop_end]
                _loop_mask = attention_mask[:, :loop_end]
            elif loop_start > 0:
                _loop_embeds = inputs_embeds[:, :loop_end, :]
                _loop_pos = position_ids[:, :loop_end]
                _loop_mask = attention_mask[:, :loop_end]
            else:
                _loop_embeds = inputs_embeds[:, loop_start:loop_end, :]
                _loop_pos = position_ids[:, loop_start:loop_end]
                _loop_mask = attention_mask[:, loop_start:loop_end]

            _fwd_embeds = _loop_embeds
            outputs = efficient_forward(
                self.base_causallm, _fwd_embeds, _loop_mask, _loop_pos,
                past_key_values=loop_kv, is_gpt2=self._is_gpt2,
            )

            # Extract loop-region hidden states (if prefix was included, slice it off)
            if prefix_kv_cache is None and loop_start > 0:
                hidden_states = outputs.last_hidden_state[:, loop_start:, :]
            else:
                hidden_states = outputs.last_hidden_state
            
            # Save hidden states for per-sample stage handling
            if per_loop_hidden_states is not None:
                per_loop_hidden_states[loop_idx] = hidden_states.clone()
            
            # Restore frozen hidden states for blocks that were already supervised
            # Use non-inplace operation to preserve gradient computation
            if freeze_supervised_blocks and frozen_hidden_states:
                # Build a mask for all frozen positions
                batch_size, seq_len, hidden_dim = hidden_states.shape
                frozen_mask = torch.zeros(batch_size, seq_len, device=hidden_states.device, dtype=torch.bool)
                
                # Accumulate frozen hidden states into a single tensor
                frozen_values = hidden_states.clone()
                
                for block_idx, saved_hs in frozen_hidden_states.items():
                    for batch_idx, latent_range in enumerate(latent_ranges):
                        if latent_range is not None:
                            start_pos, end_pos = latent_range
                            # Block i covers positions [start_pos + i*c_thought, start_pos + (i+1)*c_thought)
                            block_start = start_pos + block_idx * c_thought - loop_start
                            block_end = min(start_pos + (block_idx + 1) * c_thought, end_pos + 1) - loop_start
                            if block_start >= 0 and block_end <= seq_len:
                                frozen_mask[batch_idx, block_start:block_end] = True
                                frozen_values[batch_idx, block_start:block_end, :] = saved_hs[batch_idx, block_start:block_end, :]
                
                # Non-inplace update: where mask is True, use frozen values
                hidden_states = torch.where(frozen_mask.unsqueeze(-1), frozen_values, hidden_states)
            
            # Only store intermediate logits for generation (not training) to save memory
            if not self.training:
                intermediate_logits.append(outputs.logits)

            # Compute intermediate supervision loss for this iteration
            # Only supervise if we have a corresponding replaced CoT step and weight > 0
            if compute_intermediate_loss_in_loop and loop_idx < n_replaced_steps:
                block_start = loop_idx * c_thought

                if flat_cot_targets is not None:
                    # Flat packing: target for this block is the c_thought-sized slice of the flat sequence
                    target_tokens_block = flat_cot_targets[:, block_start:block_start + c_thought]
                    step_len = c_thought
                    if self.pad_token_id is not None:
                        valid_mask = (target_tokens_block != self.pad_token_id)
                        n_valid_tokens = valid_mask.sum().item()
                    else:
                        n_valid_tokens = target_tokens_block.numel()
                else:
                    # Per-block: target is the CoT step for this iteration
                    target_tokens_block_raw = replaced_cot_steps[:, loop_idx, :]
                    if self.pad_token_id is not None:
                        valid_mask = (target_tokens_block_raw != self.pad_token_id)
                        step_len = valid_mask.sum(dim=1).max().item()
                        n_valid_tokens = valid_mask[:, :step_len].sum().item()
                    else:
                        step_len = target_tokens_block_raw.shape[1]
                        n_valid_tokens = step_len * target_tokens_block_raw.shape[0]
                    target_tokens_block = target_tokens_block_raw[:, :step_len]
                
                # Skip loss computation if no valid tokens (all padding),
                # but do NOT use 'continue' — that would skip the IA forward
                # below, and under FSDP all ranks must call self.base_causallm
                # the same number of times (allgather deadlock).
                if step_len > 0 and n_valid_tokens > 0:
                    total_valid_cot_tokens += n_valid_tokens
                    
                    if self.intermediate_loss_type == "cosine":
                        hidden_block_start = 1 + loop_idx * c_thought
                        block_hidden = hidden_states[:, hidden_block_start:hidden_block_start + step_len, :]
                        target_sliced = target_tokens_block[:, :step_len]
                        target_embeds = self.embedding(target_sliced)
                        cos_sim = F.cosine_similarity(block_hidden, target_embeds, dim=-1)
                        valid_mask_trunc = (target_sliced != self.pad_token_id) if self.pad_token_id is not None else torch.ones_like(cos_sim, dtype=torch.bool)
                        cos_loss = (1 - cos_sim) * valid_mask_trunc.float()
                        step_loss = cos_loss.sum()
                    else:
                        block_logits = outputs.logits[:, block_start:block_start + step_len, :]
                        actual_len = block_logits.shape[1]
                        target_sliced = target_tokens_block[:, :actual_len]
                        
                        logits_flat = block_logits.contiguous().view(-1, block_logits.size(-1))
                        target_flat = target_sliced.contiguous().view(-1)
                        
                        assert self.pad_token_id is not None, "pad_token_id must be set for intermediate supervision loss"
                        step_loss_fct = CrossEntropyLoss(
                            ignore_index=self.pad_token_id, 
                            reduction='sum',
                            label_smoothing=self.intermediate_label_smoothing
                        )
                        step_loss = step_loss_fct(logits_flat, target_flat)
                    
                    intermediate_loss = intermediate_loss + step_loss
                    
                    # Save hidden states for this block so they won't be overwritten in future iterations
                    if freeze_supervised_blocks:
                        frozen_hidden_states[loop_idx] = hidden_states.clone()

            # Intermediate answer supervision via next-token prediction:
            # Use the loop region's KV cache and feed step k's CoT tokens as
            # a continuation, computing NTP loss on those tokens only.
            #
            # FSDP constraint: run IA forward on EVERY loop iteration, not just
            # loop_idx < n_replaced_steps.  n_replaced_steps comes from
            # replaced_cot_steps.shape[1] which is dynamically padded per
            # mini-batch in the collator; different ranks can have different
            # values, causing mismatched self.base_causallm() call counts and
            # an NCCL allgather deadlock.  When loop_idx >= n_replaced_steps we
            # feed dummy all-padding tokens; loss is 0 via ignore_index.
            if compute_inter_answer_loss:
                if loop_idx < n_replaced_steps:
                    target_tokens = replaced_cot_steps[:, loop_idx, :]  # (batch, max_step_len)
                else:
                    # Dummy all-padding target — keeps forward count consistent
                    target_tokens = torch.full(
                        (input_ids.shape[0], 2), self.pad_token_id,
                        device=input_ids.device, dtype=input_ids.dtype,
                    )

                # Compute valid length (exclude padding).
                # Clamp to at least 2: under FSDP all ranks must participate in
                # the forward (weight allgather). If one rank's batch is all-
                # padding (ia_step_len==0) and it skips the forward while others
                # enter it, NCCL deadlocks. Need >=2 for the NTP shift to
                # produce at least 1 element. Loss is 0 on padding via ignore_index.
                if self.pad_token_id is not None:
                    ia_valid_mask = (target_tokens != self.pad_token_id)
                    ia_step_len = max(ia_valid_mask.sum(dim=1).max().item(), 2)
                else:
                    ia_step_len = target_tokens.shape[1]

                target_sliced = target_tokens[:, :ia_step_len]  # (batch, step_len)
                target_embeds = self.embedding(target_sliced)  # (batch, step_len, hidden)

                # Prepend ALL latent hidden states from the current block so
                # the IA decoder can attend to every latent position (not just
                # the last one).  This gives the joint-distribution loss direct
                # gradient to all c_thought latent hidden states.
                #
                # hidden_states layout (loop region, prefix already sliced):
                #   position 0:                      token before latents
                #   positions 1..c_thought:          block 0 latents
                #   positions 1+k*ct .. 1+(k+1)*ct:  block k latents
                #
                # Input to IA decoder:
                #   [h_lat0, h_lat1, ..., h_lat_{ct-1}, CoT0, CoT1, ...]
                #
                # NTP alignment (causal attention):
                #   logits[ct-1]  (last latent)  predicts CoT[0]
                #   logits[ct]    (from CoT[0])  predicts CoT[1]
                #   ...
                #   logits[ct-1+step_len-1] predicts CoT[step_len-1]
                # So: logits[ct-1 : ct-1+step_len] predicts target_sliced[:]
                block_latent_start = 1 + loop_idx * c_thought
                block_latent_end = 1 + (loop_idx + 1) * c_thought
                latent_embs = hidden_states[:, block_latent_start:block_latent_end, :]  # (batch, c_thought, hidden)
                ia_input_embeds = torch.cat([latent_embs, target_embeds], dim=1)  # (batch, c_thought+step_len, hidden)
                ia_total_len = c_thought + ia_step_len

                if self.ia_decoder is not None:
                    # Lightweight auxiliary decoder: no KV cache, no position_ids needed.
                    # Just feed [latent_emb, CoT_embeds] and get logits.
                    ia_logits = self.ia_decoder(ia_input_embeds)
                elif self.ia_no_prefix:
                    # SIM-CoT style: no prefix, no KV cache.
                    # Input is just [latent_emb, CoT0, CoT1, ...] from scratch.
                    ia_position_ids = torch.arange(
                        0, ia_total_len,
                        device=input_ids.device, dtype=torch.long,
                    ).unsqueeze(0).expand(input_ids.shape[0], -1)
                    ia_attention_mask = torch.ones(
                        input_ids.shape[0], ia_total_len,
                        device=input_ids.device, dtype=attention_mask.dtype,
                    )
                    ia_outputs = self.base_causallm(
                        inputs_embeds=ia_input_embeds,
                        attention_mask=ia_attention_mask,
                        position_ids=ia_position_ids,
                        past_key_values=None,
                        use_cache=False,
                    )
                    ia_logits = ia_outputs.logits
                else:
                    # With prefix: use loop's KV cache so IA can attend to question + latents
                    # KV cache covers [0:loop_end), so position_ids start at loop_end
                    ia_position_ids = torch.arange(
                        loop_end, loop_end + ia_total_len,
                        device=input_ids.device, dtype=torch.long,
                    ).unsqueeze(0).expand(input_ids.shape[0], -1)
                    ia_attention_mask = torch.cat([
                        attention_mask[:, :loop_end],
                        torch.ones(input_ids.shape[0], ia_total_len, device=input_ids.device, dtype=attention_mask.dtype),
                    ], dim=1)
                    # Clone the KV cache before IA forward because LlamaAttention.forward()
                    # unconditionally calls DynamicCache.update(), mutating it in-place
                    # even when use_cache=False.
                    ia_kv_cache = _clone_kv_cache(outputs.past_key_values)
                    ia_outputs = self.base_causallm(
                        inputs_embeds=ia_input_embeds,
                        attention_mask=ia_attention_mask,
                        position_ids=ia_position_ids,
                        past_key_values=ia_kv_cache,
                        use_cache=False,
                    )
                    ia_logits = ia_outputs.logits

                # NTP with prepended latent hidden states:
                # ia_logits shape: (batch, c_thought + step_len, vocab)
                # logits[ct-1] (last latent) predicts CoT[0] = target_sliced[0]
                # logits[ct]   (from CoT[0]) predicts CoT[1] = target_sliced[1]
                # ...
                # logits[ct-1+step_len-1] predicts target_sliced[step_len-1]
                # So: logits[ct-1 : ct-1+step_len] predicts target_sliced[:]
                ia_shift_logits = ia_logits[:, c_thought - 1 : c_thought - 1 + ia_step_len, :].contiguous()  # (batch, step_len, vocab)
                ia_shift_labels = target_sliced.contiguous()  # (batch, step_len)

                ia_loss_fct = CrossEntropyLoss(
                    ignore_index=self.pad_token_id if self.pad_token_id is not None else -100,
                    reduction='sum',
                )
                ia_step_loss = ia_loss_fct(
                    ia_shift_logits.view(-1, ia_shift_logits.size(-1)),
                    ia_shift_labels.view(-1),
                )
                if self.pad_token_id is not None:
                    ia_step_n = (ia_shift_labels.view(-1) != self.pad_token_id).sum()
                else:
                    ia_step_n = torch.tensor(ia_shift_labels.numel(), device=input_ids.device)
                inter_answer_loss_sum = inter_answer_loss_sum + ia_step_loss
                inter_answer_n_valid = inter_answer_n_valid + ia_step_n

                # Save hidden states for this block after IA supervision
                # so they won't be overwritten in future iterations
                if freeze_supervised_blocks and loop_idx not in frozen_hidden_states:
                    frozen_hidden_states[loop_idx] = hidden_states.clone()

        # After-loop intermediate supervision: supervise ALL blocks using the
        # final iteration's logits (fully refined representations).
        if compute_intermediate_loss and self.intermediate_loss_after_loop:
            if flat_cot_targets is not None:
                # Flat packing: single CE over all slots at once
                total_slots = flat_cot_targets.shape[1]
                all_logits = outputs.logits[:, :total_slots, :]
                actual_len = all_logits.shape[1]
                target_sliced = flat_cot_targets[:, :actual_len]

                n_valid = (target_sliced != self.pad_token_id).sum().item() if self.pad_token_id is not None else target_sliced.numel()
                if n_valid > 0:
                    total_valid_cot_tokens += n_valid
                    logits_flat = all_logits.contiguous().view(-1, all_logits.size(-1))
                    target_flat = target_sliced.contiguous().view(-1)
                    step_loss_fct = CrossEntropyLoss(
                        ignore_index=self.pad_token_id,
                        reduction='sum',
                        label_smoothing=self.intermediate_label_smoothing
                    )
                    intermediate_loss = intermediate_loss + step_loss_fct(logits_flat, target_flat)
            else:
                # Per-block: iterate over each block separately
                for block_idx in range(min(n_replaced_steps, n_looped_iters)):
                    target_tokens = replaced_cot_steps[:, block_idx, :]

                    if self.pad_token_id is not None:
                        valid_mask = (target_tokens != self.pad_token_id)
                        step_len = valid_mask.sum(dim=1).max().item()
                        n_valid_tokens = valid_mask[:, :step_len].sum().item()
                    else:
                        step_len = target_tokens.shape[1]
                        n_valid_tokens = step_len * target_tokens.shape[0]

                    if step_len > 0 and n_valid_tokens > 0:
                        block_start = block_idx * c_thought
                        block_logits = outputs.logits[:, block_start:block_start + step_len, :]
                        actual_len = block_logits.shape[1]
                        target_sliced = target_tokens[:, :actual_len]

                        logits_flat = block_logits.contiguous().view(-1, block_logits.size(-1))
                        target_flat = target_sliced.contiguous().view(-1)

                        step_loss_fct = CrossEntropyLoss(
                            ignore_index=self.pad_token_id,
                            reduction='sum',
                            label_smoothing=self.intermediate_label_smoothing
                        )
                        step_loss = step_loss_fct(logits_flat, target_flat)

                        total_valid_cot_tokens += n_valid_tokens

                        intermediate_loss = intermediate_loss + step_loss

        # After-loop IA loss: single suffix forward using ALL latent hidden states
        # from the final iteration as conditioning, predicting ALL replaced CoT steps.
        if compute_ia_after_loop:
            n_steps_to_use = min(n_replaced_steps, n_looped_iters)
            # Gather all latent hidden states (all blocks, final iteration)
            all_latent_embs = hidden_states[:, 1:1 + n_steps_to_use * c_thought, :]  # (batch, n_steps*ct, hidden)
            total_latent = all_latent_embs.shape[1]

            # Flatten all replaced CoT steps into one target sequence
            all_targets = replaced_cot_steps[:, :n_steps_to_use, :]  # (batch, n_steps, max_step_len)
            flat_targets = all_targets.reshape(all_targets.shape[0], -1)  # (batch, n_steps * max_step_len)

            # Trim to valid length (exclude trailing padding)
            if self.pad_token_id is not None:
                ia_valid_mask = (flat_targets != self.pad_token_id)
                flat_target_len = max(ia_valid_mask.sum(dim=1).max().item(), 2)
            else:
                flat_target_len = flat_targets.shape[1]

            flat_targets_sliced = flat_targets[:, :flat_target_len]
            flat_target_embeds = self.embedding(flat_targets_sliced)

            # Build input: [all_latent_hidden_states, flat_CoT_embeds]
            ia_input_embeds = torch.cat([all_latent_embs, flat_target_embeds], dim=1)
            ia_total_len = total_latent + flat_target_len

            if self.ia_decoder is not None:
                ia_logits = self.ia_decoder(ia_input_embeds)
            elif self.ia_no_prefix:
                ia_position_ids = torch.arange(
                    0, ia_total_len, device=input_ids.device, dtype=torch.long,
                ).unsqueeze(0).expand(input_ids.shape[0], -1)
                ia_attention_mask = torch.ones(
                    input_ids.shape[0], ia_total_len,
                    device=input_ids.device, dtype=attention_mask.dtype,
                )
                ia_outputs = self.base_causallm(
                    inputs_embeds=ia_input_embeds,
                    attention_mask=ia_attention_mask,
                    position_ids=ia_position_ids,
                    past_key_values=None,
                    use_cache=False,
                )
                ia_logits = ia_outputs.logits
            else:
                # With prefix: use loop's KV cache so IA can attend to question + latents
                ia_position_ids = torch.arange(
                    loop_end, loop_end + ia_total_len,
                    device=input_ids.device, dtype=torch.long,
                ).unsqueeze(0).expand(input_ids.shape[0], -1)
                ia_attention_mask = torch.cat([
                    attention_mask[:, :loop_end],
                    torch.ones(input_ids.shape[0], ia_total_len, device=input_ids.device, dtype=attention_mask.dtype),
                ], dim=1)
                ia_kv_cache = _clone_kv_cache(outputs.past_key_values)
                ia_outputs = self.base_causallm(
                    inputs_embeds=ia_input_embeds,
                    attention_mask=ia_attention_mask,
                    position_ids=ia_position_ids,
                    past_key_values=ia_kv_cache,
                    use_cache=False,
                )
                ia_logits = ia_outputs.logits

            # NTP: logits[total_latent-1 : total_latent-1+flat_target_len] predicts flat_targets_sliced
            ia_shift_logits = ia_logits[:, total_latent - 1 : total_latent - 1 + flat_target_len, :].contiguous()
            ia_shift_labels = flat_targets_sliced.contiguous()

            ia_loss_fct = CrossEntropyLoss(
                ignore_index=self.pad_token_id if self.pad_token_id is not None else -100,
                reduction='sum',
            )
            ia_step_loss = ia_loss_fct(
                ia_shift_logits.view(-1, ia_shift_logits.size(-1)),
                ia_shift_labels.view(-1),
            )
            if self.pad_token_id is not None:
                ia_step_n = (ia_shift_labels.view(-1) != self.pad_token_id).sum()
            else:
                ia_step_n = torch.tensor(ia_shift_labels.numel(), device=input_ids.device)
            inter_answer_loss_sum = inter_answer_loss_sum + ia_step_loss
            inter_answer_n_valid = inter_answer_n_valid + ia_step_n

        # Per-sample hidden state selection: each sample uses hidden states from its target stage
        # Use advanced indexing to maintain gradient flow
        if sample_stages is not None and per_loop_hidden_states:
            batch_size = hidden_states.shape[0]
            # Stack all loop hidden states: (n_looped_iters, batch, seq, hidden)
            stacked_hidden = torch.stack([per_loop_hidden_states[i] for i in range(n_looped_iters)], dim=0)
            
            # Compute target loop index for each sample: stage k uses loop k-1 (0-indexed)
            # Clamp to valid range [0, n_looped_iters-1]
            target_loops = (sample_stages - 1).clamp(min=0, max=n_looped_iters - 1)  # (batch,)
            
            # Use advanced indexing: select hidden states per sample
            # target_loops[i] gives the loop index for batch item i
            hidden_states = stacked_hidden[target_loops, torch.arange(batch_size, device=hidden_states.device)]
        
        # Restore frozen hidden states for supervised blocks before final injection.
        # Each supervised block's hidden states are replaced with the snapshot taken
        # right after its supervision step.  clone() preserved the computation graph,
        # so gradients still flow back to the loop iteration that produced them.
        if freeze_supervised_blocks and frozen_hidden_states:
            batch_size_hs, seq_len_hs, hidden_dim_hs = hidden_states.shape
            frozen_mask = torch.zeros(batch_size_hs, seq_len_hs, device=hidden_states.device, dtype=torch.bool)
            frozen_values = hidden_states.clone()

            for block_idx, saved_hs in frozen_hidden_states.items():
                for batch_idx, latent_range in enumerate(latent_ranges):
                    if latent_range is not None:
                        start_pos, end_pos = latent_range
                        block_start = start_pos + block_idx * c_thought - loop_start
                        block_end = min(start_pos + (block_idx + 1) * c_thought, end_pos + 1) - loop_start
                        if block_start >= 0 and block_end <= seq_len_hs:
                            frozen_mask[batch_idx, block_start:block_end] = True
                            frozen_values[batch_idx, block_start:block_end, :] = saved_hs[batch_idx, block_start:block_end, :]

            hidden_states = torch.where(frozen_mask.unsqueeze(-1), frozen_values, hidden_states)

        # Inject final hidden states into embeddings for suffix processing
        inputs_embeds[:, loop_start:loop_end, :] = inject_latent_embeddings(
            inputs_embeds[:, loop_start:loop_end, :],
            original_embeddings[:, loop_start:loop_end, :],
            hidden_states,
            latent_ranges,
            offset=loop_start,
        )

        # After final loop iteration, save combined KV cache for processing remaining tokens.
        # With gradient checkpointing, HF silently sets use_cache=False, so
        # past_key_values may be None or an empty DynamicCache. Detect that.
        final_kv_cache = outputs.past_key_values if self.use_kv_cache else None
        if final_kv_cache is not None:
            _seq = getattr(final_kv_cache, 'get_seq_length', None)
            if _seq is not None and _seq() == 0:
                final_kv_cache = None  # empty cache from gradient checkpointing
        
        # Store KV cache for potential use in generation
        self._last_kv_cache = final_kv_cache
        if _do_timing:
            torch.cuda.synchronize()
            self._thought_time = _time_fwd.time() - _thought_start

        # Step 4: Process remaining tokens (CoT steps + answer) if they exist
        # CODI loss needs all-layer hidden states from the suffix forward
        need_suffix_hidden = (self.codi_loss_weight > 0 and self.training
                              and self.teacher_model is not None
                              and n_replaced_steps > 0 and replaced_cot_steps is not None)
        suffix_all_hidden_states = None

        if loop_end < input_ids.shape[1]:

            # When using per-sample stages, do full forward (no KV cache) for simplicity
            if sample_stages is not None:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_hidden_states=need_suffix_hidden,
                )
                if need_suffix_hidden:
                    suffix_all_hidden_states = tuple(h[:, loop_end:, :] for h in outputs.hidden_states)
            elif self.use_kv_cache and final_kv_cache is not None:
                # Use combined KV cache: only process new tokens [loop_end:]
                # Note: attention_mask must cover full sequence (cached + new tokens)
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, loop_end:, :],
                    attention_mask=attention_mask,  # Full mask including cached positions
                    position_ids=position_ids[:, loop_end:],
                    past_key_values=final_kv_cache,
                    output_hidden_states=need_suffix_hidden,
                )
                # Update stored KV cache to include suffix
                self._last_kv_cache = outputs.past_key_values
                if need_suffix_hidden:
                    suffix_all_hidden_states = outputs.hidden_states
            else:
                # No KV cache: process entire sequence
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_hidden_states=need_suffix_hidden,
                )
                if need_suffix_hidden:
                    suffix_all_hidden_states = tuple(h[:, loop_end:, :] for h in outputs.hidden_states)

        logits = outputs.logits

        # Track forward passes:
        # - 1 for prefix (if exists and use_kv_cache)
        # - 1 for initial loop region pass
        # - n_looped_iters for loop iterations
        # - 1 for remaining tokens (if they exist)
        prefix_pass = 1 if (self.use_kv_cache and loop_start > 0) else 0
        self.gen_forward_cnt += (
            prefix_pass + 1 + n_looped_iters + (1 if loop_end < input_ids.shape[1] else 0)
        )

        # Compute loss
        # When using KV cache, logits only covers answer portion [loop_end:]
        # So shift_labels must match: predict tokens at [loop_end+1:]
        if self.use_kv_cache and final_kv_cache is not None and loop_end < input_ids.shape[1] and sample_stages is None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., loop_end+1:].contiguous()
            if shift_logits.shape[0] * shift_logits.shape[1] != shift_labels.numel():
                raise RuntimeError(
                    f"[SHAPE BUG] shift_logits flat={shift_logits.shape[0]*shift_logits.shape[1]} "
                    f"shift_labels flat={shift_labels.numel()} | "
                    f"logits={logits.shape} labels={labels.shape} loop_end={loop_end} "
                    f"kv_len={final_kv_cache.get_seq_length() if final_kv_cache is not None else -1} input_ids={input_ids.shape}"
                )
            loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='sum')
            loss_sum = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            n_valid = (shift_labels.view(-1) != -100).sum()
            lm_loss = loss_sum / n_valid.clamp(min=1)
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='sum')
            loss_sum = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            n_valid = (shift_labels.view(-1) != -100).sum()
            lm_loss = loss_sum / n_valid.clamp(min=1)

        # Combine LM loss with intermediate supervision loss
        # Return raw sums for FSDP normalization in run.py
        inter_loss_sum_raw = intermediate_loss  # raw CE sum (accumulated in loop)
        if total_valid_cot_tokens > 0 and intermediate_loss > 0:
            n_inter_valid = torch.tensor(total_valid_cot_tokens, dtype=torch.float, device=input_ids.device)
            intermediate_loss_mean = intermediate_loss / n_inter_valid.clamp(min=1)
            loss = lm_loss + self.intermediate_loss_weight * intermediate_loss_mean
        else:
            intermediate_loss_mean = torch.tensor(0.0, device=input_ids.device)
            inter_loss_sum_raw = torch.tensor(0.0, device=input_ids.device)
            n_inter_valid = torch.tensor(0, device=input_ids.device)

        # Compute CODI-style answer-boundary distillation loss
        # Raw sums returned for FSDP normalization in run.py; NOT added to loss here.
        codi_loss = torch.tensor(0.0, device=input_ids.device)
        codi_loss_sum = torch.tensor(0.0, device=input_ids.device)
        n_codi_valid = torch.tensor(0, device=input_ids.device)
        if (self.teacher_model is not None and self.codi_loss_weight > 0
            and n_replaced_steps > 0 and replaced_cot_steps is not None and self.training):

            codi_loss_sum, n_codi_valid = self._compute_codi_loss(
                input_ids=input_ids,
                labels=labels,
                suffix_all_hidden_states=suffix_all_hidden_states,
                replaced_cot_steps=replaced_cot_steps,
                latent_ranges=latent_ranges,
                loop_end=loop_end,
                n_replaced_steps=n_replaced_steps,
            )
            if n_codi_valid > 0:
                codi_loss = codi_loss_sum / n_codi_valid.clamp(min=1).float()

        # Intermediate answer supervision: compute local mean for logging
        if inter_answer_n_valid > 0:
            inter_answer_loss_mean = inter_answer_loss_sum / inter_answer_n_valid.clamp(min=1).float()
        else:
            inter_answer_loss_mean = torch.tensor(0.0, device=input_ids.device)

        return Outputs(
            loss=loss if total_valid_cot_tokens > 0 and inter_loss_sum_raw > 0 else lm_loss,
            inputs_embeds=inputs_embeds, logits=logits,
            intermediate_logits=intermediate_logits,
            main_loss=lm_loss, intermediate_loss=intermediate_loss_mean,
            main_loss_sum=loss_sum, n_main_valid=n_valid,
            inter_loss_sum=inter_loss_sum_raw, n_inter_valid=n_inter_valid,
            codi_loss=codi_loss, codi_loss_sum=codi_loss_sum, n_codi_valid=n_codi_valid,
            inter_answer_loss=inter_answer_loss_mean,
            inter_answer_loss_sum=inter_answer_loss_sum,
            n_inter_answer_valid=inter_answer_n_valid,
        )

    def _compute_codi_loss(
        self,
        input_ids,
        labels,
        suffix_all_hidden_states,
        replaced_cot_steps,
        latent_ranges,
        loop_end,
        n_replaced_steps,
    ):
        """
        Compute CODI-style per-layer distillation loss at the answer BOUNDARY
        position only (matching SIM-CoT original: single position × all layers).
        Returns raw loss sum and valid count for FSDP normalization in run.py.

        Returns:
            codi_loss_sum: Raw sum of per-layer losses at the boundary position
            n_codi_valid: Number of valid (layer, batch) comparisons
        """
        if suffix_all_hidden_states is None:
            return (torch.tensor(0.0, device=input_ids.device),
                    torch.tensor(0, device=input_ids.device))

        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Find the first answer position (boundary) in the suffix for each sample
        suffix_labels = labels[:, loop_end:]  # (batch, suffix_len)
        answer_mask = (suffix_labels != -100)  # (batch, suffix_len)

        if not answer_mask.any():
            return (torch.tensor(0.0, device=device),
                    torch.tensor(0, device=device))

        # Build teacher input and find corresponding boundary positions
        teacher_input_ids_list = []
        teacher_boundary_list = []   # single position per sample
        student_boundary_list = []   # single position per sample

        for batch_idx in range(batch_size):
            latent_range = latent_ranges[batch_idx]
            if latent_range is None:
                teacher_input_ids_list.append(input_ids[batch_idx].tolist())
                teacher_boundary_list.append(None)
                student_boundary_list.append(None)
                continue

            start_pos, end_pos = latent_range

            # Question tokens (before <bot>)
            question_tokens = input_ids[batch_idx, :start_pos - 1].tolist()

            # Suffix tokens (after <eot>)
            suffix_tokens = input_ids[batch_idx, end_pos + 2:].tolist()

            # Build CoT sequence
            cot_tokens = []
            for step_idx in range(min(n_replaced_steps, replaced_cot_steps.shape[1])):
                step_tokens = replaced_cot_steps[batch_idx, step_idx].tolist()
                if self.pad_token_id is not None:
                    step_tokens = [t for t in step_tokens if t != self.pad_token_id]
                if len(step_tokens) > 0:
                    cot_tokens.extend(step_tokens)

            # Teacher sequence: question + cot + suffix
            teacher_tokens = question_tokens + cot_tokens + suffix_tokens
            teacher_input_ids_list.append(teacher_tokens)

            # Teacher boundary: position right where answer starts (after question + CoT)
            teacher_boundary_list.append(len(question_tokens) + len(cot_tokens))

            # Student boundary: first answer position in suffix hidden states
            # suffix hidden states start at loop_end, answer starts at end_pos + 2
            student_suffix_offset = end_pos + 2 - loop_end
            # Find first actual answer token (where labels != -100)
            sample_answer_mask = answer_mask[batch_idx]
            answer_indices = sample_answer_mask.nonzero(as_tuple=True)[0]
            if len(answer_indices) > 0:
                student_boundary_list.append(answer_indices[0].item())
            else:
                student_boundary_list.append(None)

        # Pad teacher inputs
        max_len = max(len(t) for t in teacher_input_ids_list)
        teacher_input_ids = torch.full((batch_size, max_len), self.pad_token_id or 0, dtype=torch.long, device=device)
        teacher_attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)

        for batch_idx, tokens in enumerate(teacher_input_ids_list):
            teacher_input_ids[batch_idx, :len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
            teacher_attention_mask[batch_idx, :len(tokens)] = 1

        # Run teacher model with all hidden states
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask,
                output_hidden_states=True,
            )
            teacher_all_hidden = teacher_outputs.hidden_states

        # Compute per-layer loss at BOUNDARY position only
        n_layers = len(suffix_all_hidden_states)
        assert len(teacher_all_hidden) == n_layers, (
            f"Layer count mismatch: student has {n_layers}, teacher has {len(teacher_all_hidden)}"
        )

        use_cosine = (self.codi_loss_type == "cosine")
        if self.codi_loss_type == "mse":
            loss_fn = nn.MSELoss()  # reduction='mean' matching SIM-CoT
        elif use_cosine:
            loss_fn = None
        else:  # smooth_l1
            loss_fn = nn.SmoothL1Loss()  # reduction='mean' matching SIM-CoT

        total_loss_sum = torch.tensor(0.0, device=device)
        total_valid = 0

        for layer_idx in range(n_layers):
            student_hidden = suffix_all_hidden_states[layer_idx]
            teacher_hidden = teacher_all_hidden[layer_idx]

            for batch_idx in range(batch_size):
                t_pos = teacher_boundary_list[batch_idx]
                s_pos = student_boundary_list[batch_idx]
                if t_pos is None or s_pos is None:
                    continue

                # Single boundary vector
                teacher_h = teacher_hidden[batch_idx, t_pos, :]  # (hidden_dim,)
                student_h = student_hidden[batch_idx, s_pos, :]  # (hidden_dim,)

                if use_cosine:
                    cos_sim = F.cosine_similarity(
                        student_h.unsqueeze(0), teacher_h.detach().unsqueeze(0), dim=-1
                    )
                    layer_loss = 1.0 - cos_sim.squeeze()
                else:
                    layer_loss = loss_fn(student_h, teacher_h.detach())

                total_loss_sum = total_loss_sum + layer_loss
                total_valid += 1

        # Average over layers (matching SIM-CoT: distill_loss /= len(outputs.hidden_states))
        if n_layers > 0 and total_valid > 0:
            total_loss_sum = total_loss_sum / n_layers

        n_valid = torch.tensor(total_valid // n_layers if n_layers > 0 else 0, device=device)
        return total_loss_sum, n_valid

    def train(self, mode=True):
        # Properly propagate training mode to all submodules (including self.training).
        super().train(mode)
        self.base_causallm.train(mode)
        if self.teacher_model is not None:
            self.teacher_model.eval()
        return self

    def eval(self):
        # super().eval() sets self.training = False on this module and all
        # children (recursively), so eval-only code paths (e.g. populating
        # intermediate_logits) take effect.
        super().eval()
        self.base_causallm.eval()
        return self

    def generate(
        self,
        input_ids,
        attention_mask,  # attention_mask is not used
        max_new_tokens=16,
        output_embedding=False,
        output_intermediate=False,
        synced_gpus=False,
        n_looped_iters=0,
        **kwargs,
    ):
        """
        Generate with full prefix looping and original input injection.

        Args:
            n_looped_iters: Number of times to loop over the entire prefix (question + latent tokens)
                     during generation.
        """

        self.gen_forward_cnt = 0

        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        tokens = input_ids[0].detach().tolist()

        import time as _time

        labels = input_ids.clone()  # placeholder. not used.
        self._in_generate = True
        outputs = self.forward(
            input_ids,
            torch.ones_like(input_ids, device=input_ids.device),
            labels,
            torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).reshape(1, -1),
            n_looped_iters=n_looped_iters,
        )
        self._in_generate = False

        # query_prefill_time and thought_time are set inside forward()

        inputs_embeds = outputs.inputs_embeds
        # Expose the per-loop intermediate logits and the final logits from the
        # forward pass so callers (e.g. analysis scripts) can extract latent
        # representations without doing a second forward. Only populated when
        # the model is in eval mode (see forward()'s `if not self.training`
        # gate that fills `outputs.intermediate_logits`).
        self._last_intermediate_logits = outputs.intermediate_logits
        self._last_final_logits = outputs.logits
        intermediate_tokens = []
        if output_intermediate:
            # Decode intermediate logits (expensive bookkeeping, only for debug/visualization)
            intermediate_logits = outputs.intermediate_logits
            for logits in intermediate_logits:
                step_tokens = []
                for i in range(logits.shape[1]):
                    next_token = torch.argmax(logits[0, i]).item()
                    step_tokens.append(next_token)
                intermediate_tokens.append(step_tokens)

        # --- Begin answer decode (timing starts here, after intermediate logits bookkeeping) ---
        torch.cuda.synchronize()
        _decode_start = _time.time()

        # Collect answer logits (eval only) for entropy analysis
        _answer_logits_list = [] if not self.training else None

        # get the first token using the current hidden state (argmax on already-computed logits, ~free)
        next_token = torch.argmax(outputs.logits[0, -1]).item()  # .item() implicitly syncs
        tokens.append(next_token)
        if _answer_logits_list is not None:
            _answer_logits_list.append(outputs.logits[0, -1:, :].detach())
        # Record per-token decode times: (token_id, generation_time)
        _decode_token_times = [(next_token, _time.time() - _decode_start)]
        
        # Use the KV cache already built during forward
        # self._last_kv_cache contains KV for the full sequence after loops
        kv_cache = self._last_kv_cache
        seq_len = input_ids.shape[1]

        # Generate tokens one by one using KV cache
        for _ in range(max_new_tokens - 1):
            self.gen_forward_cnt += 1
            _iter_start = _time.time()
            
            # Forward just the new token with KV cache
            new_token_embed = self.embedding(
                torch.tensor([next_token], device=input_ids.device)
            ).view(1, 1, -1)
            
            outputs = self.base_causallm(
                inputs_embeds=new_token_embed,
                past_key_values=kv_cache,
                position_ids=torch.tensor([[seq_len]], device=input_ids.device),
                use_cache=True,
            )
            kv_cache = outputs.past_key_values
            seq_len += 1
            
            next_token = torch.argmax(outputs.logits[0, -1]).item()  # .item() implicitly syncs
            _decode_token_times.append((next_token, _time.time() - _iter_start))
            if _answer_logits_list is not None:
                _answer_logits_list.append(outputs.logits[0, -1:, :].detach())
            if next_token == self.eos_token_id:
                break
            tokens.append(next_token)

        if synced_gpus:
            # in FSDP, the number of forward pass need to be the same across devices
            # Forward count: 1 (initial) + n_looped_iters + 1 (remaining) + autoregressive tokens
            while self.gen_forward_cnt < max_new_tokens + n_looped_iters + 2:
                self.gen_forward_cnt += 1
                _ = self.base_causallm(
                    inputs_embeds=new_token_embed,
                    past_key_values=kv_cache,
                    use_cache=True,
                )

        if output_embedding:
            # Reconstruct full embeddings if needed
            all_embeds = [inputs_embeds]
            for t in tokens[input_ids.shape[1]:]:
                all_embeds.append(self.embedding(torch.tensor([t], device=input_ids.device)).view(1, 1, -1))
            full_embeds = torch.cat(all_embeds, dim=1)
            return torch.tensor(tokens).view(1, -1), full_embeds, intermediate_tokens
        else:
            torch.cuda.synchronize()
            _decode_time = _time.time() - _decode_start
            self._last_timing = {
                "query_prefill": self._query_prefill_time,
                "thought": self._thought_time,
                "decode": _decode_time,
            }
            self._last_decode_token_times = _decode_token_times
            # Expose per-token answer logits (eval only) for entropy analysis
            if _answer_logits_list is not None and _answer_logits_list:
                self._last_answer_logits = torch.cat(_answer_logits_list, dim=0)  # (n_gen, vocab)
            else:
                self._last_answer_logits = None
            if output_intermediate:
                return torch.tensor(tokens).view(1, -1), intermediate_tokens
            return torch.tensor(tokens).view(1, -1)
