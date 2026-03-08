"""Low-VRAM inference for KugelAudio using component-level CPU↔GPU offloading.

Inspired by AirLLM's approach: instead of keeping the entire ~19GB model on GPU,
we offload components to CPU and only move the active component to GPU when needed.

Architecture recap (7B model, bfloat16):
  - Qwen2 language model (28 layers): ~14GB
  - LM head:                          ~1GB
  - Diffusion head (4 layers):        ~0.3GB
  - Acoustic tokenizer decoder:       ~0.05GB
  - Acoustic connector:               ~0.01GB
  - KV cache + activations:           ~1-3GB

Strategy:
  1. Load model on CPU with bfloat16 (or 4-bit quantized for even less RAM)
  2. During generate(), move each component to GPU only when it's needed:
     - LM forward pass  → move LM to GPU, compute, move back to CPU
     - Diffusion sample → move diffusion head to GPU, compute, move back to CPU
     - Audio decode     → move acoustic decoder to GPU, compute, move back to CPU
  3. Pin CPU memory for faster host↔device transfers
  4. Only ~2-4GB of VRAM is needed at any time (one component + KV cache)

Trade-off: Inference is ~5-10x slower due to constant CPU↔GPU transfers,
but it makes the 7B model usable on a 6GB GPU.
"""

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import ModelOutput
from transformers.utils import logging

from .kugelaudio_inference import (
    KugelAudioForConditionalGenerationInference,
    KugelAudioGenerationOutput,
    KugelAudioTokenConstraintProcessor,
    _get_cache_tensors,
)
from .tokenizer import KugelAudioTokenizerStreamingCache

logger = logging.get_logger(__name__)


def _move_to(module: nn.Module, device: torch.device, non_blocking: bool = True):
    """Move a module to a device. Uses non-blocking transfers when possible."""
    module.to(device, non_blocking=non_blocking)


def _pin_module_memory(module: nn.Module):
    """Pin CPU tensors for faster CPU→GPU transfers.

    Only pins parameters and buffers that are on CPU and contiguous.
    """
    for param in module.parameters():
        if param.device.type == "cpu" and param.data.is_contiguous() and not param.data.is_pinned():
            try:
                param.data = param.data.pin_memory()
            except RuntimeError:
                pass  # Some tensors can't be pinned (e.g., sparse)
    for buf in module.buffers():
        if buf.device.type == "cpu" and buf.data.is_contiguous() and not buf.data.is_pinned():
            try:
                buf.data = buf.data.pin_memory()
            except RuntimeError:
                pass


class LowVRAMInferenceWrapper:
    """Wraps KugelAudioForConditionalGenerationInference for low-VRAM inference.

    Instead of keeping the full model on GPU, this wrapper:
    1. Keeps the model on CPU
    2. Moves individual components to GPU only during their forward pass
    3. Moves them back to CPU immediately after

    Usage:
        model = KugelAudioForConditionalGenerationInference.from_pretrained(
            "kugelaudio/kugelaudio-0-open", torch_dtype=torch.bfloat16
        )
        model.eval()
        model.model.strip_encoders()

        wrapper = LowVRAMInferenceWrapper(model, device="cuda")
        outputs = wrapper.generate(**inputs, cfg_scale=3.0)
    """

    def __init__(
        self,
        model: KugelAudioForConditionalGenerationInference,
        device: str = "cuda",
        pin_memory: bool = True,
    ):
        self.model = model
        self.device = torch.device(device)
        self.cpu_device = torch.device("cpu")
        self.config = model.config
        self.ddpm_inference_steps = model.ddpm_inference_steps

        # Ensure model is on CPU and in eval mode
        self.model.to(self.cpu_device)
        self.model.eval()

        # Pin CPU memory for faster transfers
        if pin_memory and torch.cuda.is_available():
            logger.info("Pinning model CPU memory for faster transfers...")
            _pin_module_memory(self.model)
            logger.info("Memory pinning complete.")

        # Pre-identify the main components for offloading
        self._lm = self.model.model.language_model  # Qwen2 backbone (~14GB total)
        self._lm_head = self.model.lm_head  # LM head (~1GB)
        self._embed = self.model.model.get_input_embeddings()  # Embedding layer (~0.3GB)
        self._connector = self.model.model.acoustic_connector  # Small
        self._diffusion_head = self.model.model.prediction_head  # ~0.3GB
        self._acoustic_decoder = self.model.model.acoustic_tokenizer  # ~0.05GB
        self._noise_scheduler = self.model.model.noise_scheduler  # No parameters (CPU is fine)

        # Identify individual Qwen2 transformer layers for layer-by-layer offloading
        # Each layer is ~0.5GB in bfloat16, which easily fits in 6GB VRAM
        self._lm_layers = self._lm.layers  # nn.ModuleList of Qwen2DecoderLayer
        self._lm_norm = self._lm.norm  # Final RMSNorm (tiny)
        self._lm_rotary_emb = getattr(self._lm, "rotary_emb", None)  # RoPE (tiny)
        self._num_layers = len(self._lm_layers)

        # Scaling buffers - always keep on GPU (tiny)
        self._scaling_factor = self.model.model.speech_scaling_factor.to(self.device)
        self._bias_factor = self.model.model.speech_bias_factor.to(self.device)

        logger.info(
            f"LowVRAMInferenceWrapper initialized. Model on CPU, active device: {self.device}"
        )
        logger.info(
            f"  LM has {self._num_layers} transformer layers "
            f"(~{self._num_layers * 0.5:.1f}GB total, ~0.5GB per layer)"
        )

    @torch.no_grad()
    def _embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Compute token embeddings. Moves embedding layer to GPU temporarily."""
        _move_to(self._embed, self.device)
        token_ids = token_ids.to(self.device)
        embeds = self._embed(token_ids)
        _move_to(self._embed, self.cpu_device)
        return embeds

    @torch.no_grad()
    def _lm_forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values=None,
        use_cache: bool = True,
    ):
        """Run LM forward pass with LAYER-BY-LAYER offloading.

        Instead of moving the entire ~14GB LM to GPU (which wouldn't fit in 6GB),
        we process one transformer layer at a time (~0.5GB each):

        1. Move layer N to GPU
        2. Run layer N forward (with its KV cache slice)
        3. Move layer N back to CPU
        4. Repeat for layer N+1

        This is the core AirLLM-style optimization that makes 7B models
        runnable on 6GB GPUs.
        """
        inputs_embeds = inputs_embeds.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Initialize DynamicCache if needed
        if past_key_values is None and use_cache:
            past_key_values = DynamicCache()

        # Compute position_ids and cache_position from attention_mask
        batch_size, seq_len = inputs_embeds.shape[:2]
        if past_key_values is not None and len(past_key_values.key_cache) > 0:
            # Incremental decoding: past_len = length of first cached key
            past_len = past_key_values.key_cache[0].shape[2]
        else:
            past_len = 0
        cache_position = torch.arange(past_len, past_len + seq_len, device=self.device)
        position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)

        # Compute causal attention mask
        target_length = past_len + seq_len
        causal_mask = self.model.model._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=seq_len,
            target_length=target_length,
            dtype=inputs_embeds.dtype,
            device=self.device,
            cache_position=cache_position,
            batch_size=batch_size,
        )

        # Start with hidden_states = inputs_embeds
        hidden_states = inputs_embeds

        # Move per-layer KV cache entries to GPU as needed
        # Process each transformer layer one at a time
        for layer_idx in range(self._num_layers):
            layer = self._lm_layers[layer_idx]

            # Move this layer to GPU
            _move_to(layer, self.device)

            # Move this layer's KV cache to GPU (if it exists)
            if past_key_values is not None and layer_idx < len(past_key_values.key_cache):
                past_key_values.key_cache[layer_idx] = past_key_values.key_cache[layer_idx].to(
                    self.device, non_blocking=True
                )
                past_key_values.value_cache[layer_idx] = past_key_values.value_cache[layer_idx].to(
                    self.device, non_blocking=True
                )

            torch.cuda.synchronize()

            # Run layer forward
            layer_outputs = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
            )

            hidden_states = layer_outputs[0]

            # Move this layer's KV cache back to CPU
            if past_key_values is not None and layer_idx < len(past_key_values.key_cache):
                past_key_values.key_cache[layer_idx] = past_key_values.key_cache[layer_idx].to(
                    self.cpu_device, non_blocking=True
                )
                past_key_values.value_cache[layer_idx] = past_key_values.value_cache[layer_idx].to(
                    self.cpu_device, non_blocking=True
                )

            # Move layer back to CPU
            _move_to(layer, self.cpu_device)

        # Apply final norm (tiny, fits easily)
        _move_to(self._lm_norm, self.device)
        hidden_states = self._lm_norm(hidden_states)
        _move_to(self._lm_norm, self.cpu_device)

        # Apply LM head to get logits (only for last token position)
        _move_to(self._lm_head, self.device)
        logits = self._lm_head(hidden_states[:, -1:, :])
        _move_to(self._lm_head, self.cpu_device)

        # Keep outputs on GPU (they're small)
        last_hidden = hidden_states[:, -1:, :].clone()
        logits = logits.clone()

        torch.cuda.empty_cache()

        return logits, last_hidden, past_key_values

    def _move_cache_to(self, cache, device):
        """Move a DynamicCache or tuple-based cache to a device."""
        if cache is None:
            return None
        if isinstance(cache, DynamicCache):
            for i in range(len(cache.key_cache)):
                cache.key_cache[i] = cache.key_cache[i].to(device, non_blocking=True)
                cache.value_cache[i] = cache.value_cache[i].to(device, non_blocking=True)
            return cache
        # Tuple-based cache
        if isinstance(cache, (list, tuple)):
            new_cache = []
            for layer_cache in cache:
                if isinstance(layer_cache, (list, tuple)):
                    new_cache.append(tuple(t.to(device, non_blocking=True) for t in layer_cache))
                else:
                    new_cache.append(layer_cache.to(device, non_blocking=True))
            return type(cache)(new_cache)
        return cache

    @torch.no_grad()
    def _process_speech_inputs(self, voice_cache: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process pre-encoded voice features. Moves connector to GPU temporarily."""
        _move_to(self._connector, self.device)

        acoustic_mean = voice_cache["acoustic_mean"].to(device=self.device, dtype=torch.bfloat16)
        fix_std = voice_cache.get("acoustic_std", self.model.acoustic_tokenizer.fix_std)
        acoustic_features = acoustic_mean + fix_std * torch.randn_like(acoustic_mean)

        # Apply scaling
        if not torch.isnan(self._scaling_factor):
            acoustic_features = (acoustic_features + self._bias_factor) * self._scaling_factor

        acoustic_embed = self._connector(acoustic_features)

        batch_size = acoustic_features.shape[0]
        seq_len = acoustic_features.shape[1]
        speech_masks = torch.ones(batch_size, seq_len, dtype=torch.bool, device=self.cpu_device)
        speech_embeds = acoustic_embed[speech_masks.to(self.device)]

        _move_to(self._connector, self.cpu_device)
        return acoustic_features, speech_embeds

    @torch.no_grad()
    def _sample_speech_tokens(
        self, condition: torch.Tensor, neg_condition: torch.Tensor, cfg_scale: float = 3.0
    ) -> torch.Tensor:
        """Sample speech latents using diffusion. Moves diffusion head to GPU temporarily."""
        _move_to(self._diffusion_head, self.device)
        torch.cuda.synchronize()

        self._noise_scheduler.set_timesteps(self.ddpm_inference_steps)

        condition = condition.to(self.device)
        neg_condition = neg_condition.to(self.device)

        if cfg_scale == 1.0:
            speech = torch.randn(condition.shape[0], self.config.acoustic_vae_dim).to(condition)
            for t in self._noise_scheduler.timesteps:
                eps = self._diffusion_head(
                    speech, t.repeat(speech.shape[0]).to(speech), condition=condition
                )
                speech = self._noise_scheduler.step(eps, t, speech).prev_sample
        else:
            combined_condition = torch.cat([condition, neg_condition], dim=0).to(self.device)
            speech = torch.randn(combined_condition.shape[0], self.config.acoustic_vae_dim).to(
                combined_condition
            )
            for t in self._noise_scheduler.timesteps:
                half = speech[: len(speech) // 2]
                combined = torch.cat([half, half], dim=0)
                eps = self._diffusion_head(
                    combined, t.repeat(combined.shape[0]).to(combined), condition=combined_condition
                )
                cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
                half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
                eps = torch.cat([half_eps, half_eps], dim=0)
                speech = self._noise_scheduler.step(eps, t, speech).prev_sample
            speech = speech[: len(speech) // 2]

        result = speech.clone()
        _move_to(self._diffusion_head, self.cpu_device)
        torch.cuda.empty_cache()
        return result

    @torch.no_grad()
    def _decode_audio(
        self,
        scaled_latent: torch.Tensor,
        acoustic_cache: KugelAudioTokenizerStreamingCache,
        sample_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Decode speech latents to audio. Moves acoustic decoder to GPU temporarily."""
        _move_to(self._acoustic_decoder, self.device)
        torch.cuda.synchronize()

        scaled_latent = scaled_latent.to(self.device)
        audio = self._acoustic_decoder.decode(
            scaled_latent.unsqueeze(1).permute(0, 2, 1),
            cache=acoustic_cache,
            sample_indices=sample_indices.to(self.device),
            use_cache=True,
        )

        result = audio.cpu()
        _move_to(self._acoustic_decoder, self.cpu_device)
        torch.cuda.empty_cache()
        return result

    @torch.no_grad()
    def generate(
        self,
        text_ids: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        voice_cache: Optional[dict] = None,
        speech_input_mask: Optional[torch.Tensor] = None,
        cfg_scale: float = 3.0,
        max_new_tokens: int = 2048,
        do_sample: bool = False,
        temperature: float = 1.0,
        show_progress: bool = True,
        **kwargs,
    ) -> KugelAudioGenerationOutput:
        """Generate speech with component-level CPU↔GPU offloading.

        Same API as KugelAudioForConditionalGenerationInference.generate(),
        but uses ~3-4GB of VRAM instead of ~19GB.
        """
        dtype = torch.bfloat16

        if text_ids is None and input_ids is not None:
            text_ids = input_ids
        if text_ids is None:
            raise ValueError("text_ids or input_ids is required")

        text_ids = text_ids.to(self.cpu_device)
        batch_size = text_ids.shape[0]

        # Special token IDs
        speech_start_id = getattr(self.config, "speech_start_id", None) or 151652
        speech_end_id = getattr(self.config, "speech_end_id", None) or 151653
        speech_diffusion_id = getattr(self.config, "speech_diffusion_id", None) or 151654
        eos_token_id = getattr(self.config.decoder_config, "eos_token_id", None) or 151643

        # Initialize streaming cache for acoustic tokenizer
        acoustic_cache_streaming = KugelAudioTokenizerStreamingCache()

        # Get initial embeddings (embed layer → GPU → CPU)
        inputs_embeds = self._embed_tokens(text_ids)

        # Process voice features if provided
        if voice_cache is not None:
            _, speech_embeds = self._process_speech_inputs(voice_cache)
            if speech_input_mask is not None:
                speech_input_mask_device = speech_input_mask.to(self.device)
                inputs_embeds[speech_input_mask_device] = speech_embeds

        # Setup for CFG
        current_ids = text_ids.to(self.device)
        attention_mask = torch.ones_like(current_ids)

        negative_ids = torch.full(
            (batch_size, 1), speech_start_id, dtype=torch.long, device=self.device
        )
        negative_attention_mask = torch.ones_like(negative_ids)
        negative_inputs_embeds = self._embed_tokens(negative_ids)

        # Token constraint
        valid_tokens = [speech_start_id, speech_end_id, speech_diffusion_id, eos_token_id]
        token_constraint = KugelAudioTokenConstraintProcessor(valid_tokens, device=self.device)

        # Storage
        audio_chunks = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        correct_cnt = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # KV caches (stored on CPU between steps)
        past_key_values = None
        negative_past_key_values = None

        progress_iter = (
            tqdm(range(max_new_tokens), desc="Generating (low-VRAM)", leave=False)
            if show_progress
            else range(max_new_tokens)
        )

        for step in progress_iter:
            if finished.all():
                break

            # ── LM forward pass (GPU) ────────────────────────────────
            if past_key_values is None:
                logits, last_hidden, past_key_values = self._lm_forward(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    use_cache=True,
                )
            else:
                logits, last_hidden, past_key_values = self._lm_forward(
                    inputs_embeds=inputs_embeds[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            logits = logits[:, -1, :].to(self.device)
            logits = token_constraint(current_ids, logits)

            # Sample or greedy decode
            if do_sample and temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tokens = torch.argmax(logits, dim=-1)

            next_tokens = torch.where(
                finished, torch.tensor(eos_token_id, device=self.device), next_tokens
            )

            current_ids = torch.cat([current_ids, next_tokens.unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((batch_size, 1), device=self.device, dtype=attention_mask.dtype),
                ],
                dim=-1,
            )

            # Check for EOS / speech_end
            eos_mask = (next_tokens == eos_token_id) & ~finished
            if eos_mask.any():
                finished = finished | eos_mask

            speech_end_mask = (next_tokens == speech_end_id) & ~finished
            if speech_end_mask.any():
                finished = finished | speech_end_mask
                speech_end_indices = speech_end_mask.nonzero(as_tuple=False).squeeze(-1)
                acoustic_cache_streaming.set_to_zero(speech_end_indices)

            # Handle speech_start tokens
            speech_start_mask = (next_tokens == speech_start_id) & ~finished
            if (
                speech_start_mask.any()
                and cfg_scale != 1.0
                and negative_past_key_values is not None
            ):
                speech_start_indices = speech_start_mask.nonzero(as_tuple=False).squeeze(-1)
                if speech_start_indices.dim() == 0:
                    speech_start_indices = speech_start_indices.unsqueeze(0)

                # Move negative cache to device temporarily
                negative_past_key_values = self._move_cache_to(
                    negative_past_key_values, self.device
                )
                for sample_idx in speech_start_indices.tolist():
                    negative_attention_mask[sample_idx, :] = 0
                    negative_attention_mask[sample_idx, -1] = 1

                    key_caches, value_caches = _get_cache_tensors(negative_past_key_values)
                    for k_cache, v_cache in zip(key_caches, value_caches):
                        k_cache[sample_idx, :, -1, :] = k_cache[sample_idx, :, 0, :].clone()
                        v_cache[sample_idx, :, -1, :] = v_cache[sample_idx, :, 0, :].clone()

                    negative_ids[sample_idx, -1] = speech_start_id
                negative_past_key_values = self._move_cache_to(
                    negative_past_key_values, self.cpu_device
                )

            # Prepare next embeddings
            next_inputs_embeds = (
                self._embed_tokens(next_tokens.unsqueeze(0)).squeeze(0).unsqueeze(1)
            )

            # ── Handle diffusion tokens ──────────────────────────────
            diffusion_mask = (next_tokens == speech_diffusion_id) & ~finished
            if diffusion_mask.any():
                diffusion_indices = diffusion_mask.nonzero(as_tuple=False).squeeze(-1)
                if diffusion_indices.dim() == 0:
                    diffusion_indices = diffusion_indices.unsqueeze(0)

                # Negative LM forward for CFG
                if cfg_scale != 1.0:
                    if negative_past_key_values is None:
                        neg_logits, neg_last_hidden, negative_past_key_values = self._lm_forward(
                            inputs_embeds=negative_inputs_embeds,
                            attention_mask=negative_attention_mask,
                            past_key_values=None,
                            use_cache=True,
                        )
                    else:
                        neg_logits, neg_last_hidden, negative_past_key_values = self._lm_forward(
                            inputs_embeds=negative_inputs_embeds[:, -1:],
                            attention_mask=negative_attention_mask,
                            past_key_values=negative_past_key_values,
                            use_cache=True,
                        )

                    # Handle non-diffusion samples KV cache correction
                    non_diffusion_mask = ~diffusion_mask & ~finished
                    if non_diffusion_mask.any():
                        non_diffusion_indices = non_diffusion_mask.nonzero(as_tuple=False).squeeze(
                            -1
                        )
                        if non_diffusion_indices.dim() == 0:
                            non_diffusion_indices = non_diffusion_indices.unsqueeze(0)

                        negative_past_key_values = self._move_cache_to(
                            negative_past_key_values, self.device
                        )
                        key_caches, value_caches = _get_cache_tensors(negative_past_key_values)
                        for sample_idx in non_diffusion_indices.tolist():
                            start_idx = correct_cnt[sample_idx].item()
                            seq_len = negative_attention_mask.shape[1]

                            if start_idx + 1 < seq_len - 1:
                                negative_attention_mask[sample_idx, start_idx + 1 :] = (
                                    negative_attention_mask[sample_idx, start_idx:-1].clone()
                                )
                            negative_attention_mask[sample_idx, start_idx] = 0

                            for k_cache, v_cache in zip(key_caches, value_caches):
                                if start_idx + 1 < k_cache.shape[2] - 1:
                                    k_cache[sample_idx, :, start_idx + 1 :, :] = k_cache[
                                        sample_idx, :, start_idx:-1, :
                                    ].clone()
                                    v_cache[sample_idx, :, start_idx + 1 :, :] = v_cache[
                                        sample_idx, :, start_idx:-1, :
                                    ].clone()

                            if start_idx + 1 < negative_ids.shape[1] - 1:
                                negative_ids[sample_idx, start_idx + 1 :] = negative_ids[
                                    sample_idx, start_idx:-1
                                ].clone()

                        correct_cnt[non_diffusion_indices] += 1
                        negative_past_key_values = self._move_cache_to(
                            negative_past_key_values, self.cpu_device
                        )

                    neg_condition = neg_last_hidden[diffusion_indices, -1, :].to(self.device)
                else:
                    neg_condition = torch.zeros(
                        diffusion_indices.shape[0],
                        self.config.decoder_config.hidden_size,
                        device=self.device,
                        dtype=dtype,
                    )

                condition = last_hidden[diffusion_indices, -1, :].to(self.device)

                # ── Diffusion sampling (GPU) ─────────────────────────
                speech_latents = self._sample_speech_tokens(condition, neg_condition, cfg_scale)

                # Unscale latents
                scaled_latent = speech_latents / self._scaling_factor - self._bias_factor

                # ── Audio decode (GPU) ───────────────────────────────
                audio = self._decode_audio(
                    scaled_latent, acoustic_cache_streaming, diffusion_indices
                )

                # Store audio chunks
                for i, idx in enumerate(diffusion_indices.tolist()):
                    if not finished[idx]:
                        audio_chunks[idx].append(audio[i].cpu())

                # Compute embeddings for next step
                _move_to(self._connector, self.device)
                acoustic_embed = self._connector(speech_latents.unsqueeze(1))
                diffusion_embeds = acoustic_embed.squeeze(1)
                _move_to(self._connector, self.cpu_device)

                next_inputs_embeds[diffusion_indices] = diffusion_embeds.unsqueeze(1)

            # Update embeddings
            inputs_embeds = torch.cat([inputs_embeds, next_inputs_embeds], dim=1)
            negative_inputs_embeds = torch.cat([negative_inputs_embeds, next_inputs_embeds], dim=1)
            negative_attention_mask = torch.cat(
                [
                    negative_attention_mask,
                    torch.ones(
                        (batch_size, 1), device=self.device, dtype=negative_attention_mask.dtype
                    ),
                ],
                dim=-1,
            )
            negative_ids = torch.cat([negative_ids, next_tokens.unsqueeze(-1)], dim=-1)

        # ── Concatenate audio chunks ─────────────────────────────────
        speech_outputs = []
        for chunks in audio_chunks:
            if chunks:
                concatenated = torch.cat(chunks, dim=-1).squeeze()
                max_val = concatenated.abs().max()
                if max_val > 1.0:
                    concatenated = concatenated * (0.95 / max_val)
                # Apply watermark
                concatenated = self._apply_watermark(concatenated, sample_rate=24000)
                speech_outputs.append(concatenated)
            else:
                speech_outputs.append(None)

        return KugelAudioGenerationOutput(
            sequences=current_ids,
            speech_outputs=speech_outputs,
        )

    def _apply_watermark(self, audio: torch.Tensor, sample_rate: int = 24000) -> torch.Tensor:
        """Apply watermark using the underlying model's method."""
        return self.model._apply_watermark(audio, sample_rate=sample_rate)


def _get_available_ram_gb() -> float:
    """Get available system RAM in GB."""
    try:
        import psutil

        return psutil.virtual_memory().available / (1024**3)
    except ImportError:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024**2)  # kB to GB
    except (OSError, ValueError):
        pass
    return float("inf")  # Unknown, assume plenty


def load_model_low_vram(
    model_id: str = "kugelaudio/kugelaudio-0-open",
    device: str = "cuda",
    pin_memory: bool = True,
) -> LowVRAMInferenceWrapper:
    """Load KugelAudio model for low-VRAM inference.

    Loads the model in bfloat16 on CPU, strips encoder weights,
    and wraps it for component-level GPU offloading.

    This requires ~15GB of system RAM but only ~3-4GB of VRAM.
    If you have less than 18GB of RAM, use load_model_quantized() instead.

    Args:
        model_id: HuggingFace model ID or local path.
        device: GPU device to use for computation (default: "cuda").
        pin_memory: Whether to pin CPU memory for faster transfers.

    Returns:
        LowVRAMInferenceWrapper ready for generation.

    Example:
        >>> from kugelaudio_open.models.low_vram import load_model_low_vram
        >>> wrapper = load_model_low_vram("kugelaudio/kugelaudio-0-open")
        >>> processor = KugelAudioProcessor.from_pretrained("kugelaudio/kugelaudio-0-open")
        >>> inputs = processor(text="Hello!", voice="default", return_tensors="pt")
        >>> outputs = wrapper.generate(**inputs, cfg_scale=3.0)
        >>> processor.save_audio(outputs.speech_outputs[0], "output.wav")
    """
    available_ram = _get_available_ram_gb()
    if available_ram < 18:
        logger.warning(
            f"Only {available_ram:.1f}GB RAM available. "
            f"low-vram mode needs ~15GB RAM for the bf16 weights on CPU. "
            f"Consider using --quantize for 4-bit quantization (~4GB RAM + ~4GB VRAM)."
        )

    logger.info(f"Loading model {model_id} for low-VRAM inference...")
    logger.info("Model will be loaded on CPU with bfloat16 precision.")

    model = KugelAudioForConditionalGenerationInference.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        # Load to CPU - no .to(device)
    )
    model.eval()
    model.model.strip_encoders()

    # Auto-disable pin_memory if RAM is tight (pinned pages can't be swapped)
    if pin_memory and available_ram < 18:
        logger.warning("Disabling memory pinning due to low available RAM.")
        pin_memory = False

    wrapper = LowVRAMInferenceWrapper(model, device=device, pin_memory=pin_memory)

    logger.info("Low-VRAM model ready. VRAM usage: ~3-4GB during generation.")
    return wrapper


def load_model_quantized(
    model_id: str = "kugelaudio/kugelaudio-0-open",
    device: str = "cuda",
) -> KugelAudioForConditionalGenerationInference:
    """Load KugelAudio model with 4-bit quantization (bitsandbytes NF4).

    Uses device_map="auto" to distribute layers between GPU and CPU based
    on available VRAM. Quantized transformer layers go to GPU first; any
    overflow (embeddings, encoder weights) stays on CPU and is moved on
    the fly by accelerate hooks.

    This is the best option when you have limited RAM (<16GB) and a GPU
    with 6GB+ VRAM.

    Requires: pip install bitsandbytes

    Args:
        model_id: HuggingFace model ID or local path.
        device: GPU device to use (default: "cuda").

    Returns:
        KugelAudioForConditionalGenerationInference with quantized weights.

    Example:
        >>> from kugelaudio_open.models.low_vram import load_model_quantized
        >>> model = load_model_quantized("kugelaudio/kugelaudio-0-open")
        >>> processor = KugelAudioProcessor.from_pretrained("kugelaudio/kugelaudio-0-open")
        >>> inputs = processor(text="Hello!", voice="default", return_tensors="pt")
        >>> inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
        >>> outputs = model.generate(**inputs, cfg_scale=3.0)
        >>> processor.save_audio(outputs.speech_outputs[0], "output.wav")
    """
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        raise ImportError(
            "4-bit quantization requires bitsandbytes. Install it with: pip install bitsandbytes"
        )

    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        raise ImportError(
            "4-bit quantization requires bitsandbytes. Install it with: pip install bitsandbytes"
        )

    logger.info(f"Loading model {model_id} with 4-bit quantization (NF4)...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,  # nested quantization saves a bit more
        llm_int8_enable_fp32_cpu_offload=True,  # allow CPU offload for modules that don't fit GPU
    )

    # Detect available GPU memory and reserve headroom for quantization overhead
    # and inference (KV cache, activations, diffusion sampling, audio decode).
    # During 4-bit weight loading, PyTorch needs temporary GPU memory for
    # dequantization buffers, so we reserve more on smaller GPUs.
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        # Reserve ~40% on small GPUs (<=8GB), ~25% on larger ones
        if gpu_mem_gb <= 8:
            headroom = gpu_mem_gb * 0.4
        else:
            headroom = gpu_mem_gb * 0.25
        max_gpu = f"{max(gpu_mem_gb - headroom, 2.0):.1f}GiB"
        logger.info(
            f"GPU has {gpu_mem_gb:.1f}GB total, limiting model to {max_gpu} "
            f"(reserving {headroom:.1f}GB for quantization + inference)"
        )
    else:
        max_gpu = "4GiB"

    # Compute device map explicitly so we can add scalar buffers that
    # infer_auto_device_map misses (speech_scaling_factor, speech_bias_factor).
    from accelerate import infer_auto_device_map, init_empty_weights

    from ..configs import KugelAudioConfig

    config = KugelAudioConfig.from_pretrained(model_id)
    max_memory = {0: max_gpu, "cpu": "24GiB"}
    with init_empty_weights():
        empty_model = KugelAudioForConditionalGenerationInference(config)
    device_map = infer_auto_device_map(
        empty_model,
        max_memory=max_memory,
        no_split_module_classes=empty_model._no_split_modules or [],
    )
    # Scalar buffers registered on KugelAudioModel are not auto-mapped
    for buf_key in ["model.speech_bias_factor", "model.speech_scaling_factor"]:
        if buf_key not in device_map:
            device_map[buf_key] = 0
    del empty_model

    model = KugelAudioForConditionalGenerationInference.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map=device_map,
        max_memory=max_memory,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    model.model.strip_encoders()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Log memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        logger.info(f"Quantized model loaded. GPU memory used: {allocated:.1f}GB")

    return model


__all__ = [
    "LowVRAMInferenceWrapper",
    "load_model_low_vram",
    "load_model_quantized",
]
