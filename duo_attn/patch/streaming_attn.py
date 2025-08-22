import torch

try:
    import xformers.ops as xops
except ImportError:
    xops = None

try:
    from block_sparse_attn import block_streaming_attn_func
except ImportError:
    block_streaming_attn_func = None

from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, repeat_kv, logger, Optional, Cache, Tuple


@torch.no_grad()
def generate_streaming_mask(seq_len, sink_size, recent_size, device):
    # round seq_len to the nearest multiple of 8
    seq_len = (seq_len + 7) // 8 * 8
    ones = torch.ones(seq_len, seq_len, dtype=torch.bool)
    causal_mask = ~torch.triu(ones, diagonal=1)
    recent_mask = torch.triu(ones, diagonal=-recent_size + 1)
    sink_mask = ones
    sink_mask[:, sink_size:] = False
    mask = (recent_mask | sink_mask) & causal_mask
    return mask.to(device=device).unsqueeze(0).unsqueeze(0)


def streaming_attn_sdpa(query_states, key_states, value_states, streaming_causal_mask):
    bsz, seq_len, num_heads, head_dim = query_states.size()

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    streaming_attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=streaming_causal_mask[:, :, :seq_len, :seq_len],
        dropout_p=0.0,
        enable_gqa=True,
    )

    return streaming_attn_output.transpose(1, 2)


def streaming_attn_xformers(
    query_states, key_states, value_states, streaming_causal_mask
):
    # query_states: [bsz, seq_len, num_heads, head_dim]
    # key_states: [bsz, seq_len, num_heads, head_dim]
    # value_states: [bsz, seq_len, num_heads, head_dim]
    # Return: [bsz, seq_len, num_heads, head_dim]

    bsz, seq_len, num_heads, head_dim = query_states.size()
    attn_bias = streaming_causal_mask[:, :, :seq_len, :seq_len].expand(
        bsz, num_heads, seq_len, seq_len
    )

    streaming_attn_output = xops.memory_efficient_attention(
        query_states,
        key_states,
        value_states,
        attn_bias=attn_bias,
        p=0.0,
    )

    return streaming_attn_output


def generate_streaming_info_blocksparse_flash_attn(
    sink_block_num, local_block_num, n_query_heads, device
):
    streaming_info = torch.tensor(
        [sink_block_num, local_block_num] * n_query_heads,
        device=device,
        dtype=torch.int32,
    )
    return streaming_info


def streaming_attn_blocksparse_flash_attn(
    query_states, key_states, value_states, streaming_info
):
    bts, seqlen, query_heads, head_dim = query_states.size()
    key_value_heads = key_states.size(2)
    query_unpad = query_states.view(bts * seqlen, query_heads, head_dim)
    key_unpad = key_states.view(bts * seqlen, key_value_heads, head_dim)
    value_unpad = value_states.view(bts * seqlen, key_value_heads, head_dim)
    cu_seqlens = torch.arange(
        0, (bts + 1) * seqlen, step=seqlen, dtype=torch.int32, device=query_unpad.device
    )
    head_mask_type = torch.tensor(
        [-1] * query_heads, device=query_unpad.device, dtype=torch.int32
    )
    attn_output = block_streaming_attn_func(
        query_unpad,
        key_unpad,
        value_unpad,
        cu_seqlens,
        cu_seqlens,
        head_mask_type,
        streaming_info,
        seqlen,
        seqlen,
        p_dropout=0.0,
        is_causal=True,
    )
    return attn_output.reshape(bts, seqlen, query_heads, head_dim)


def streaming_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    is_causal = True if causal_mask is None and q_len > 1 else False

    if q_len > 1:
        rng = torch.arange(q_len, dtype=int, device='cuda')
        mask = rng[None, :] > rng[:, None] - 1024
        mask |= rng[None, :] < 64
        mask &= rng[:, None] >= rng[None, :]
        mask = mask[None, None, :, :]
    else:
        kv_length = key_states.shape[2]
        rng = torch.arange(kv_length, dtype=int, device='cuda')
        mask = (rng > kv_length - 1024) | (rng < 64)
        mask = mask[None, None, None, :]

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, -1)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value
