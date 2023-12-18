from typing import Tuple
import torch
import torch.nn as nn

# Python Wrapper
import flash_attn_2_cuda as flash_attn_cuda

def get_block_size(device: torch.device, head_dim: int, use_dropout: bool, causal_mode: bool) -> Tuple[int, int]:
    """
    Determines the block size for attention calculation based on device capabilities and input dimensions.
    """
    assert head_dim <= 256
    major, minor = torch.cuda.get_device_capability(device)
    is_sm8x = major == 8 and minor > 0
    is_sm80 = major == 8 and minor == 0
    is_sm90 = major == 9 and minor == 0

    if head_dim <= 32:
        return 128, 128
    elif head_dim <= 64:
        return (128, 128) if not use_dropout else (128, 64)
    elif head_dim <= 96:
        return (64, 64) if (is_sm8x and causal_mode) else (128, 64)
    elif head_dim <= 128:
        return (64, 64) if (is_sm8x and not use_dropout and causal_mode) else (128, 32)
    elif head_dim <= 160:
        return (128, 64) if is_sm8x and not causal_mode else (64, 64)
    elif head_dim <= 192:
        return (128, 64) if not use_dropout else (64, 64)
    elif head_dim <= 224:
        return (128, 64) if (is_sm80 or is_sm90) else (64, 64)
    elif head_dim <= 256:
        return (128, 64) if is_sm80 else (64, 64)

def make_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous() if tensor.stride(-1) != 1 else tensor

def forward_attention(q, k, v, dropout_prob, softmax_scaling, causal, wnd_size, return_softmax):
    q, k, v = map(make_contiguous, [q, k, v])
    return flash_attn_cuda.fwd(
        q, k, v, None, dropout_prob, softmax_scaling, causal, wnd_size[0], wnd_size[1], return_softmax, None)

def backward_attention(dout, q, k, v, out, softmax_lse, dq, dk, dv, dropout_prob, softmax_scaling, causal, wnd_size, rng_state=None):
    dout, q, k, v, out = map(make_contiguous, [dout, q, k, v, out])
    return flash_attn_cuda.bwd(
        dout, q, k, v, out, softmax_lse, dq, dk, dv, dropout_prob, softmax_scaling, causal, wnd_size[0], wnd_size[1], None, rng_state)

class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, dropout_prob, softmax_scale, causal, wnd_size, return_softmax):
        softmax_scale = softmax_scale or q.shape[-1] ** (-0.5)
        results = forward_attention(q, k, v, dropout_prob, softmax_scale, causal, wnd_size, return_softmax)
        out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = results
        ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
        ctx.intermediate_results = (dropout_prob, softmax_scale, causal, wnd_size)
        return (out, softmax_lse, S_dmask) if return_softmax else out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
        dropout_prob, softmax_scale, causal, wnd_size = ctx.intermediate_results
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        dq, dk, dv, softmax_d = backward_attention(
            dout, q, k, v, out, softmax_lse, dq, dk, dv, dropout_prob, softmax_scale, causal, wnd_size, rng_state)
        dq, dk, dv = dq[..., :dout.shape[-1]], dk[..., :dout.shape[-1]], dv[..., :dout.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None

def flash_attention(q, k, v, dropout_prob=0.0, softmax_scale=None, causal=False, wnd_size=(-1, -1), return_attn_probs=False):
    return FlashAttentionFunction.apply(q, k, v, dropout_prob, softmax_scale, causal, wnd_size, return_attn_probs)
