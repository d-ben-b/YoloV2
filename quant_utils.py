import torch
import numpy as np

def quantize_tensor_power2(x, num_bits=8):
    """
    Quantize tensor to power of 2 values.
    
    Args:
        x: Input tensor
        num_bits: Number of bits for quantization
    
    Returns:
        Quantized tensor, scale factor
    """
    # Compute the value range
    max_val = torch.max(torch.abs(x)).item()
    
    # Avoid log2(0)
    if max_val == 0:
        return x, 1.0
    
    qmax = 2 ** (num_bits - 1) - 1  # 127 for int8
    
    # 1) 取得絕對最大值 ➜ 求原始 scale
    scale = x.abs().max() / qmax
    
    # 2) 把 scale round 成 2 的 n 次方
    scale = 2 ** torch.round(torch.log2(scale + 1e-12))
    
    # 3) 量化 → 反量化
    tensor_q = torch.clamp(torch.round(x / scale), -qmax, qmax)
    tensor_dq = tensor_q * scale
    
    return tensor_dq, scale

def quantize_tensor_linear(x, num_bits=8):
    """
    Linear quantization of tensor.
    May give better results than power-of-2 quantization in some cases.
    
    Args:
        x: Input tensor
        num_bits: Number of bits for quantization
    
    Returns:
        Quantized tensor, scale factor
    """
    # Determine range
    min_val = x.min().item()
    max_val = x.max().item()
    
    # Compute scale and zero_point for symmetric quantization
    max_abs = max(abs(min_val), abs(max_val))
    scale = max_abs / (2 ** (num_bits - 1) - 1)
    
    # If scale is 0, return the original tensor
    if scale == 0:
        return x, scale
    
    # Quantize
    x_q = torch.round(x / scale) * scale
    
    return x_q, scale