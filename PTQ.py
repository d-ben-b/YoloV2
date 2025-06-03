import torch, torch.nn as nn
import numpy as np
from model import YoloV2Net, load_weights
from quant_utils import quantize_tensor_power2     # 第 1 步抽出來的 util

# Add debugging function
def analyze_tensor(tensor, name):
    print(f"--- {name} ---")
    print(f"Shape: {tensor.shape}")
    print(f"Range: [{tensor.min().item():.6f}, {tensor.max().item():.6f}]")
    print(f"Mean: {tensor.mean().item():.6f}, Std: {tensor.std().item():.6f}")
    # Check if there are any NaN or Inf values
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print("Warning: NaN or Inf values detected!")
    print()

model_fp32 = YoloV2Net(num_classes=20)
load_weights(model_fp32, "weights/yolov2.weights")
with torch.no_grad():
    for m in model_fp32.modules():
        if isinstance(m, torch.nn.Conv2d):
            # Debug original weights
            analyze_tensor(m.weight.data, f"Original weights for {m}")
            
            # Try with 16 bits instead of 8 for more precision
            w_q, scale = quantize_tensor_power2(m.weight.data, num_bits=16)
            
            # Debug quantized weights
            analyze_tensor(w_q, f"Quantized weights for {m}")
            print(f"Quantization scale: {scale}")
            
            m.weight.data.copy_(w_q)          # 只改 weight, bias 直接保留 FP32

torch.save(model_fp32.state_dict(), "yolov2_pot2_weightonly.pth")
print("✓ PoT weight-only 量化完成 (仍可跑 GPU)")

# Optionally add a simple validation check
try:
    dummy_input = torch.randn(1, 3, 416, 416)
    output = model_fp32(dummy_input)
    print("Model inference successful")
except Exception as e:
    print(f"Error during inference: {e}")