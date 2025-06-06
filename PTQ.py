import torch, time
from torch.ao.quantization import prepare, convert
from torch.ao.quantization.observer import (
    MinMaxObserver, PerChannelMinMaxObserver
)
from torch.ao.quantization.fake_quantize import FakeQuantize
from tqdm.auto import tqdm

# ---------- 1. 自訂 PoT Observer/FakeQuant ---------- #
class Po2Observer(MinMaxObserver):
    def _calculate_qparams(self, min_val, max_val):
        scale, zp = super()._calculate_qparams(min_val, max_val)
        scale = 2 ** torch.round(torch.log2(scale + 1e-12))
        return scale, zp

class Po2PerChObserver(PerChannelMinMaxObserver):
    def _calculate_qparams(self, min_val, max_val):
        scale, zp = super()._calculate_qparams(min_val, max_val)
        scale = 2 ** torch.round(torch.log2(scale + 1e-12))
        return scale, zp

class Po2FakeQuant(FakeQuantize):
    def __init__(self, **kwargs):
        super().__init__(observer=Po2Observer, **kwargs)

class Po2PerChFakeQuant(FakeQuantize):
    def __init__(self, **kwargs):
        super().__init__(observer=Po2PerChObserver, **kwargs)

# ---------- 2. QConfig ---------- #
po2_qconfig = torch.ao.quantization.QConfig(
    activation=Po2FakeQuant.with_args(
        dtype=torch.quint8, qscheme=torch.per_tensor_affine
    ),
    weight=Po2PerChFakeQuant.with_args(
        dtype=torch.qint8, qscheme=torch.per_channel_symmetric,
        ch_axis=0              # out-channel
    )
)

# ---------- 3. 建立 FP32 模型並 fuse ---------- #
from model import YoloV2Net, load_weights
from PIL import Image
import os

def main():
    model_fp32 = YoloV2Net(num_classes=80)
    load_weights(model_fp32, "weights/yolov2.weights")
    model_fp32.eval()
    model_fp32.fuse_model()
    print("✅ fuse 完成")

    # ---------- 4. prepare → calibration → convert ---------- #
    if torch.backends.quantized.supported_engines:          # 為保險起見列一下
        print("可用量化後端:", torch.backends.quantized.supported_engines)
    torch.backends.quantized.engine = "fbgemm"
    model_fp32.qconfig = po2_qconfig
    model_prepared = prepare(model_fp32, inplace=False)
    print("✅ prepare 完成（Observer 已插入）")

    # 這裡用假資料示範；實務上丟幾十張真圖就行
    with torch.no_grad():
        # Load and preprocess real images for calibration
        import torchvision.transforms as transforms
        
        # Set up image preprocessing
        transform = transforms.Compose([
            transforms.Resize((608, 608)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load images from images directory
        img_dir = "images"
        img_files = ["dog.jpg", "eagle.jpg", "giraffe.jpg"]
        images = []
        
        for img_file in img_files:
            try:
                img_path = os.path.join(img_dir, img_file)
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
                images.append(img_tensor)
                # Use for calibration
                model_prepared(img_tensor)
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
        
        # Ensure we have enough calibration data
        # Use tqdm with blue color for the remaining calibration iterations
        for _ in tqdm(range(20 - len(images)), desc="Calibration", colour="blue"):
            # Reuse existing images if we don't have enough
            idx = len(images) % len(img_files)
            model_prepared(images[idx])
        # for image in range(20):
        #     dummy = torch.randn(1, 3, 608, 608)
        #     model_prepared(dummy)
    print("✅ calibration 完成")

    model_int8 = convert(model_prepared, inplace=False).eval()
    model_int8.to("cpu")
    print("✅ convert 完成，模型已是 int8")

    # ---------- 5. 簡單 benchmark ---------- #
    x = torch.randn(1, 3, 608, 608)
    for _ in range(5): model_int8(x)          # warm-up
    t0 = time.perf_counter()
    for _ in range(20): model_int8(x)
    print(f"⏱  Avg latency {(time.perf_counter()-t0)/20*1e3:.2f} ms (CPU-int8)")

    # ---------- 6. 儲存 ---------- #
    torch.save(model_int8.state_dict(), "yolov2_int8_fbgemm.pth")   # 存整個模型
    print("💾 已儲存 yolov2_int8_fbgemm.pth")

if __name__ == "__main__":
    main()