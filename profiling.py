# benchmark.py
import argparse
from time import perf_counter
from tqdm import trange 
import torch
from torch.profiler import profile, record_function, ProfilerActivity

from model import YoloV2Net, load_weights          # 你的原始模型 + Darknet ↔︎ YOLOv2 權重載入
from PTQ     import po2_qconfig                    # 如果要自己做 Power-of-2 量化可用
import torch.ao.quantization as tq


# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YOLOv2 ── FP32 / int8 量化模型 profiler")
    p.add_argument("model_path",
                   help="模型權重路徑。int8 給 .pth / .pt；FP32 給 darknet .weights")
    p.add_argument("-b", "--backend",
                   choices=["fp32", "fbgemm", "qnnpack", "power2"],
                   default="fp32",
                   help="量化後端 / 精度；預設 fp32（不量化）")
    p.add_argument("--device",
                   choices=["auto", "cpu", "cuda"], default="auto",
                   help="推論裝置，auto 會自行偵測 GPU")
    p.add_argument("--img-size", type=int, default=608,
                   help="輸入解析度，預設 608×608")
    return p.parse_args()


# ---------- 取模型 ----------
def build_model(args):
    # A. 直接存整個模型 .pt  ➜  一行 torch.load
    if args.model_path.endswith(".pt"):
        return torch.load(args.model_path, map_location="cpu").eval()

    # B. FP32 .weights
    if args.backend == "fp32":
        m = YoloV2Net(num_classes=80); load_weights(m, args.model_path); return m.eval()

    # C. int8 「state-dict」 .pth  (已經是量化權重)
    torch.backends.quantized.engine = "qnnpack" if args.backend=="qnnpack" else "fbgemm"

    m = YoloV2Net(num_classes=80).eval(); m.fuse_model()

    m.qconfig = po2_qconfig if args.backend=="power2" else \
                tq.get_default_qconfig(torch.backends.quantized.engine)

    # ★ 不要 prepare→convert 再校正 —— 直接把 *convert* 後的「空殼」建立好
    m_q = tq.convert(tq.prepare(m, inplace=False), inplace=False)

    # ★ 接著 load_state_dict 就行；不再跑校正資料
    m_q.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    return m_q.eval()




# ---------- 主程式 ----------
@torch.inference_mode()
def main():
    args   = parse_args()
    device = torch.device("cpu")
    model  = build_model(args).to(device)
    x      = torch.randn(1, 3, args.img_size, args.img_size, device=device)

    # ① warm-up -------------------------------------------------------------
    for _ in trange(5, desc="Warm-up", ncols=80, colour="cyan"):
        model(x)

    # ② profiler ------------------------------------------------------------
    with profile(activities=[ProfilerActivity.CPU],
                 record_shapes=True,
                 profile_memory=True) as prof:
        with record_function("inference"):
            model(x)

    print("\nTop-5 layers / kernels by self-CPU-time")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

    # ③ latency loop --------------------------------------------------------
    t0 = perf_counter()
    for _ in trange(50, desc="Latency", ncols=80, colour="green"):
        model(x)
    print(f"\nAvg latency {(perf_counter()-t0)/50*1e3:.2f} ms  ({args.backend})")




if __name__ == "__main__":
    main()
