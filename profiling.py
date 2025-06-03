import argparse

import torch
from torch.profiler import profile, record_function, ProfilerActivity

from model import YoloV2Net as Net
from model import load_model

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to the model file")
    parser.add_argument(
        "-b",
        "--backend",
        help="Quantization backend",
        choices=["qnnpack", "dyadic", "power2"],
    )
    return parser.parse_args()


def main():
    args = parse_args()

    backend = args.backend
    model_path = args.model_path

    in_channels, in_size = 5, 80
    if backend:
        # qconfig = CustomQConfig[backend.upper()].value
        fuse_modules = True
    else:
        qconfig = None
        fuse_modules = False
    model = load_model(
        Net(in_channels,in_size),
        model_path,
        qconfig=qconfig,
        fuse_modules=fuse_modules,
    ).to(DEFAULT_DEVICE)

    dummy_input = torch.randn(1, 3, 416, 416) 

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        with record_function("model_inference"):
            model(dummy_input)

    # Display the profile results
    print(
        prof.key_averages(group_by_input_shape=False).table(
            sort_by="cpu_time_total", row_limit=10
        )
    )


if __name__ == "__main__":
    main()
