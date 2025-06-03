import torch
from PIL import Image
from model import YoloV2Net
from utils import filtered_boxes,draw_boxes  # 你需自行實作 decode + NMS

def adjust_state_dict_keys(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        # 若開頭是 main1 或 main2，補上 darknet.
        if k.startswith('main1') or k.startswith('main2'):
            new_key = f'darknet.{k}'
        elif k.startswith('conv1') or k.startswith('conv2') or k.startswith('conv'):
            new_key = k  # 這些不改
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict

def main():
    device = torch.device("cpu")

    # Step 1: Load model
    # model = YoloV2Net(num_classes=20)  # VOC = 20 類
    # state_dict = torch.load("./yolov2_pot2_int8.pth", map_location=device)
    # state_dict = adjust_state_dict_keys(state_dict)
    model = YoloV2Net(num_classes=20)
    model.load_state_dict(torch.load("yolov2_pot2_weightonly.pth"))
    print("✅ 已載入模型：yolov2_pot2_int8.pt !!!"*2)
    print(type(model))
    model.eval().to(device)

    # Step 2: Load input image
    img = Image.open("test.jpg").convert("RGB").resize((608, 608))
    boxes = filtered_boxes(model, device, img, conf_thresh=0.3, nms_thresh=0.5)

    image_with_boxes = draw_boxes(img.copy(), boxes)
    image_with_boxes.save("output_quantized.jpg")
    print("✅ 已儲存推論結果至：output_quantized.jpg")

if __name__ == "__main__":
    main()
