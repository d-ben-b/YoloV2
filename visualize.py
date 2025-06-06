import os
import cv2
import math
import time
import torch
from PIL import Image, ImageDraw
from model import YoloV2Net
from utils import *
from PTQ import po2_qconfig       # ← 第 2 步的 qconfig 定義


def plot_boxes(img, boxes, savename=None, class_names=None, color=None):
    colors = torch.FloatTensor([
        [1, 0, 1], [0, 0, 1], [0, 1, 1],
        [0, 1, 0], [1, 1, 0], [1, 0, 0]
        ]);
    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)
 
    width  = img.width
    height = img.height
    draw = ImageDraw.Draw(img)
    detections = []
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = (box[0] - box[2] / 2.0) * width
        y1 = (box[1] - box[3] / 2.0) * height
        x2 = (box[0] + box[2] / 2.0) * width
        y2 = (box[1] + box[3] / 2.0) * height
        rgb = color if color else(255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id   = box[6]
            detections += [(cls_conf, class_names[cls_id])]
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red   = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue  = get_color(0, offset, classes)
            rgb = (red, green, blue)
            draw.rectangle([x1, y1 - 15, x1 + 6.5 * len(class_names[cls_id]), y1], fill=rgb)
            draw.text((x1 + 2, y1 - 13), class_names[cls_id], fill=(0, 0, 0))
        draw.rectangle([x1, y1, x2, y2], outline=rgb, width=3)
    for(cls_conf, class_name) in sorted(detections, reverse=True):
        print('%-10s: %f' %(class_name, cls_conf))
    if savename:
        print('save plot results to %s' %savename)
        img.save(savename)
    return img


def plot_boxes_cv2(img, boxes, savename=None, class_names=None, color=None):
    colors = torch.FloatTensor([
        [1, 0, 1], [0, 0, 1], [0, 1, 1],
        [0, 1, 0], [1, 1, 0], [1, 0, 0]
        ]);
    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)
 
    width  = img.shape[1]
    height = img.shape[0]
   
    detections = []
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int((box[0] - box[2] / 2.0) * width)
        y1 = int((box[1] - box[3] / 2.0) * height)
        x2 = int((box[0] + box[2] / 2.0) * width)
        y2 = int((box[1] + box[3] / 2.0) * height)
        rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id   = box[6]
            detections += [(cls_conf, class_names[cls_id])]
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red   = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue  = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            cv2.rectangle(img, (x1, y1 - 17), (x1 + 8 * len(class_names[cls_id]) + 10, y1), rgb, -1)
            cv2.putText(img, class_names [cls_id], (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 2)
    for(cls_conf, class_name) in sorted(detections, reverse=True):
        print('%-10s: %f' %(class_name, cls_conf))
    if savename:
        print('save plot results to %s' %savename)
        cv2.imwrite(savename, img)
    return savename

def predict(model, conf_thresh, nms_thresh, img_path, class_names, device, save_to=None):
    assert os.path.exists(img_path), 'Error! Input image does not exists.'
    model.eval()
    img = Image.open(img_path).convert('RGB')
 
    tic = time.time()
    boxes = filtered_boxes(model, device, img.resize((608, 608)), conf_thresh, nms_thresh)
 
    toc = time.time()
    print('Prediction took {:.5f} ms.'.format((toc - tic) * 1000))
    pred_img = plot_boxes(img, boxes, save_to, class_names)
   
    return pred_img


def predict_cv2(model, conf_thresh, nms_thresh, img_path, class_names, device, save_to=None):
    assert os.path.exists(img_path), 'Error! Input image does not exists.'
    model.eval()
    img = cv2.imread(img_path)
 
    tic = time.time()
    boxes = filtered_boxes(model, device, cv2.resize(img, (608, 608)), conf_thresh, nms_thresh)
 
    toc = time.time()
    print('Prediction took {:.5f} ms.'.format((toc - tic) * 1000))
    pred_img = plot_boxes_cv2(img, boxes, save_to, class_names)
   
    return pred_img

def predict_ptq(model_path, conf_thresh, nms_thresh, img_path, class_names, device, save_to=None, num_classes=80):
    """
    Perform prediction using a quantized YOLOv2 model.
    
    Args:
        model_path: Path to the quantized model weights (.pth)
        conf_thresh: Confidence threshold for detections
        nms_thresh: Non-maximum suppression threshold
        img_path: Path to the input image
        class_names: List of class names
        device: Device to run inference on ('cpu' or 'cuda')
        save_to: Path to save the output image (optional)
        num_classes: Number of classes (default: 80 for VOC)
    
    Returns:
        PIL.Image: Annotated image with bounding boxes
    """
    assert os.path.exists(img_path), 'Error! Input image does not exists.'
    assert os.path.exists(model_path), 'Error! Model file does not exists.'
    
    # Load the model
    from model import YoloV2Net
    model = YoloV2Net(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load and prepare image
    img = Image.open(img_path).convert('RGB')
    
    # Measure inference time
    tic = time.time()
    boxes = filtered_boxes(model, device, img.resize((608, 608)), conf_thresh, nms_thresh)
    toc = time.time()
    
    print('PTQ prediction took {:.5f} ms.'.format((toc - tic) * 1000))
    print(f'Using quantized model: {model_path}')
    
    # Draw bounding boxes on image
    # Add this before calling plot_boxes or draw_boxes
    print(f"Number of detected boxes: {len(boxes)}")
    for i, box in enumerate(boxes[:5]):  # Print first 5 boxes
        print(f"Box {i}: {[float('{:.4f}'.format(x)) for x in box]}")
    pred_img = plot_boxes(img, boxes, save_to, class_names)
    
    if save_to:
        pred_img.save(save_to)
    
    return pred_img

def load_int8_model(state_dict_path: str,
                    num_classes: int = 80,
                    device: str | torch.device = 'cpu') -> torch.nn.Module:

    # 1. 建立 FP32 結構 & fuse
    model = YoloV2Net(num_classes=num_classes).eval()
    model.fuse_model()

    # 2. 指定跟訓練時一致的 qconfig，並做 prepare → convert
    model.qconfig = po2_qconfig
    import torch.ao.quantization as tq
    model_prepared = tq.prepare(model, inplace=False)
    model_int8     = tq.convert(model_prepared, inplace=False)

    # 3. 載入剛才存下來的 int8 權重
    sd = torch.load(state_dict_path, map_location='cpu')
    model_int8.load_state_dict(sd, strict=True)

    model_int8.eval().to(device)
    return model_int8

# ---------- 對外的推論 API ----------
@torch.inference_mode()
def predict_quant(img_path      : str,
                  model_weights : str,
                  conf_thresh   : float = 0.3,
                  nms_thresh    : float = 0.5,
                  class_names   : list[str] | None = None,
                  device        : str | torch.device = 'cpu',
                  save_path     : str | None = None):
    """
    img_path      : 要推論的影像路徑
    model_weights : yolov2_int8_fbgemm.pth
    """
    assert os.path.isfile(img_path),   f"❌ {img_path} not found"
    assert os.path.isfile(model_weights), f"❌ {model_weights} not found"
    device = torch.device(device)

    # 1. 載入量化模型
    model = load_int8_model(model_weights, num_classes=80, device=device)
    print("✅ 已載入 int8 模型")

    # 2. 預處理影像
    pil_img = Image.open(img_path).convert("RGB")
    inp_img = pil_img.resize((608, 608))

    # 3. 推論
    t0 = time.perf_counter()
    boxes = filtered_boxes(model, device, inp_img,
                           conf_thresh=conf_thresh, nms_thresh=nms_thresh)
    t1 = time.perf_counter()
    print(f"⏱️  {1000*(t1-t0):.2f} ms,  detect {len(boxes)} boxes")

    # 4. 繪製 & 儲存
    out_img = plot_boxes(pil_img, boxes,save_path, class_names)
    print(f"📷  已存檔：{save_path}")
    return out_img