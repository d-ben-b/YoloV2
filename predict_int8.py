# predict_int8.py
import os, time, math, cv2, torch
from PIL import Image, ImageDraw
from utils import filtered_boxes          # 你之前寫好的解碼+ NMS
from model import YoloV2Net       # ← 放到自己路徑

# ---------- 繪圖 ----------
def plot_boxes(img: Image.Image, boxes, class_names=None, rgb=(255, 0, 0)):
    draw = ImageDraw.Draw(img)
    w, h  = img.size
    for box in boxes:
        x1 = (box[0] - box[2] / 2.) * w
        y1 = (box[1] - box[3] / 2.) * h
        x2 = (box[0] + box[2] / 2.) * w
        y2 = (box[1] + box[3] / 2.) * h
        draw.rectangle([x1, y1, x2, y2], outline=rgb, width=3)
        if len(box) >= 7 and class_names:
            cls_id = int(box[6])
            draw.text((x1, y1), class_names[cls_id], fill=rgb)
    return img

# ---------- 重新構建 Int8 網路並加載權重 ----------



# ------------- Demo -------------
if __name__ == "__main__":
    names = [line.strip() for line in open("class_names")]
    _ = predict_quant("test.jpg",
                      "yolov2_int8_fbgemm.pth",
                      class_names=names,
                      save_path="demo_out.jpg")
