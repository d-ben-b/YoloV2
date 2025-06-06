import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from PIL import Image
import os
import time

def gpu2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)


def find_all_boxes(
    output,
    device,
    conf_thresh,
    num_classes,
    anchors,
    num_anchors,
    only_objectness=1,
    validation=False):
    """ extracting bboxes and confidece from output """
    num_classes, num_anchors = int(num_classes), int(num_anchors)
    anchor_step = int(len(anchors) / num_anchors)
    if output.dim == 3:
       output = output.unsqueeze(0)
    batch = output.size(0)
    assert (output.size(1) == (5 + num_classes) * num_anchors)
    h = output.size(2)
    w = output.size(3)
    
    all_boxes = []
    output = output.view(batch * num_anchors, 5 + num_classes, h * w).transpose(0, 1).contiguous().view(5 + num_classes, batch * num_anchors * h * w)
   
    grid_x = torch.linspace(0, h-1, h).repeat(h, 1).repeat(batch * num_anchors, 1, 1).view(batch * num_anchors * h * w).to(device)
    grid_y = torch.linspace(0, w-1, w).repeat(w, 1).t().repeat(batch * num_anchors, 1, 1).view(batch * num_anchors * h * w).to(device)
    xs = torch.sigmoid(output[0]) + grid_x
    ys = torch.sigmoid(output[1]) + grid_y
   
    anchor_w = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([0]))
    anchor_h = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([1]))
    anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, h * w).view(batch * num_anchors * h * w).to(device)
    anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, h * w).view(batch * num_anchors * h * w).to(device)
    ws = torch.exp(output[2]) * anchor_w
    hs = torch.exp(output[3]) * anchor_h
 
    det_confs = torch.sigmoid(output[4])
 
    cls_confs = nn.Softmax(dim = 0)(output[5: 5 + num_classes].transpose(0, 1)).data
    cls_max_confs, cls_max_ids = torch.max (cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids   = cls_max_ids.view(-1)
 
    sz_hw  = h * w
    sz_hwa = sz_hw * num_anchors

    det_confs     = det_confs.cpu()
    cls_max_confs = cls_max_confs.cpu()
    cls_max_ids   = gpu2cpu_long(cls_max_ids)
    xs, ys = xs.cpu(), ys.cpu()
    ws, hs = ws.cpu(), hs.cpu()
 
    if validation:
        cls_confs = cls_confs.view(-1, num_classes).cpu()
 
    for b in range(batch):
        boxes = []
        for cy in range(h):
            for cx in range(w):
                for i in range(num_anchors):
                    ind = b * sz_hwa + i * sz_hw + cy * w + cx
                    det_conf = det_confs[ind]
                    if only_objectness:
                        conf = det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]
 
                    if conf > conf_thresh:
                        bcx = xs[ind]
                        bcy = ys[ind]
                        bw  = ws[ind]
                        bh  = hs[ind]
 
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id   = cls_max_ids[ind]
                        box = [bcx/w, bcy/h, bw/w, bh/h, det_conf, cls_max_conf, cls_max_id]
                        if (not only_objectness) and validation:
                            for c in range(num_classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[ind] * tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        boxes.append(box)
        all_boxes.append(boxes)

    return all_boxes


def iou(box1, box2, x1y1x2y2=True):
    """ Intersection Over Union """
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
 
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else: # (x, y, w, h)
        mx = min(box1[0] - box1[2] / 2, box2[0] - box2[2] / 2)
        Mx = max(box1[0] + box1[2] / 2, box2[0] + box2[2] / 2)
        my = min(box1[1] - box1[3] / 2, box2[1] - box2[3] / 2)
        My = max(box1[1] + box1[3] / 2, box2[1] + box2[3] / 2)
 
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
 
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
   
    corea = 0
    if cw <= 0 or ch <= 0:
        return 0.0
    area1 = w1 * h1
    area2 = w2 * h2
    corea = cw * ch
    uarea = area1 + area2 - corea
    return corea / uarea


def nms(boxes, nms_thresh):
    """ None Max Separetion """
    if len(boxes) == 0:
        return boxes
    det_confs = torch.zeros(len(boxes))
    for i in range(len(boxes)):
        det_confs[i] = 1 - boxes[i][4]
 
    _, sortIds = torch.sort(det_confs)
    out_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]
        if box_i[4] > 0:
           out_boxes.append(box_i)
           for j in range(i + 1, len(boxes)):
               box_j = boxes[sortIds[j]]
               if iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                  box_j[4] = 0
    return out_boxes


def filtered_boxes(model, device, img, conf_thresh, nms_thresh):
    """ filter best boxes from all boxes """
    model.eval()
   
    if isinstance(img, Image.Image):
        img = transforms.ToTensor ()(img).unsqueeze(0)
    elif type (img) == np.ndarray:
        img = torch.from_numpy(img.transpose (2, 0, 1)).float().div(255.0).unsqueeze(0)
    else:
        print('unknown image type')
        exit(-1)
 
    img = img.to(device)
 
    output = model(img)
    output = output.data
 
    boxes = find_all_boxes(
        output,
        device,
        conf_thresh,
        model.num_classes,
        model.anchors,
        model.num_anchors)[0]
 
    boxes = nms(boxes, nms_thresh)
 
    return boxes

def predict_ptq_low_conf(model_path, img_path, class_names, device, save_to=None, num_classes=80, 
                        conf_thresh=0.01, nms_thresh=0.4):
    """
    Perform prediction using a quantized YOLOv2 model with lower confidence threshold.
    Specially designed for quantized models which typically produce lower confidence scores.
    
    Args:
        model_path: Path to the quantized model weights (.pth)
        img_path: Path to the input image
        class_names: List of class names
        device: Device to run inference on ('cpu' or 'cuda')
        save_to: Path to save the output image (optional)
        num_classes: Number of classes (default: 80 for COCO)
        conf_thresh: Confidence threshold for detections (default: 0.01, much lower than normal)
        nms_thresh: Non-maximum suppression threshold (default: 0.4)
    
    Returns:
        PIL.Image: Annotated image with bounding boxes
    """
    assert os.path.exists(img_path), 'Error! Input image does not exist.'
    assert os.path.exists(model_path), 'Error! Model file does not exist.'
    
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
    
    # Debug detection info
    print(f"Number of detected boxes: {len(boxes)}")
    if len(boxes) > 0:
        # Sort boxes by confidence and print top 5
        sorted_boxes = sorted(boxes, key=lambda x: x[4]*x[5], reverse=True)
        for i, box in enumerate(sorted_boxes[:5]):
            cls_id = int(box[6])
            cls_name = class_names[cls_id] if isinstance(class_names, list) and cls_id < len(class_names) else f"Class {cls_id}"
            conf = box[4] * box[5]
            print(f"Box {i}: {cls_name} - Confidence: {conf:.4f}")
    
    # Draw bounding boxes on image
    pred_img = plot_boxes(img, boxes, save_to, class_names)
    
    return pred_img


def predict_ptq_rescaled(model_path, img_path, class_names, device, save_to=None, num_classes=80, 
                        conf_thresh=0.0001, nms_thresh=0.4, confidence_scale=200.0):
    """
    Prediction with confidence rescaling to compensate for quantization effects.
    
    Args:
        model_path: Path to the quantized model weights (.pth)
        img_path: Path to the input image
        class_names: List of class names
        device: Device to run inference on ('cpu' or 'cuda')
        save_to: Path to save the output image (optional)
        num_classes: Number of classes (default: 80 for COCO)
        conf_thresh: Confidence threshold for detections 
        nms_thresh: Non-maximum suppression threshold (default: 0.4)
        confidence_scale: Factor to multiply confidence scores by (default: 200.0)
    
    Returns:
        PIL.Image: Annotated image with bounding boxes
    """
    assert os.path.exists(img_path), 'Error! Input image does not exist.'
    assert os.path.exists(model_path), 'Error! Model file does not exist.'
    
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
    
    # Debug detection info
    print(f"Number of detected boxes: {len(boxes)}")
    
    # Apply confidence scaling - this doesn't change which boxes are detected,
    # but makes the visualization more informative
    scaled_boxes = []
    for box in boxes:
        # Create a copy of the box
        scaled_box = box.copy()
        # Scale the objectness score for visualization (box[4] is obj score)
        scaled_box[4] = min(scaled_box[4] * confidence_scale, 1.0)
        scaled_boxes.append(scaled_box)
    
    # Show the top detections with rescaled confidence
    if len(scaled_boxes) > 0:
        # Sort boxes by confidence and print top 5
        sorted_boxes = sorted(scaled_boxes, key=lambda x: x[4]*x[5], reverse=True)
        print("\nTop detections with rescaled confidence:")
        for i, box in enumerate(sorted_boxes[:5]):
            cls_id = int(box[6])
            cls_name = class_names[cls_id] if isinstance(class_names, list) and cls_id < len(class_names) else f"Class {cls_id}"
            conf = box[4] * box[5]  # This is now the rescaled confidence
            print(f"Box {i}: {cls_name} - Rescaled Confidence: {conf:.4f}")
    
    # Draw bounding boxes on image using the scaled boxes
    pred_img = plot_boxes(img, scaled_boxes, save_to, class_names)
    
    if save_to:
        print(f"Saved prediction with rescaled confidence to {save_to}")
    
    return pred_img


def predict_ptq_rescaled_v2(model_path, img_path, class_names, device, save_to=None, num_classes=80, 
                        conf_thresh=0.0001, nms_thresh=0.4, confidence_scale=200.0):
    """
    Enhanced prediction with confidence rescaling that properly affects visualization.
    
    Args:
        model_path: Path to the quantized model weights (.pth)
        img_path: Path to the input image
        class_names: List of class names
        device: Device to run inference on ('cpu' or 'cuda')
        save_to: Path to save the output image (optional)
        num_classes: Number of classes (default: 80 for COCO)
        conf_thresh: Confidence threshold for detections 
        nms_thresh: Non-maximum suppression threshold (default: 0.4)
        confidence_scale: Factor to multiply confidence scores by (default: 200.0)
    
    Returns:
        PIL.Image: Annotated image with bounding boxes
    """
    assert os.path.exists(img_path), 'Error! Input image does not exist.'
    assert os.path.exists(model_path), 'Error! Model file does not exist.'
    
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
    
    # Debug detection info
    print(f"Number of detected boxes: {len(boxes)}")
    
    # Apply confidence scaling and draw boxes directly instead of using plot_boxes
    draw = ImageDraw.Draw(img)
    
    # Try to use a built-in font or fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Sort boxes by combined confidence
    sorted_boxes = sorted(boxes, key=lambda x: x[4]*x[5], reverse=True)
    
    # Print original top detections
    print("\nTop detections with original confidence:")
    for i, box in enumerate(sorted_boxes[:5]):
        cls_id = int(box[6])
        cls_name = class_names[cls_id] if isinstance(class_names, list) and cls_id < len(class_names) else f"Class {cls_id}"
        conf = box[4] * box[5]
        print(f"Box {i}: {cls_name} - Original Confidence: {conf:.6f}")
    
    # Draw boxes with rescaled confidence values
    colors = torch.FloatTensor([
        [1, 0, 1], [0, 0, 1], [0, 1, 1],
        [0, 1, 0], [1, 1, 0], [1, 0, 0]
    ])
    
    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(np.floor(ratio))
        j = int(np.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)
    
    # Print rescaled detections
    print("\nTop detections with rescaled confidence:")
    
    for i, box in enumerate(sorted_boxes):
        x, y, w, h, obj_score, cls_score, cls_id = box[:7]
        
        # Apply confidence scaling
        rescaled_obj_score = min(obj_score * confidence_scale, 1.0)
        rescaled_conf = rescaled_obj_score * cls_score
        
        # Print top 5 rescaled confidences
        if i < 5:
            cls_name = class_names[int(cls_id)] if isinstance(class_names, list) and int(cls_id) < len(class_names) else f"Class {int(cls_id)}"
            print(f"Box {i}: {cls_name} - Rescaled Confidence: {rescaled_conf:.4f}")
        
        # Calculate box coordinates
        xmin = (x - w/2) * img.width
        ymin = (y - h/2) * img.height
        xmax = (x + w/2) * img.width
        ymax = (y + h/2) * img.height
        
        # Get color based on class
        if isinstance(class_names, list) and int(cls_id) < len(class_names):
            cls_id_int = int(cls_id)
            offset = cls_id_int * 123457 % len(class_names)
            red = get_color(2, offset, len(class_names))
            green = get_color(1, offset, len(class_names))
            blue = get_color(0, offset, len(class_names))
            rgb = (red, green, blue)
            
            # Draw class name with rescaled confidence
            label = f"{class_names[cls_id_int]} {rescaled_conf:.2f}"
            draw.rectangle([xmin, ymin - 20, xmin + 8 * len(label), ymin], fill=rgb)
            draw.text((xmin + 2, ymin - 18), label, fill=(0, 0, 0), font=font)
        else:
            rgb = (255, 0, 0)  # Red default
            
        # Draw bounding box
        draw.rectangle([xmin, ymin, xmax, ymax], outline=rgb, width=3)
    
    if save_to:
        img.save(save_to)
        print(f"Saved prediction with properly rescaled confidence to {save_to}")
    
    return img