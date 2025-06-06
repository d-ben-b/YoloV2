import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from model import YoloV2Net, load_weights
from utils import filtered_boxes

def compare_model_confidence(
    original_weights_path,
    quantized_weights_path,
    image_path,
    num_classes=80,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Compare confidence scores between original and quantized models.
    """
    # Load original model
    orig_model = YoloV2Net(num_classes=num_classes)
    load_weights(orig_model, original_weights_path)
    orig_model.to(device)
    orig_model.eval()
    
    # Load quantized model
    quant_model = YoloV2Net(num_classes=num_classes)
    quant_model.load_state_dict(torch.load(quantized_weights_path, map_location=device))
    quant_model.to(device)
    quant_model.eval()
    
    # Load and prepare image
    img = Image.open(image_path).convert('RGB')
    tensor_img = transforms.ToTensor()(img.resize((608, 608))).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        # Forward pass through both models
        orig_output = orig_model(tensor_img).data
        quant_output = quant_model(tensor_img).data
        
        # Extract and prepare objectness confidence scores properly
        # YOLOv2 format: For each anchor box, outputs [x, y, w, h, obj, class1, class2, ...]
        num_anchors = orig_model.num_anchors
        
        # Calculate objectness scores (index 4 of each anchor's output)
        batch, channels, height, width = orig_output.shape
        # Reshape to isolate the objectness scores
        obj_idx = 4  # Index for objectness score in the output format
        
        orig_obj_scores = []
        quant_obj_scores = []
        
        for a in range(num_anchors):
            # Extract objectness score for each anchor
            start_idx = a * (5 + num_classes) + obj_idx
            orig_obj_scores.append(orig_output[0, start_idx, :, :].sigmoid().cpu().numpy().flatten())
            quant_obj_scores.append(quant_output[0, start_idx, :, :].sigmoid().cpu().numpy().flatten())
        
        # Combine scores from all anchors
        orig_obj_conf = np.concatenate(orig_obj_scores)
        quant_obj_conf = np.concatenate(quant_obj_scores)
        
        print(f"Original model max objectness: {orig_obj_conf.max():.6f}")
        print(f"Quantized model max objectness: {quant_obj_conf.max():.6f}")
        
        # Calculate ratio of confidence reduction
        if orig_obj_conf.max() > 0:
            conf_ratio = quant_obj_conf.max() / orig_obj_conf.max()
            print(f"Confidence ratio (quantized/original): {conf_ratio:.6f}")
        
        # Compare class confidence distributions (extract max class confidence for each position)
        orig_class_scores = []
        quant_class_scores = []
        
        for a in range(num_anchors):
            for c in range(num_classes):
                start_idx = a * (5 + num_classes) + 5 + c  # 5 for bbox + obj, then add class index
                orig_class_scores.append(orig_output[0, start_idx, :, :].cpu().numpy().flatten())
                quant_class_scores.append(quant_output[0, start_idx, :, :].cpu().numpy().flatten())
        
        # Convert to numpy arrays and apply softmax
        orig_class_scores = np.array(orig_class_scores).T  # Transpose to get classes as columns
        quant_class_scores = np.array(quant_class_scores).T
        
        # Calculate max class confidence per position
        orig_max_class = np.max(orig_class_scores, axis=1)
        quant_max_class = np.max(quant_class_scores, axis=1)
        
        print(f"Original model max class conf: {orig_max_class.max():.6f}")
        print(f"Quantized model max class conf: {quant_max_class.max():.6f}")
        
        # Plot confidence histograms
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(orig_obj_conf, bins=20, alpha=0.5, label='Original')
        plt.hist(quant_obj_conf, bins=20, alpha=0.5, label='Quantized')
        plt.title('Objectness Confidence Distribution')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.hist(orig_max_class, bins=20, alpha=0.5, label='Original')
        plt.hist(quant_max_class, bins=20, alpha=0.5, label='Quantized')
        plt.title('Max Class Confidence Distribution')
        plt.legend()
        plt.tight_layout()
        plt.savefig('confidence_comparison.png')
        print("Saved confidence histogram to confidence_comparison.png")

def compare_layer_outputs(
    original_weights_path,
    quantized_weights_path,
    image_path,
    num_classes=80,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Compare layer outputs between original and quantized models.
    """
    # Load models
    orig_model = YoloV2Net(num_classes=num_classes)
    load_weights(orig_model, original_weights_path)
    orig_model.to(device)
    orig_model.eval()
    
    quant_model = YoloV2Net(num_classes=num_classes)
    quant_model.load_state_dict(torch.load(quantized_weights_path, map_location=device))
    quant_model.to(device)
    quant_model.eval()
    
    # Prepare hook for layer outputs
    layer_outputs_orig = {}
    layer_outputs_quant = {}
    
    def get_hook(name, outputs_dict):
        def hook(module, input, output):
            outputs_dict[name] = output.detach().cpu()
        return hook
    
    # Register hooks for key layers
    hooks_orig = []
    hooks_quant = []
    
    # You might need to adjust these layer names based on your model architecture
    # Find some late-stage convolutional layers that are important for detection
    key_layers = []
    for name, module in orig_model.named_modules():
        if isinstance(module, nn.Conv2d):
            if 'conv' in name and name.endswith(('81', '82', '83')):  # Adjust based on your model
                key_layers.append(name)
    
    # If no specific layers found, use any available convolutional layer
    if not key_layers:
        for name, module in orig_model.named_modules():
            if isinstance(module, nn.Conv2d):
                key_layers.append(name)
                break
    
    print(f"Analyzing layers: {key_layers}")
    
    # Register hooks
    for name, module in orig_model.named_modules():
        if name in key_layers:
            hooks_orig.append(module.register_forward_hook(
                get_hook(name, layer_outputs_orig)
            ))
    
    for name, module in quant_model.named_modules():
        if name in key_layers:
            hooks_quant.append(module.register_forward_hook(
                get_hook(name, layer_outputs_quant)
            ))
    
    # Load and prepare image
    img = Image.open(image_path).convert('RGB')
    tensor_img = transforms.ToTensor()(img.resize((608, 608))).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        orig_model(tensor_img)
        quant_model(tensor_img)
    
    # Compare layer outputs
    for layer_name in layer_outputs_orig:
        if layer_name in layer_outputs_quant:
            orig_out = layer_outputs_orig[layer_name]
            quant_out = layer_outputs_quant[layer_name]
            
            # Convert to numpy arrays for plotting
            orig_out_np = orig_out.numpy().flatten()
            quant_out_np = quant_out.numpy().flatten()
            
            # Calculate error metrics
            abs_diff = np.abs(orig_out_np - quant_out_np)
            mean_error = np.mean(abs_diff)
            max_error = np.max(abs_diff)
            
            print(f"Layer {layer_name}:")
            print(f"  Mean absolute error: {mean_error:.6f}")
            print(f"  Max absolute error: {max_error:.6f}")
            
            # Plot distribution of differences
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.hist(orig_out_np, bins=50, alpha=0.5, label='Original')
            plt.hist(quant_out_np, bins=50, alpha=0.5, label='Quantized')
            plt.title(f'Layer {layer_name} Output Distribution')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.hist(abs_diff, bins=50)
            plt.title(f'Layer {layer_name} Absolute Error')
            plt.tight_layout()
            plt.savefig(f'layer_{layer_name}_comparison.png')
            print(f"Saved layer comparison to layer_{layer_name}_comparison.png")
    
    # Remove hooks
    for hook in hooks_orig + hooks_quant:
        hook.remove()

def compare_detections(
    original_weights_path,
    quantized_weights_path,
    image_path,
    class_names,
    conf_thresh=0.001,  # CHANGED: Using a much lower threshold to catch more boxes
    nms_thresh=0.4,
    num_classes=80,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Compare object detections between original and quantized models.
    """
    # Load models - fix the loading to be consistent
    orig_model = YoloV2Net(num_classes=num_classes)
    load_weights(orig_model, original_weights_path)  # Use consistent loading method
    orig_model.to(device)
    orig_model.eval()
    
    quant_model = YoloV2Net(num_classes=num_classes)
    quant_model.load_state_dict(torch.load(quantized_weights_path, map_location=device))
    quant_model.to(device)
    quant_model.eval()
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Get detections with very low threshold
    print(f"Using confidence threshold: {conf_thresh}")
    orig_boxes = filtered_boxes(orig_model, device, img.resize((608, 608)), conf_thresh, nms_thresh)
    quant_boxes = filtered_boxes(quant_model, device, img.resize((608, 608)), conf_thresh, nms_thresh)
    
    print(f"Original model detected {len(orig_boxes)} boxes")
    print(f"Quantized model detected {len(quant_boxes)} boxes")
    
    # Show top detections from each model
    print("\nTop detections from original model:")
    sorted_orig = sorted(orig_boxes, key=lambda x: x[4]*x[5], reverse=True)[:5]
    for i, box in enumerate(sorted_orig):
        cls_id = int(box[6])
        conf = box[4] * box[5]
        if isinstance(class_names, list):
            class_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
            print(f"  {i+1}. {class_name}: {conf:.6f}")
        else:
            print(f"  {i+1}. Class {cls_id}: {conf:.6f}")
    
    print("\nTop detections from quantized model:")
    sorted_quant = sorted(quant_boxes, key=lambda x: x[4]*x[5], reverse=True)[:5]
    for i, box in enumerate(sorted_quant):
        cls_id = int(box[6])
        conf = box[4] * box[5]
        if isinstance(class_names, list):
            class_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
            print(f"  {i+1}. {class_name}: {conf:.6f}")
        else:
            print(f"  {i+1}. Class {cls_id}: {conf:.6f}")
    
    # Create side-by-side visualization
    orig_img = img.copy()
    quant_img = img.copy()
    
    # Add text to indicate which image is which
    from PIL import ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    draw_orig = ImageDraw.Draw(orig_img)
    draw_orig.text((20, 20), "Original Model", fill=(255, 0, 0), font=font)
    
    draw_quant = ImageDraw.Draw(quant_img)
    draw_quant.text((20, 20), "Quantized Model", fill=(255, 0, 0), font=font)
    
    # Ensure class_names is properly formatted for plot_boxes
    if not isinstance(class_names, list):
        # Create a dummy list of class names
        temp_class_names = [f"Class {i}" for i in range(num_classes)]
        orig_img = plot_boxes(orig_img, orig_boxes, None, temp_class_names)
        quant_img = plot_boxes(quant_img, quant_boxes, None, temp_class_names)
    else:
        orig_img = plot_boxes(orig_img, orig_boxes, None, class_names)
        quant_img = plot_boxes(quant_img, quant_boxes, None, class_names)
    
    # Create a side-by-side comparison image
    w, h = img.size
    comparison = Image.new('RGB', (w*2, h))
    comparison.paste(orig_img, (0, 0))
    comparison.paste(quant_img, (w, 0))
    
    # Save the comparison image
    comparison.save('detection_comparison.jpg')
    print("Saved detection comparison to detection_comparison.jpg")
    
    return comparison

if __name__ == "__main__":
    # Example usage:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Update these paths for your system
    original_weights = "weights/yolov2.weights"
    quantized_weights = "yolov2_pot2_weightonly.pth"
    test_image = "./images/Trucks-2.jpg"  # Replace with your test image
    
    # COCO class names
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 
        'toothbrush'
    ]
    
    # Run detection comparison first with very low threshold to ensure we get boxes
    print("\n=== Running detection comparison with low threshold ===")
    compare_detections(original_weights, quantized_weights, test_image, COCO_CLASSES)
    
    # Then run the other comparisons
    print("\n=== Running confidence score comparison ===")
    compare_model_confidence(original_weights, quantized_weights, test_image)
    print("\n=== Running layer output comparison ===")
    compare_layer_outputs(original_weights, quantized_weights, test_image)
