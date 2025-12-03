import time
import os
import sys

import torch
from dataloader.transformers import Rescale
from model.lanenet.LaneNet import LaneNet
from model.lanenet.LaneNetPlus import LaneNetPlus
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from model.utils.cli_helper_test import parse_args
import numpy as np
from PIL import Image
import pandas as pd
import cv2

# Import homography rectification
try:
    from preprocess.homography_rectification import HomographyRectifier
    RECTIFICATION_AVAILABLE = True
except ImportError:
    RECTIFICATION_AVAILABLE = False
    print("Warning: Homography rectification not available.")

# Import Kalman filter fusion module
try:
    from model.utils.kalman_fusion import KalmanLaneFusion
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False
    print("Warning: Kalman filter module not available. Install required dependencies.")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_test_data(img_path, transform, use_rectification=False, rectifier=None):
    """
    Load and optionally rectify test image
    
    Args:
        img_path: Path to image
        transform: Image transforms
        use_rectification: Whether to apply homography rectification
        rectifier: HomographyRectifier instance
    
    Returns:
        Transformed image tensor
    """
    img = Image.open(img_path)
    
    # Apply rectification if requested
    if use_rectification and RECTIFICATION_AVAILABLE and rectifier is not None:
        # Convert PIL to numpy for rectification
        img_np = np.array(img)
        if len(img_np.shape) == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Rectify
        rectified_img, _ = rectifier.rectify_image(img_np)
        
        # Convert back to PIL
        if len(rectified_img.shape) == 3:
            rectified_img = cv2.cvtColor(rectified_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rectified_img)
    
    img = transform(img)
    return img

def test():
    if os.path.exists('test_output') == False:
        os.mkdir('test_output')
    args = parse_args()
    img_path = args.img
    resize_height = args.height
    resize_width = args.width

    data_transform = transforms.Compose([
        transforms.Resize((resize_height,  resize_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model_path = args.model
    
    # Create model (LaneNet or LaneNetPlus)
    if args.use_lanenet_plus:
        model = LaneNetPlus(
            arch=args.model_type,
            use_attention=args.use_attention,
            use_multitask=args.use_multitask,
            freeze_encoder=args.freeze_backbone
        )
    else:
        model = LaneNet(arch=args.model_type, freeze_encoder=args.freeze_backbone)
    
    # Ensure the checkpoint can be loaded on CPU-only machines
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    
    # Initialize homography rectifier if requested
    rectifier = None
    if args.use_rectification and RECTIFICATION_AVAILABLE:
        rectifier = HomographyRectifier()
        print("Homography rectification enabled")
    
    # Initialize Kalman filter if requested
    kalman_fusion = None
    if args.use_kalman and KALMAN_AVAILABLE:
        print("Initializing Kalman filter for lane tracking...")
        kalman_fusion = KalmanLaneFusion(
            max_lanes=args.max_lanes,
            process_noise=args.kalman_process_noise,
            measurement_noise=args.kalman_measurement_noise
        )
    elif args.use_kalman and not KALMAN_AVAILABLE:
        print("Warning: Kalman filter requested but module not available. Continuing without Kalman.")

    dummy_input = load_test_data(img_path, data_transform, 
                                 use_rectification=args.use_rectification,
                                 rectifier=rectifier).to(DEVICE)
    dummy_input = torch.unsqueeze(dummy_input, dim=0)
    outputs = model(dummy_input)

    input_img = Image.open(img_path)
    
    # Apply rectification to input image for visualization if enabled
    if args.use_rectification and RECTIFICATION_AVAILABLE and rectifier is not None:
        input_img_np = np.array(input_img)
        if len(input_img_np.shape) == 3:
            input_img_np = cv2.cvtColor(input_img_np, cv2.COLOR_RGB2BGR)
        input_img_np, _ = rectifier.rectify_image(input_img_np)
        if len(input_img_np.shape) == 3:
            input_img_np = cv2.cvtColor(input_img_np, cv2.COLOR_BGR2RGB)
        input_img = Image.fromarray(input_img_np)
    
    input_img = input_img.resize((resize_width, resize_height))
    input_array = np.array(input_img)

    # Get predictions (handle both LaneNet and LaneNetPlus outputs)
    if 'instance_seg_logits' in outputs:
        instance_pred = torch.squeeze(outputs['instance_seg_logits'].detach().to('cpu')).numpy() * 255
    else:
        instance_pred = None
    
    # Get binary/lane prediction
    if 'lane_pred' in outputs:
        binary_pred = torch.squeeze(outputs['lane_pred']).to('cpu').numpy()
    elif 'binary_seg_pred' in outputs:
        binary_pred = torch.squeeze(outputs['binary_seg_pred']).to('cpu').numpy()
    else:
        raise ValueError("No lane/binary prediction found in model output")

    # Convertir binary_pred a máscara binaria 2D
    binary_mask = binary_pred.squeeze()
    if len(binary_mask.shape) > 2:
        binary_mask = binary_mask.squeeze()
    
    # Asegurar que la máscara sea binaria (valores 0 o 1) y tenga las dimensiones correctas
    binary_mask = (binary_mask > 0).astype(np.uint8)
    
    # Asegurar que la máscara tenga las mismas dimensiones que la imagen
    if binary_mask.shape != input_array.shape[:2]:
        binary_mask = cv2.resize(binary_mask, (input_array.shape[1], input_array.shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)
    
    # Prepare instance mask for Kalman filter
    instance_mask_resized = None
    if len(instance_pred.shape) == 3:
        instance_mask_resized = instance_pred.transpose((1, 2, 0))
        if instance_mask_resized.shape[:2] != input_array.shape[:2]:
            # Resize instance mask
            h, w = input_array.shape[:2]
            instance_mask_resized = cv2.resize(instance_mask_resized, (w, h), 
                                               interpolation=cv2.INTER_LINEAR)
        # Normalize to [0, 1]
        instance_mask_resized = instance_mask_resized / 255.0
    
    # Apply Kalman filter fusion if enabled
    if kalman_fusion is not None:
        print("Applying Kalman filter fusion...")
        try:
            # Convert to 0-255 format for Kalman filter
            binary_mask_255 = (binary_mask * 255).astype(np.uint8)
            fusion_result = kalman_fusion.fuse_detections(
                binary_mask_255,
                instance_mask=instance_mask_resized,
                enet_confidence=0.7
            )
            
            # Use fused binary mask (already in 0-255 format)
            binary_mask_fused = fusion_result['binary_mask']
            if binary_mask_fused is not None and binary_mask_fused.size > 0:
                binary_mask = (binary_mask_fused > 127).astype(np.uint8)  # Convert to 0-1 for compatibility
            else:
                print("Warning: Kalman fusion returned empty mask, using ENet detection")
            
            # Optionally use fused instance mask
            if fusion_result.get('instance_mask') is not None:
                instance_mask_resized = fusion_result['instance_mask']
            
            lanes_count = len(fusion_result.get('lanes', {}))
            print(f"Fusion weights - Kalman: {fusion_result['fusion_weights']['kalman']:.3f}, "
                  f"ENet: {fusion_result['fusion_weights']['enet']:.3f}")
            print(f"Tracking {lanes_count} lanes")
        except Exception as e:
            print(f"Error in Kalman fusion: {e}")
            print("Falling back to ENet detections only")
            import traceback
            traceback.print_exc()
    
    # Convertir imagen RGB a BGR para OpenCV
    if len(input_array.shape) == 3 and input_array.shape[2] == 3:
        overlay_image = cv2.cvtColor(input_array, cv2.COLOR_RGB2BGR)
    else:
        overlay_image = input_array.copy()
    
    # Crear una máscara de color para las líneas (amarillo brillante en BGR)
    color_overlay = np.zeros_like(overlay_image)
    binary_mask_vis = (binary_mask * 255).astype(np.uint8) if binary_mask.max() <= 1 else binary_mask.astype(np.uint8)
    color_overlay[binary_mask_vis > 0] = [0, 255, 255]  # Amarillo en formato BGR
    
    # Superponer las líneas sobre la imagen original con transparencia
    # Usamos addWeighted: 65% imagen original, 35% líneas amarillas para mejor visibilidad
    overlay_result = cv2.addWeighted(overlay_image, 0.65, color_overlay, 0.35, 0)
    
    # Guardar las imágenes
    # input.jpg ahora muestra la imagen original con las líneas superpuestas
    cv2.imwrite(os.path.join('test_output', 'input.jpg'), overlay_result)
    
    # Asegurar dimensiones correctas para instance_output
    if instance_pred is not None:
        instance_output = instance_pred.transpose((1, 2, 0)) if len(instance_pred.shape) == 3 else instance_pred
        if instance_output.max() > 255:
            instance_output = np.clip(instance_output, 0, 255)
        cv2.imwrite(os.path.join('test_output', 'instance_output.jpg'), instance_output.astype(np.uint8))
    
    cv2.imwrite(os.path.join('test_output', 'binary_output.jpg'), (binary_mask * 255).astype(np.uint8))
    
    # Save drivable area prediction if available
    if args.use_multitask and 'drivable_pred' in outputs:
        drivable_pred = torch.squeeze(outputs['drivable_pred']).to('cpu').numpy()
        drivable_mask = drivable_pred.squeeze()
        if len(drivable_mask.shape) > 2:
            drivable_mask = drivable_mask.squeeze()
        drivable_mask = (drivable_mask > 0).astype(np.uint8)
        
        # Resize if needed
        if drivable_mask.shape != input_array.shape[:2]:
            drivable_mask = cv2.resize(drivable_mask, (input_array.shape[1], input_array.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
        
        # Create overlay for drivable area
        drivable_overlay = np.zeros_like(overlay_image)
        drivable_mask_vis = (drivable_mask * 255).astype(np.uint8) if drivable_mask.max() <= 1 else drivable_mask.astype(np.uint8)
        drivable_overlay[drivable_mask_vis > 0] = [255, 0, 0]  # Red in BGR
        drivable_result = cv2.addWeighted(overlay_image, 0.65, drivable_overlay, 0.35, 0)
        
        cv2.imwrite(os.path.join('test_output', 'drivable_output.jpg'), (drivable_mask * 255).astype(np.uint8))
        cv2.imwrite(os.path.join('test_output', 'drivable_overlay.jpg'), drivable_result)


if __name__ == "__main__":
    test()