"""
Explainability Module
Grad-CAM visualization cho ArcFace model
"""

import os
import sys
import numpy as np
from typing import Optional, Union, Tuple
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)


class GradCAM:
    """
    Grad-CAM implementation cho CNN models
    Visualize vung anh quan trong cho prediction
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module = None):
        """
        Args:
            model: PyTorch model
            target_layer: Layer de tinh gradients (mac dinh: layer cuoi cua backbone)
        """
        self.model = model
        self.model.eval()
        
        if target_layer is None:
            if hasattr(model, 'backbone') and hasattr(model.backbone, 'layer4'):
                self.target_layer = model.backbone.layer4
            else:
                self.target_layer = self._find_last_conv_layer(model)
        else:
            self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        
        self._register_hooks()
    
    def _find_last_conv_layer(self, model: nn.Module) -> nn.Module:
        """Tim layer Conv cuoi cung trong model"""
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        return last_conv
    
    def _register_hooks(self):
        """Dang ky forward/backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        handle_fwd = self.target_layer.register_forward_hook(forward_hook)
        handle_bwd = self.target_layer.register_full_backward_hook(backward_hook)
        self.hook_handles.extend([handle_fwd, handle_bwd])
    
    def remove_hooks(self):
        """Remove hooks de tranh memory leak"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_embedding: torch.Tensor = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Input image tensor (1, 3, H, W)
            target_embedding: Target embedding de tinh gradient 
                              (None = su dung embedding cua chinh input)
        
        Returns:
            Heatmap array (H, W) voi gia tri 0-1
        """
        self.model.eval()
        self.model.zero_grad()
        input_tensor.requires_grad = True
        
        output = self.model(input_tensor)
        
        if isinstance(output, tuple):
            embeddings = output[1] if len(output) > 1 else output[0]
        else:
            embeddings = output
        
        if target_embedding is not None:
            score = F.cosine_similarity(embeddings, target_embedding).sum()
        else:
            score = (embeddings ** 2).sum()
        
        score.backward(retain_graph=True)
        
        if self.gradients is None:
            print("Warning: No gradients captured")
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))
        
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        
        cam = cam.squeeze().cpu().numpy()
        
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam


def generate_heatmap(
    cam: np.ndarray,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Convert Grad-CAM array thanh colored heatmap
    
    Args:
        cam: Grad-CAM array (H, W), values 0-1
        colormap: OpenCV colormap
    
    Returns:
        Colored heatmap (H, W, 3) BGR
    """
    cam_uint8 = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam_uint8, colormap)
    return heatmap


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Overlay heatmap len anh goc
    
    Args:
        image: Original image (H, W, 3) BGR hoac RGB
        heatmap: Colored heatmap (H, W, 3) BGR
        alpha: Trong so cua heatmap (0-1)
    
    Returns:
        Overlayed image (H, W, 3)
    """
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlay


def explain_prediction(
    model: nn.Module,
    image_path: str,
    transform,
    device: str = 'cpu',
    target_embedding: torch.Tensor = None,
    output_path: str = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate Grad-CAM explanation cho mot prediction
    
    Args:
        model: ArcFace model
        image_path: Duong dan toi anh
        transform: Image transform
        device: Device
        target_embedding: Target embedding (optional)
        output_path: Duong dan luu anh ket qua (optional)
    
    Returns:
        (original_image, heatmap, overlay)
    """
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    pil_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(pil_image).unsqueeze(0).to(device)
    
    gradcam = GradCAM(model)
    
    if target_embedding is not None:
        target_embedding = target_embedding.to(device)
    
    cam = gradcam.generate(input_tensor, target_embedding)
    
    heatmap = generate_heatmap(cam)
    
    heatmap_resized = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    
    overlay = overlay_heatmap(original, heatmap_resized, alpha=0.5)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        h, w = original.shape[:2]
        combined = np.zeros((h, w * 3, 3), dtype=np.uint8)
        combined[:, :w] = original
        combined[:, w:w*2] = heatmap_resized
        combined[:, w*2:] = overlay
        
        cv2.imwrite(output_path, combined)
        print(f"Saved explanation: {output_path}")
    
    return original_rgb, cv2.cvtColor(heatmap_resized, cv2.COLOR_BGR2RGB), cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


class ExplainabilityEngine:
    """
    Engine de generate explanations cho recognition results
    """
    
    def __init__(self, model: nn.Module, transform, device: str = 'cpu'):
        self.model = model
        self.transform = transform
        self.device = device
        self.gradcam = GradCAM(model)
    
    def explain(
        self,
        image_input: Union[str, Image.Image],
        target_embedding: torch.Tensor = None
    ) -> dict:
        """
        Generate explanation cho mot anh
        
        Returns:
            Dict chua cam, heatmap, overlay
        """
        if isinstance(image_input, str):
            original = cv2.imread(image_input)
            pil_image = Image.open(image_input).convert('RGB')
        else:
            pil_image = image_input.convert('RGB')
            original = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        if original is None:
            return {'error': 'Cannot read image'}
        
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        if target_embedding is not None:
            target_embedding = target_embedding.to(self.device)
        
        cam = self.gradcam.generate(input_tensor, target_embedding)
        
        heatmap = generate_heatmap(cam)
        heatmap_resized = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
        
        overlay = overlay_heatmap(original, heatmap_resized, alpha=0.5)
        
        return {
            'cam': cam,
            'heatmap': cv2.cvtColor(heatmap_resized, cv2.COLOR_BGR2RGB),
            'overlay': cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
            'original': cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        }
    
    def save_explanation(
        self,
        result: dict,
        output_path: str,
        mode: str = 'overlay'
    ):
        """
        Luu explanation ra file
        
        Args:
            result: Output tu explain()
            output_path: Duong dan luu
            mode: 'overlay', 'heatmap', 'combined'
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        if mode == 'overlay':
            img = cv2.cvtColor(result['overlay'], cv2.COLOR_RGB2BGR)
        elif mode == 'heatmap':
            img = cv2.cvtColor(result['heatmap'], cv2.COLOR_RGB2BGR)
        elif mode == 'combined':
            h, w = result['original'].shape[:2]
            combined = np.zeros((h, w * 3, 3), dtype=np.uint8)
            combined[:, :w] = cv2.cvtColor(result['original'], cv2.COLOR_RGB2BGR)
            combined[:, w:w*2] = cv2.cvtColor(result['heatmap'], cv2.COLOR_RGB2BGR)
            combined[:, w*2:] = cv2.cvtColor(result['overlay'], cv2.COLOR_RGB2BGR)
            img = combined
        else:
            img = cv2.cvtColor(result['overlay'], cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(output_path, img)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    print("="*60)
    print("EXPLAINABILITY MODULE TEST")
    print("="*60)
    
    print("\nModule loaded successfully!")
    print("Available classes: GradCAM, ExplainabilityEngine")
    print("Available functions: generate_heatmap, overlay_heatmap, explain_prediction")
