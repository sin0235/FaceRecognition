"""
ArcFace Model Implementation
Bao gom ResNet50 backbone va ArcFace loss layer
Ho tro: download pretrained weights, mixed precision training
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
from typing import Optional, Tuple


PRETRAINED_URLS = {
    'resnet50_imagenet': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet50_vggface2': None,
    'resnet50_ms1mv2': None,
}


class ArcMarginProduct(nn.Module):
    """
    ArcFace Loss Layer (Additive Angular Margin)
    Cong thuc: cos(theta + m)
    """
    def __init__(self, in_features: int, out_features: int, 
                 scale: float = 64.0, margin: float = 0.5, easy_margin: bool = False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
    
    def forward(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=1e-7))
        
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        return output


class ResNetBackbone(nn.Module):
    """
    ResNet50 Backbone cho feature extraction
    Output: 2048-dim feature vector
    """
    def __init__(self, pretrained: bool = True, pretrained_path: Optional[str] = None):
        super(ResNetBackbone, self).__init__()
        
        import torchvision.models as models
        
        # Load ResNet50
        if pretrained and pretrained_path is None:
            try:
                resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                print("Loaded ImageNet pretrained weights")
            except:
                resnet = models.resnet50(pretrained=True)
                print("Loaded ImageNet pretrained weights (legacy)")
        else:
            resnet = models.resnet50(pretrained=False)
            if pretrained_path and os.path.exists(pretrained_path):
                self._load_custom_pretrained(resnet, pretrained_path)
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.avgpool = resnet.avgpool
    
    def _load_custom_pretrained(self, resnet, path):
        """Load custom pretrained weights"""
        try:
            state_dict = torch.load(path, map_location='cpu', weights_only=False)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # Filter keys that match
            model_dict = resnet.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() 
                             if k in model_dict and model_dict[k].shape == v.shape}
            
            model_dict.update(pretrained_dict)
            resnet.load_state_dict(model_dict)
            print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers from {path}")
        except Exception as e:
            print(f"Failed to load custom pretrained: {e}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x


class ArcFaceModel(nn.Module):
    """
    ArcFace Model hoan chinh
    
    Architecture:
        ResNet50 Backbone (2048-dim) -> BN -> Dropout -> FC (512-dim) -> BN -> ArcFace Head
    """
    def __init__(self, num_classes: int, embedding_size: int = 512, 
                 pretrained: bool = True, pretrained_path: Optional[str] = None,
                 scale: float = 64.0, margin: float = 0.5, easy_margin: bool = False,
                 dropout: float = 0.5):
        super(ArcFaceModel, self).__init__()
        
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        
        self.backbone = ResNetBackbone(pretrained=pretrained, pretrained_path=pretrained_path)
        
        # Feature dimension tu ResNet50 la 2048
        self.bn1 = nn.BatchNorm1d(2048)
        self.dropout = nn.Dropout(p=dropout)
        
        # Embedding layer
        self.fc = nn.Linear(2048, embedding_size)
        self.bn2 = nn.BatchNorm1d(embedding_size)
        
        # ArcFace layer
        self.arcface = ArcMarginProduct(
            in_features=embedding_size,
            out_features=num_classes,
            scale=scale,
            margin=margin,
            easy_margin=easy_margin
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Khoi tao weights cho FC layer"""
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input images (B, 3, H, W)
            labels: Class labels (B,) - required for training with ArcFace loss
            
        Returns:
            If labels provided: (logits, embeddings)
            If labels is None: embeddings only
        """
        x = self.backbone(x)
        x = self.bn1(x)
        x = self.dropout(x)
        
        embeddings = self.fc(x)
        embeddings = self.bn2(embeddings)
        
        if labels is not None:
            output = self.arcface(embeddings, labels)
            return output, embeddings
        else:
            return embeddings
    
    def extract_features(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Trich xuat embeddings (dung cho inference)
        
        Args:
            x: Input images
            normalize: L2 normalize embeddings
        """
        self.eval()
        with torch.no_grad():
            embeddings = self.forward(x, labels=None)
            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def get_embedding_dim(self) -> int:
        return self.embedding_size


def freeze_layers(model: ArcFaceModel, freeze_ratio: float = 0.8) -> ArcFaceModel:
    """
    Dong bang mot phan backbone de fine-tune
    
    Args:
        model: ArcFaceModel instance
        freeze_ratio: Ty le layers can dong bang (0.8 = 80%)
    """
    backbone_params = list(model.backbone.parameters())
    num_params = len(backbone_params)
    num_freeze = int(num_params * freeze_ratio)
    
    for i, param in enumerate(backbone_params):
        param.requires_grad = (i >= num_freeze)
    
    frozen_count = sum(1 for p in backbone_params if not p.requires_grad)
    trainable_count = sum(1 for p in backbone_params if p.requires_grad)
    
    print(f"Backbone: {frozen_count} params frozen, {trainable_count} params trainable")
    
    return model


def unfreeze_all(model: ArcFaceModel) -> ArcFaceModel:
    """Mo dong bang tat ca layers"""
    for param in model.parameters():
        param.requires_grad = True
    print("Unfreeze all layers")
    return model


def freeze_bn(model: ArcFaceModel) -> ArcFaceModel:
    """Dong bang BatchNorm layers (useful for small batch sizes)"""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
    print("BatchNorm layers frozen")
    return model


def load_pretrained_backbone(model: ArcFaceModel, checkpoint_path: str) -> ArcFaceModel:
    """
    Load pretrained weights cho backbone
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint khong ton tai: {checkpoint_path}")
        print("Su dung ImageNet pretrained weights")
        return model
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Try to load backbone weights
        backbone_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('backbone.'):
                new_key = key.replace('backbone.', '')
                backbone_state_dict[new_key] = value
        
        if backbone_state_dict:
            model.backbone.load_state_dict(backbone_state_dict, strict=False)
            print(f"Da load pretrained backbone tu {checkpoint_path}")
        else:
            # Try loading full model
            model.load_state_dict(state_dict, strict=False)
            print(f"Da load full model tu {checkpoint_path}")
            
    except Exception as e:
        print(f"Loi khi load checkpoint: {e}")
        print("Su dung ImageNet pretrained weights")
    
    return model


def get_model_summary(model: ArcFaceModel) -> dict:
    """Lay thong tin tong quan ve model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'num_classes': model.num_classes,
        'embedding_size': model.embedding_size,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'trainable_ratio': trainable_params / total_params * 100
    }


def test_model():
    """Test model voi dummy data"""
    print("="*50)
    print("Testing ArcFace Model")
    print("="*50)
    
    num_classes = 100
    batch_size = 8
    
    model = ArcFaceModel(num_classes=num_classes, embedding_size=512)
    
    # Model summary
    summary = get_model_summary(model)
    print(f"\nModel Summary:")
    print(f"  - Num classes: {summary['num_classes']}")
    print(f"  - Embedding size: {summary['embedding_size']}")
    print(f"  - Total params: {summary['total_params']:,}")
    print(f"  - Trainable params: {summary['trainable_params']:,} ({summary['trainable_ratio']:.1f}%)")
    
    dummy_images = torch.randn(batch_size, 3, 112, 112)
    dummy_labels = torch.randint(0, num_classes, (batch_size,))
    
    print(f"\nInput shape: {dummy_images.shape}")
    
    # Test training mode
    model.train()
    output, embeddings = model(dummy_images, dummy_labels)
    print(f"Training output shape: {output.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Test inference mode
    model.eval()
    embeddings = model.extract_features(dummy_images)
    print(f"Inference embeddings shape: {embeddings.shape}")
    print(f"Embeddings normalized: {torch.allclose(embeddings.norm(dim=1), torch.ones(batch_size))}")
    
    # Test freezing
    print("\n--- Test Freezing ---")
    model = freeze_layers(model, freeze_ratio=0.8)
    
    summary_frozen = get_model_summary(model)
    print(f"After freezing: {summary_frozen['trainable_params']:,} trainable ({summary_frozen['trainable_ratio']:.1f}%)")
    
    print("\nModel test thanh cong!")
    
    return model


if __name__ == "__main__":
    test_model()
