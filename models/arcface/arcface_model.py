"""
ArcFace Model Implementation
Bao gồm ResNet50 backbone và ArcFace loss layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


class ArcMarginProduct(nn.Module):
    """
    ArcFace Loss Layer (Additive Angular Margin)
    """
    def __init__(self, in_features, out_features, scale=64.0, margin=0.5, easy_margin=False):
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
    
    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
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
    """
    def __init__(self, pretrained=True):
        super(ResNetBackbone, self).__init__()
        
        import torchvision.models as models
        resnet = models.resnet50(pretrained=pretrained)
        
        # Lấy tất cả layers trừ FC cuối
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.avgpool = resnet.avgpool
        
    def forward(self, x):
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
    ArcFace Model hoàn chỉnh
    """
    def __init__(self, num_classes, embedding_size=512, pretrained=True, 
                 scale=64.0, margin=0.5, easy_margin=False):
        super(ArcFaceModel, self).__init__()
        
        self.backbone = ResNetBackbone(pretrained=pretrained)
        
        # Feature dimension từ ResNet50 là 2048
        self.bn1 = nn.BatchNorm1d(2048)
        self.dropout = nn.Dropout(p=0.5)
        
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
    
    def forward(self, x, labels=None):
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
    
    def extract_features(self, x):
        """
        Trích xuất embeddings (dùng cho inference)
        """
        with torch.no_grad():
            embeddings = self.forward(x, labels=None)
            embeddings = F.normalize(embeddings)
        return embeddings


def freeze_layers(model, freeze_ratio=0.8):
    """
    Đóng băng một phần backbone để fine-tune
    
    Args:
        model: ArcFaceModel instance
        freeze_ratio: Tỷ lệ layers cần đóng băng (0.8 = 80%)
    """
    backbone_params = list(model.backbone.parameters())
    num_params = len(backbone_params)
    num_freeze = int(num_params * freeze_ratio)
    
    for i, param in enumerate(backbone_params):
        if i < num_freeze:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    frozen_count = sum(1 for p in backbone_params if not p.requires_grad)
    trainable_count = sum(1 for p in backbone_params if p.requires_grad)
    
    print(f"Backbone: {frozen_count} layers đóng băng, {trainable_count} layers trainable")
    
    return model


def load_pretrained_backbone(model, checkpoint_path):
    """
    Load pretrained weights cho backbone
    
    Args:
        model: ArcFaceModel instance
        checkpoint_path: Đường dẫn đến file checkpoint
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        backbone_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('backbone.'):
                new_key = key.replace('backbone.', '')
                backbone_state_dict[new_key] = value
        
        if backbone_state_dict:
            model.backbone.load_state_dict(backbone_state_dict, strict=False)
            print(f"Đã load pretrained backbone từ {checkpoint_path}")
        else:
            print("Không tìm thấy backbone weights trong checkpoint")
            
    except Exception as e:
        print(f"Lỗi khi load checkpoint: {e}")
        print("Sử dụng ImageNet pretrained weights từ torchvision")


def test_model():
    """
    Test model với dummy data
    """
    print("Testing ArcFace Model...")
    
    num_classes = 100
    batch_size = 8
    
    model = ArcFaceModel(num_classes=num_classes, embedding_size=512)
    model.eval()
    
    dummy_images = torch.randn(batch_size, 3, 112, 112)
    dummy_labels = torch.randint(0, num_classes, (batch_size,))
    
    print(f"Input shape: {dummy_images.shape}")
    
    # Test training mode
    model.train()
    output, embeddings = model(dummy_images, dummy_labels)
    print(f"Training output shape: {output.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Test inference mode
    model.eval()
    embeddings = model.extract_features(dummy_images)
    print(f"Inference embeddings shape: {embeddings.shape}")
    
    # Test freezing
    model = freeze_layers(model, freeze_ratio=0.8)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTổng parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    print("\nModel test thành công!")


if __name__ == "__main__":
    test_model()
