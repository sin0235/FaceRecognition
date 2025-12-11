import torch
from models.facenet.facenet_model import FaceNetModel

def test_forward_pass():
    model = FaceNetModel(embedding_size=512, device='cpu')
    dummy = torch.randn(4, 3, 160, 160)  # batch 4

    out = model(dummy)
    assert out.shape == (4, 512)
    assert torch.allclose(out.norm(dim=1), torch.ones(4), atol=1e-3)

    print("Forward pass OK:", out.shape)
