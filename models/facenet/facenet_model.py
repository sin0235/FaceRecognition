import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1


class FaceNetModel(nn.Module):
    def __init__(self, embedding_size=512, pretrained='vggface2'):
        super().__init__()

        self.backbone = InceptionResnetV1(
            pretrained=pretrained,
            classify=False
        )

        if embedding_size != 512:
            self.projection = nn.Linear(512, embedding_size)
        else:
            self.projection = None

    def forward(self, x):
        # ⚠️ KHÔNG DÙNG no_grad khi training
        embeddings = self.backbone(x)

        if self.projection is not None:
            embeddings = self.projection(embeddings)

        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)
