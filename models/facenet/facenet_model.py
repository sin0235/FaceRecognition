import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1


class FaceNetModel(nn.Module):
    def __init__(self, embedding_size=512, pretrained='vggface2', device='cpu'):
        super(FaceNetModel, self).__init__()

        # Load pretrained InceptionResNetV1
        self.model = InceptionResnetV1(
            pretrained=pretrained,
            classify=False,
            num_classes=None
        ).to(device)

        # Default FaceNet output is 512-d → If user wants 128-d, add FC layer
        self.target_dim = embedding_size
        if embedding_size != 512:
            self.projection = nn.Linear(512, embedding_size)
        else:
            self.projection = None

        self.device = device
        self.to(device)

    def forward(self, x):
        """Forward pass: return embeddings."""
        embeddings = self.model(x)

        if self.projection is not None:
            embeddings = self.projection(embeddings)

        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def extract_embedding(self, x):
        """Inference-only method với no_grad để tiết kiệm memory."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def get_embedding_dim(self):
        """Return embedding dimension."""
        return self.target_dim


# ---------------------------
# Triplet Loss Implementation
# ---------------------------

class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.loss_fn = nn.TripletMarginLoss(
            margin=margin,
            p=2
        )

    def forward(self, anchor, positive, negative):
        loss = self.loss_fn(anchor, positive, negative)
        return loss


# -------------------------------------
# Semi-hard negative mining (simplified)
# -------------------------------------

def mine_triplets(embeddings, labels, margin=0.5):
    """
    embeddings: tensor (batch, dim)
    labels: tensor (batch,)
    """
    triplets = []

    # Compute pairwise distances
    dist_matrix = torch.cdist(embeddings, embeddings)

    batch_size = embeddings.size(0)

    for i in range(batch_size):
        anchor_label = labels[i]
        anchor_emb = embeddings[i]

        # Positives
        pos_indices = torch.where(labels == anchor_label)[0]
        pos_indices = pos_indices[pos_indices != i]  # exclude itself

        if len(pos_indices) == 0:
            continue

        # Negatives
        neg_indices = torch.where(labels != anchor_label)[0]

        for pos_idx in pos_indices:
            pos_dist = dist_matrix[i][pos_idx]

            # Semi-hard: negatives that are farther than positive but within margin
            semi_hard_neg = []
            for neg_idx in neg_indices:
                neg_dist = dist_matrix[i][neg_idx]
                if pos_dist < neg_dist < pos_dist + margin:
                    semi_hard_neg.append(neg_idx)

            # If none found → fallback to hardest negative
            if len(semi_hard_neg) == 0:
                neg_idx = neg_indices[torch.argmax(dist_matrix[i][neg_indices])]
            else:
                neg_idx = semi_hard_neg[0]

            triplets.append((i, pos_idx, neg_idx))

    return triplets
