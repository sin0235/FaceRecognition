import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class FaceNetTripletDataset(Dataset):
    def __init__(self, root_dir):
        """
        root_dir: data/CelebA_Aligned_Balanced/train
        """
        self.root_dir = root_dir
        self.identities = os.listdir(root_dir)

        self.id_to_images = {
            pid: os.listdir(os.path.join(root_dir, pid))
            for pid in self.identities
            if len(os.listdir(os.path.join(root_dir, pid))) >= 2
        }

        self.identities = list(self.id_to_images.keys())

        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),          # resize 160x160
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )  # [-1, 1]
        ])

    def __len__(self):
        return len(self.identities)

    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def __getitem__(self, index):
        # Anchor & Positive (same identity)
        anchor_id = self.identities[index]
        imgs = self.id_to_images[anchor_id]
        a_name, p_name = random.sample(imgs, 2)

        # Negative (different identity)
        neg_id = random.choice([i for i in self.identities if i != anchor_id])
        n_name = random.choice(self.id_to_images[neg_id])

        anchor = self._load_image(os.path.join(self.root_dir, anchor_id, a_name))
        positive = self._load_image(os.path.join(self.root_dir, anchor_id, p_name))
        negative = self._load_image(os.path.join(self.root_dir, neg_id, n_name))

        return anchor, positive, negative
