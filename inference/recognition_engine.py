import os
import numpy as np
from .extract_embeddings import extract_embedding_single_image


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(float)
    b = b.astype(float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class RecognitionEngine:
    def __init__(self, db_path: str = "data/embeddings_db.npy") -> None:
        if not os.path.exists(db_path):
            raise FileNotFoundError("Database file not found: " + db_path)
        self.db: dict[str, np.ndarray] = np.load(db_path, allow_pickle=True).item()

    def recognize(self, img_path: str, threshold: float = 0.5) -> tuple[str, float]:
        emb = extract_embedding_single_image(img_path)
        if emb is None:
            return "No face or file not found", 0.0

        best_name = "Unknown"
        best_score = -1.0

        for name, vec in self.db.items():
            score = cosine_similarity(emb, vec)
            if score > best_score:
                best_score = score
                best_name = name

        if best_score < threshold:
            return "Unknown", best_score

        return best_name, best_score


if __name__ == "__main__":
    engine = RecognitionEngine()

    test_image = "data/celeb/mytam/anh1.jpg"
    if not os.path.exists(test_image):
        print("Change test_image path in recognition_engine.py to an existing file.")
    else:
        name, score = engine.recognize(test_image)
        print("Result:", name)
        print("Score:", score)
