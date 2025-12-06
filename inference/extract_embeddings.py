import os
import numpy as np


def fake_model(image_path: str) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(image_path)) % (2**32))
    return rng.random(128, dtype=np.float32)


def extract_embedding_single_image(img_path: str) -> np.ndarray | None:
    if not os.path.exists(img_path):
        return None
    emb = fake_model(img_path)
    return emb


def extract_embedding_for_folder(folder: str) -> np.ndarray | None:
    files = os.listdir(folder)
    embeddings = []

    for f in files:
        name = f.lower()
        if name.endswith(".jpg") or name.endswith(".jpeg") or name.endswith(".png"):
            path = os.path.join(folder, f)
            emb = extract_embedding_single_image(path)
            if emb is not None:
                embeddings.append(emb)

    if len(embeddings) == 0:
        return None

    stacked = np.stack(embeddings, axis=0)
    mean_emb = np.mean(stacked, axis=0)
    return mean_emb


def build_db(root_folder: str = "data/celeb", save_path: str = "data/embeddings_db.npy") -> None:
    if not os.path.exists(root_folder):
        print("Root folder does not exist:", root_folder)
        return

    db: dict[str, np.ndarray] = {}

    for person in os.listdir(root_folder):
        person_folder = os.path.join(root_folder, person)
        if not os.path.isdir(person_folder):
            continue

        print("Processing:", person)
        emb = extract_embedding_for_folder(person_folder)
        if emb is not None:
            db[person] = emb

    if len(db) == 0:
        print("No embeddings created")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, db)
    print("Saved database to", save_path)


if __name__ == "__main__":
    build_db()
