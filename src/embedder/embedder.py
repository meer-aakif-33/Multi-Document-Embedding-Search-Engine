from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List


class Embedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        # returns numpy array (n, dim)
        embs = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embs

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    @staticmethod
    def normalize(embs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        return embs / norms


if __name__ == '__main__':
    e = Embedder()
    v = e.embed('hello world')
    print(v.shape)