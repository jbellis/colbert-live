import os
import io
from itertools import cycle
from typing import Any
import torch
import torch.nn.functional as F
from PIL import Image
from colbert_live.db import DB
from colbert_live.models import Model

class InMemoryDB(DB):
    def __init__(self, model: Model):
        self.model = model
        self.encodings: dict[str, torch.Tensor] = {}
        self._load_and_encode_images()

    def _load_and_encode_images(self):
        resources_dir = os.path.join(os.path.dirname(__file__), 'resources')
        for filename in os.listdir(resources_dir):
            if not filename.lower().endswith('.png'):
                continue
            file_path = os.path.join(resources_dir, filename)
            with Image.open(file_path) as img:
                self.encodings[filename] = self.model.encode_doc([img])[0]

    def query_ann(self, embeddings: torch.Tensor, limit: int) -> list[list[tuple[Any, float]]]:
        results = []
        for query_embedding in embeddings:
            similarities = []
            for filename, encoding in self.encodings.items():
                # Calculate similarity for each 1D vector in the 2D encoding tensor
                similarities_per_encoding = F.cosine_similarity(query_embedding, encoding, dim=0).tolist()
                similarities.extend(zip(cycle([filename]), similarities_per_encoding))
            similarities.sort(key=lambda x: x[1], reverse=True)
            results.append(similarities[:limit])
        return results

    def query_chunks(self, pks: list[Any]) -> list[torch.Tensor]:
        return [self.encodings[pk] for pk in pks if pk in self.encodings]
