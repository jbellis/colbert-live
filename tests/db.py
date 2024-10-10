import os
from itertools import cycle
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from colbert_live.db import DB
from colbert_live.models import Model


class InMemoryDB(DB):
    def __init__(self, model: Model, filetype: str):
        self.model = model
        self.filetype = filetype
        self.encodings: dict[str, torch.Tensor] = {}

        resources_dir = os.path.join(os.path.dirname(__file__), 'resources')
        for filename in os.listdir(resources_dir):
            if not filename.lower().endswith('.' + self.filetype):
                continue
            file_path = os.path.join(resources_dir, filename)
            self.encodings[filename] = self.encode_file(file_path)
            
    def encode_file(self, file_path: str):
        raise NotImplementedError()

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


class ColpaliTestDB(InMemoryDB):
    def __init__(self, model: Model):
        super().__init__(model, "png")

    def encode_file(self, file_path: str):
        with Image.open(file_path) as img:
            return self.model.encode_doc([img])[0]


class ColbertTestDB(InMemoryDB):
    def __init__(self, model: Model):
        super().__init__(model, "txt")

    def encode_file(self, file_path: str):
        with open(file_path, 'r') as f:
            return self.model.encode_doc([f.read()])[0]
