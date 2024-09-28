from uuid import uuid4, UUID
import base64

import torch

from colbert_live.db.astra import AstraDoc


class CmdlineDB(AstraDoc):
    """
    To keep things as simple as possible we basically ignore the top-level record as much as possible.
    For a "real" application you would want to flesh that out.
    """
    def __init__(self, collection_name: str, embedding_dim: int):
        super().__init__(collection_name, embedding_dim)

    def add_record(self, pages: list[bytes], embeddings: list[torch.Tensor]) -> UUID:
        doc_id = uuid4()
        record = {'_id': doc_id,}
        chunks = [dict(body=base64.b64encode(bytes).decode('utf-8'), page_num=i) for i, bytes in enumerate(pages, start=1)]
        self.insert(record, chunks, embeddings)
        return doc_id

    def get_page_content(self, chunk_id: UUID) -> bytes:
        page = self._chunks.find_one({'_id': chunk_id})
        page['body'] = base64.b64decode(page['body'])
        return page
