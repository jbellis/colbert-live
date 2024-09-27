import uuid
from typing import List, Any

import torch

from colbert_live.db.astra import AstraDB
from cassandra.cluster import ResultSet


class CmdlineDB(AstraDB):
    def __init__(self, keyspace: str, embedding_dim: int, astra_db_id: str, astra_token: str):
        super().__init__(keyspace, embedding_dim, astra_db_id, astra_token, verbose=True)

    def prepare(self, embedding_dim: int):
        # Create tables asynchronously
        futures = []

        # Create pages table
        futures.append(self.session.execute_async(f"""
            CREATE TABLE IF NOT EXISTS {self.keyspace}.pages (
                doc_id uuid,
                page_num int,
                content blob,
                PRIMARY KEY (doc_id, page_num)
            )
        """))

        # Create page_embeddings table
        futures.append(self.session.execute_async(f"""
            CREATE TABLE IF NOT EXISTS {self.keyspace}.page_embeddings (
                doc_id uuid,
                page_num int,
                embedding_id int,
                embedding vector<float, {embedding_dim}>,
                PRIMARY KEY (doc_id, page_num, embedding_id)
            )
        """))

        # Wait for all CREATE TABLE operations to complete
        for future in futures:
            future.result()

        # Create colbert_ann index
        index_future = self.session.execute_async(f"""
            CREATE CUSTOM INDEX IF NOT EXISTS colbert_ann 
            ON {self.keyspace}.page_embeddings(embedding) 
            USING 'StorageAttachedIndex'
            WITH OPTIONS = {{ 'source_model': 'bert' }}
        """)

        # Prepare statements
        self.insert_page_stmt = self.session.prepare(f"""
            INSERT INTO {self.keyspace}.pages (doc_id, page_num, content) VALUES (?, ?, ?)
        """)
        self.insert_embedding_stmt = self.session.prepare(f"""
            INSERT INTO {self.keyspace}.page_embeddings (doc_id, page_num, embedding_id, embedding) VALUES (?, ?, ?, ?)
        """)
        self.query_chunks_stmt = self.session.prepare(f"""
            SELECT embedding FROM {self.keyspace}.page_embeddings WHERE doc_id = ? AND page_num = ?
        """)

        index_future.result()
        self.query_ann_stmt = self.session.prepare(f"""
            SELECT doc_id, page_num, similarity_cosine(embedding, ?) AS similarity
            FROM {self.keyspace}.page_embeddings
            ORDER BY embedding ANN OF ?
            LIMIT ?
        """)

        print("Schema ready")

    def add_documents(self, pages: List[bytes]) -> uuid.UUID:
        doc_id = uuid.uuid4()

        for page_num, page_content in enumerate(pages, start=1):
            self.session.execute(self.insert_page_stmt, (doc_id, page_num, page_content))

        return doc_id

    def add_embeddings(self, doc_id: uuid.UUID, page_num: int, embeddings: torch.Tensor):
        for embedding_id, embedding in enumerate(embeddings):
            self.session.execute(self.insert_embedding_stmt,
                                 (doc_id, page_num, embedding_id, embedding.tolist()))

    def process_ann_rows(self, result: ResultSet) -> List[tuple[Any, float]]:
        return [((row.doc_id, row.page_num), row.similarity) for row in result]

    def process_chunk_rows(self, result: ResultSet) -> List[torch.Tensor]:
        return [torch.tensor(row.embedding) for row in result]

    def get_page_content(self, doc_id: uuid.UUID, page_num: int) -> bytes:
        query = f"SELECT content FROM {self.keyspace}.pages WHERE doc_id = %s AND page_num = %s"
        result = self.session.execute(query, [doc_id, page_num])
        row = result.one()
        return row.content if row else None
