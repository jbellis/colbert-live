import random
import time

import torch
from cassandra import ConsistencyLevel
from cassandra.policies import RetryPolicy

from colbert_live.db.astra import AstraDB


# All we care about for BEIR is getting our data in and out, we don't care about latencies.
# Don't use this as a general-purpose RetryPolicy.
class ExponentialRetryPolicy(RetryPolicy):
    def __init__(self, max_retries, base_delay, max_delay):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    def _exponential_backoff(self, retry_num):
        delay = min(self.base_delay * (2 ** retry_num), self.max_delay)
        jitter = random.uniform(0, 0.1 * delay)
        return delay + jitter

    def on_read_timeout(self, query, consistency, required_responses,
                        received_responses, data_retrieved, retry_num):
        if retry_num < self.max_retries:
            delay = self._exponential_backoff(retry_num)
            time.sleep(delay)
            return self.RETRY, consistency
        return self.RETHROW, None

    def on_write_timeout(self, query, consistency, write_type,
                         required_responses, received_responses, retry_num):
        if retry_num < self.max_retries:
            delay = self._exponential_backoff(retry_num)
            time.sleep(delay)
            return self.RETRY, consistency
        return self.RETHROW, None

    def on_unavailable(self, query, consistency, required_replicas, alive_replicas, retry_num):
        if retry_num < self.max_retries:
            delay = self._exponential_backoff(retry_num)
            time.sleep(delay)
            return self.RETRY_NEXT_HOST, consistency
        return self.RETHROW, None

    def on_request_error(self, query, consistency, error, retry_num):
        if retry_num < self.max_retries:
            delay = self._exponential_backoff(retry_num)
            time.sleep(delay)
            return self.RETRY_NEXT_HOST, consistency
        return self.RETHROW, None


class AstraDBBeir(AstraDB):
    def __init__(self, keyspace: str, embedding_dim: int, astra_db_id: str, astra_token: str):
        super().__init__(keyspace, embedding_dim, astra_db_id, astra_token)
        self.cluster.default_retry_policy = ExponentialRetryPolicy(max_retries=5, base_delay=1, max_delay=60)

    def prepare(self, embedding_dim):
        # Create chunks table
        self.session.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.keyspace}.chunks (
                id text,
                title text,
                body text,
                PRIMARY KEY (id)
            )
        """)

        # Create colbert_embeddings table
        self.session.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.keyspace}.colbert_embeddings (
                chunk_id text,
                embedding_id int,
                bert_embedding vector<float, {embedding_dim}>,
                PRIMARY KEY (chunk_id, embedding_id)
            )
        """)

        # Create colbert_ann index
        self.session.execute(f"""
            CREATE CUSTOM INDEX IF NOT EXISTS colbert_ann 
            ON {self.keyspace}.colbert_embeddings(bert_embedding) 
            USING 'StorageAttachedIndex'
            WITH OPTIONS = {{ 'source_model': 'bert' }}
        """)

        query_colbert_ann_cql = f"""
                SELECT chunk_id, similarity_dot_product(?, bert_embedding) as similarity
                FROM {self.keyspace}.colbert_embeddings
                ORDER BY bert_embedding ANN OF ?
                LIMIT ?
                """
        self.query_ann_stmt = self.session.prepare(query_colbert_ann_cql)
        self.query_ann_stmt.consistency_level = ConsistencyLevel.LOCAL_ONE

        query_colbert_parts_cql = f"""
                SELECT chunk_id, bert_embedding
                FROM {self.keyspace}.colbert_embeddings
                WHERE chunk_id = ?
                """
        self.query_chunks_stmt = self.session.prepare(query_colbert_parts_cql)

        # Not part of colbert.DB api
        self.insert_chunk_stmt = self.session.prepare(f"""
        INSERT INTO {self.keyspace}.chunks (id, title, body)
        VALUES (?, ?, ?)
        """)
        self.insert_embeddings_stmt = self.session.prepare(f"""
        INSERT INTO {self.keyspace}.colbert_embeddings (chunk_id, embedding_id, bert_embedding)
        VALUES (?, ?, ?)
        """)

    def process_ann_rows(self, result):
        return [(row.chunk_id, row.similarity) for row in result]

    def process_chunk_rows(self, result):
        return [torch.tensor(row.bert_embedding) for row in result]
