import json
import os
import time
import urllib.error
import urllib.error
import urllib.request
import urllib.request
from itertools import chain
from typing import List, Tuple
from typing import Optional, Any
from uuid import UUID, uuid4

import torch
from astrapy import DataAPIClient
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster, ResultSet
from cassandra.concurrent import execute_concurrent_with_args
from cassandra.policies import ExponentialReconnectionPolicy

from .db import DB


def _get_astra_bundle_url(db_id, token):
    # set up the request
    url = f"https://api.astra.datastax.com/v2/databases/{db_id}/secureBundleURL"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    req = urllib.request.Request(url, method="POST", headers=headers, data=b"")
    try:
        with urllib.request.urlopen(req) as response:
            response_data = json.loads(response.read().decode())
            # happy path
            if 'downloadURL' in response_data:
                return response_data['downloadURL']
            # handle errors
            if 'errors' in response_data:
                raise Exception(response_data['errors'][0]['message'])
            raise Exception('Unknown error in ' + str(response_data))
    except urllib.error.URLError as e:
        raise Exception(f"Error connecting to Astra API: {str(e)}")

def _get_secure_connect_bundle(token: str, db_id: str, verbose: bool = False) -> str:
    scb_path = f'astra-secure-connect-{db_id}.zip'
    if not os.path.exists(scb_path):
        if verbose: print('Downloading Secure Cloud Bundle')
        url = _get_astra_bundle_url(db_id, token)
        try:
            with urllib.request.urlopen(url) as r:
                with open(scb_path, 'wb') as f:
                    f.write(r.read())
        except urllib.error.URLError as e:
            raise Exception(f"Error downloading secure connect bundle: {str(e)}")
    return scb_path


class AstraCQL(DB):
    """
    AstraDB implements the ColBERT Live DB interface for Astra databases as well as local Cassandra.

    Args:
        keyspace (str): The keyspace to use in the database.  AstraDB will create it if it doesn't exist
        embedding_dim (int): The dimension of the ColBERT embeddings
        astra_db_id (Optional[str]): The Astra database ID (required for Astra connections)
        astra_token (Optional[str]): The Astra authentication token (required for Astra connections)
        verbose (bool): If True, print verbose output

    Subclasses must implement:
    - prepare: Set up necessary database statements and perform any required schema manipulation
    - process_ann_rows: Process the results of the ANN query
    - process_chunk_rows: Process the results of the chunk query

    Raises:
        Exception: If Astra credentials are incomplete or connection fails
    """
    def __init__(self,
                 keyspace: str,
                 embedding_dim: int,
                 astra_db_id: Optional[str],
                 astra_token: Optional[str],
                 verbose: bool = False):
        self.verbose = verbose
        self.embedding_dim = embedding_dim
        self.keyspace = keyspace
        
        if not astra_token:
            if self.verbose: print('Connecting to local Cassandra')
            self._connect_local()
        else:
            if not astra_db_id:
                raise Exception('ASTRA_DB_ID not set')
            if self.verbose: print(f'Connecting to Astra db {astra_db_id}')
            self._connect_astra(astra_token, astra_db_id)
        self.session.default_timeout = 60  # this is the client timeout, server still has internal timeouts

        self.maybe_create_keyspace(astra_db_id, astra_token)
        self.prepare(embedding_dim)

    def prepare(self, embedding_dim: int):
        """
        Prepare two statements for querying the database:
        - query_colbert_ann_stmt
          - three parameters: [embedding, embedding, limit]
          - resultset: [(pk, similarity)]
          Example: `SELECT pk, similarity_cosine(embedding, ?) FROM table ORDER BY ? ANN OF column_name LIMIT ?`
        - query_colbert_chunks_stmt
          - one parameter: [pk]
          - resultset: [embedding] (multiple rows per pk)
          Example: `SELECT embedding FROM table WHERE pk = ?`

        The results of these queries will be processed by process_ann_rows and process_chunk_rows, respectively.
        MAKE SURE that compound primary keys are represented as tuples, or `query_ann` will fail.

        Other idempotent initialization logic (e.g. schema manipulation) may also be done here.
        """
        self.query_ann_stmt = None
        self.query_chunks_stmt = None
        raise NotImplementedError('Subclasses must implement prepare_statements')

    def process_ann_rows(self, result: ResultSet) -> list[tuple[Any, float]]:
        """
        Turn a resultset from query_colbert_ann_stmt, into a list of (pk, similarity) tuples.
        """
        raise NotImplementedError('Subclasses must implement process_ann_rows')

    def process_chunk_rows(self, result: ResultSet) -> list[torch.Tensor]:
        """
        Turn a resultset from query_colbert_chunks_stmt, into a list of embeddings.
        """
        raise NotImplementedError('Subclasses must implement process_chunk_rows')

    def query_ann(self, embeddings: torch.Tensor, limit: int) -> list[list[tuple[str, float]]]:
        if self.verbose: print(f'Querying ANN with {len(embeddings)} embeddings')
        embedding_list = embeddings.tolist()
        params = [(emb, emb, limit) for emb in embedding_list]
        results = execute_concurrent_with_args(self.session, self.query_ann_stmt, params)

        ann_results = []
        for success, result in results:
            if not success:
                raise Exception('Failed to execute ANN query')
            ann_results.append(self.process_ann_rows(result))

        return ann_results

    def query_chunks(self, chunk_ids: list[Any]) -> list[list[torch.Tensor]]:
        if self.verbose: print(f'Loading embeddings from {len(chunk_ids)} chunks for full ColBERT scoring')
        transformed_pks = [pk if isinstance(pk, tuple) else (pk,) for pk in chunk_ids]
        results = execute_concurrent_with_args(self.session, self.query_chunks_stmt, transformed_pks)
        chunk_embeddings = []
        for success, result in results:
            if not success:
                raise Exception('Failed to retrieve chunk embeddings')
            chunk_embeddings.append(self.process_chunk_rows(result))
        return chunk_embeddings

    def _connect_local(self):
        reconnection_policy = ExponentialReconnectionPolicy(base_delay=1, max_delay=60)
        self.cluster = Cluster(reconnection_policy=reconnection_policy)
        self.session = self.cluster.connect()

    def _connect_astra(self, token: str, db_id: str):
        scb_path = _get_secure_connect_bundle(token, db_id, self.verbose)
        cloud_config = {
            'secure_connect_bundle': scb_path
        }
        auth_provider = PlainTextAuthProvider('token', token)
        reconnection_policy = ExponentialReconnectionPolicy(base_delay=1, max_delay=60)
        self.cluster = Cluster(
            cloud=cloud_config,
            auth_provider=auth_provider,
            reconnection_policy=reconnection_policy
        )
        self.session = self.cluster.connect()
        if self.verbose: print(f"Connected to Astra db {db_id}")

    def maybe_create_keyspace(self, db_id, token):
        if token:
            # Use Astra REST API to create keyspace
            url = f"https://api.astra.datastax.com/v2/databases/{db_id}/keyspaces/{self.keyspace}"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            data = json.dumps({
                "name": self.keyspace
            }).encode('utf-8')
            
            req = urllib.request.Request(url, method="POST", headers=headers, data=data)
            
            try:
                with urllib.request.urlopen(req) as response:
                    if response.status == 201:
                        if self.verbose: print(f"Keyspace '{self.keyspace}' created or verified")
                        # Wait for the keyspace to be available (max 10 seconds)
                        start_time = time.time()
                        while time.time() - start_time < 10:
                            try:
                                self.session.execute(f"USE {self.keyspace}")
                                break
                            except Exception:
                                time.sleep(0.1)
                        else:
                            raise Exception(f"Keyspace '{self.keyspace}' creation successful, but still unavailable after 10 seconds")
                    else:
                        raise Exception(f"Failed to create keyspace: {response.read().decode()}")
            except urllib.error.HTTPError as e:
                raise Exception(f"Failed to create keyspace: {e.read().decode()}")
        else:
            # Use CQL to create keyspace
            create_keyspace_query = f"""
            CREATE KEYSPACE IF NOT EXISTS {self.keyspace}
            WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}}
            """
            self.session.execute(create_keyspace_query)
            if self.verbose: print(f"Keyspace '{self.keyspace}' created or verified")


class AstraDoc(DB):
    def __init__(self, collection_name: str, embedding_dim: int):
        """
        Initialize the AstraDoc class.

        Args:
            collection_name (str): The name of the collection to use for documents.
        """
        self._client = DataAPIClient(token=os.environ["ASTRA_DB_TOKEN"])
        self._db = self._client.get_database(os.environ["ASTRA_DB_ID"])
        self._docs = None
        chunks_collection_name = f"{collection_name}_chunks"
        embeddings_collection_name = f"{collection_name}_embeddings"
        collections = set(c.name for c in self._db.list_collections())
        if collection_name in collections:
            self._docs = self._db[collection_name]
        else:
            self._docs = self._db.create_collection(collection_name)
        if chunks_collection_name in collections:
            self._chunks = self._db[chunks_collection_name]
        else:
            self._chunks = self._db.create_collection(
                chunks_collection_name,
                indexing={'deny': ['body']},
            )
        # Ideally, the embeddings would just be a list in the chunks collection,
        # but Astra doesn't know how to index a list of vectors yet
        if embeddings_collection_name in collections:
            self._embeddings = self._db[embeddings_collection_name]
        else:
            self._embeddings = self._db.create_collection(
                embeddings_collection_name,
                dimension=embedding_dim,
                # TODO source_model isn't exposed yet
            )

    def query_ann(self, embeddings: torch.Tensor, limit: int) -> List[List[Tuple[UUID, float]]]:
        ann_results = []
        for embedding in embeddings:
            results = self._embeddings.find(
                {},
                sort={"$vector": embedding.tolist()},
                limit=limit,
                projection={"_chunk_id": 1, "$similarity": 1}
            )
            ann_results.append([(r['_chunk_id'], r['$similarity']) for r in results])
        return ann_results

    def query_chunks(self, chunk_ids: List[UUID]) -> List[List[torch.Tensor]]:
        chunk_results = []
        for chunk_id in chunk_ids:
            r = self._chunks.find_one(
                {"_id": chunk_id},
                projection={"_embedding_ids": 1}
            )
            embedding_ids = r['_embedding_ids']
            results = self._embeddings.find(
                {"_id": {"$in": embedding_ids}},
                projection={"$vector": 1}
            )
            chunk_results.append([torch.tensor(r['$vector']) for r in results])
        return chunk_results

    def insert(self, document: dict, chunks: list[dict], embeddings: list[torch.Tensor]):
        """
        Insert the document, chunks, and embeddings associated with the given document.

        Args:
            document: The document to insert; all items in the dict will be stored as fields.
                      Must contain an '_id' field of any type.
            chunks (list): The chunks of the document; all items in each dict will be stored as fields
                           with a generated ID
            embeddings (list[torch.Tensor]): a 2D tensor of embeddings per chunk
        """
        for chunk, embedding_tensors in zip(chunks, embeddings):
            chunk['_id'] = uuid4()
            embedding_docs = [{'_id': uuid4(), 'chunk_id': chunk['_id'], '$vector': embedding.tolist()}
                              for embedding in embeddings]
            chunk['_embedding_ids'] = [doc['_id'] for doc in embedding_docs]
            self._embeddings.many(embedding_docs)
        document['_chunk_ids'] = [chunk['_id'] for chunk in chunks]

        self._chunks.insert_many(chunks)
        self._docs.insert_one(document)

    def delete(self, doc_id):
        """
        Delete the document, chunks, and embeddings associated with the given document.

        Args:
            doc_id: The ID of the document to delete.
        """
        doc = self._docs.find_one({"_id": doc_id})
        chunks = self._chunks.find({"_id": {"$in": doc['_chunk_ids']}})
        all_embedding_ids = list(chain.from_iterable(chunk['_embedding_ids'] for chunk in chunks))
        self._embeddings.delete_many({"_id": {"$in": all_embedding_ids}})
        self._chunks.delete_many({"_id": {"$in": doc['_chunk_ids']}})
        self._docs.delete_one({"_id": doc_id})
