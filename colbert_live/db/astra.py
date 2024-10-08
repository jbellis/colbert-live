import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import Future
from typing import Any
import uuid

import torch
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster, ResultSet, NoHostAvailable
from cassandra.cluster import EXEC_PROFILE_DEFAULT
from cassandra.concurrent import ConcurrentExecutorListResults
from cassandra.concurrent import execute_concurrent_with_args
from cassandra.policies import ExponentialReconnectionPolicy

from .db import DB

_model_dimensions = {'colbertv2.0': 128,
                     'answerai-colbert-small-v1': 96}

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
    AstraCQL implements the ColBERT Live DB interface for Astra CQL databases as well as local Cassandra.

    This class provides a foundation for creating application-specific implementations.
    Subclasses should override the prepare, process_ann_rows, and process_chunk_rows methods
    to customize the behavior for their specific use case.

    Args:
        keyspace (str): The keyspace to use in the database. AstraCQL will create it if it doesn't exist.
        embedding_dim (int): The dimension of the ColBERT embeddings.
        astra_db_id (Optional[str]): The Astra database ID (required for Astra connections).
        astra_token (Optional[str]): The Astra authentication token (required for Astra connections).
        verbose (bool): If True, print verbose output.

    Attributes:
        session: The database session object.
        query_ann_stmt: The prepared statement for ANN queries.
        query_chunks_stmt: The prepared statement for chunk queries.

    Subclasses must implement:
    - prepare: Set up necessary database statements and perform any required table manipulation.
    - process_ann_rows: Process the results of the ANN query.
    - process_chunk_rows: Process the results of the chunk query.
    See the docstrings of these methods for details.

    Example usage in a subclass:
        class MyDB(AstraCQL):
            def prepare(self, embedding_dim):
                # Create tables and indexes
                self.session.execute(f"CREATE TABLE IF NOT EXISTS ...")
                self.session.execute(f"CREATE CUSTOM INDEX IF NOT EXISTS ...")

                # Prepare statements
                self.query_ann_stmt = self.session.prepare(f"SELECT ... ORDER BY ... ANN OF ...")
                self.query_chunks_stmt = self.session.prepare(f"SELECT ... WHERE ...")

            def process_ann_rows(self, result):
                return [(row.primary_key, row.similarity) for row in result]

            def process_chunk_rows(self, result):
                return [torch.tensor(row.embedding) for row in result]

    Raises:
        Exception: If Astra credentials are incomplete or connection fails.
    """
    def __init__(self,
                 keyspace: str,
                 embedding_dim: int,
                 astra_db_id: str | None = None,
                 astra_token: str | None = None,
                 astra_endpoint: str | None = None,
                 verbose: bool = False):
        self.verbose = verbose
        self.embedding_dim = embedding_dim
        self.keyspace = keyspace
        
        if astra_db_id and astra_endpoint:
            raise ValueError("Both astra_db_id and astra_endpoint cannot be provided simultaneously.")
        
        if astra_endpoint:
            import re
            match = re.search(r'https://([0-9a-f-]{36})-', astra_endpoint)
            if not match:
                raise ValueError("Invalid astra_endpoint format. Expected UUID not found.")
            astra_db_id = match.group(1)
        
        if astra_db_id:
            try:
                uuid.UUID(astra_db_id)
            except ValueError:
                raise ValueError(f"Invalid astra_db_id: {astra_db_id}. It must be a valid UUID.")
        
        if not astra_token:
            if self.verbose: print('Connecting to local Cassandra')
            self._connect_local()
        else:
            if not astra_db_id:
                raise Exception('ASTRA_DB_ID not set')
            if self.verbose: print(f'Connecting to Astra db {astra_db_id}')
            self._connect_astra(astra_token, astra_db_id)
        self.session.default_timeout = 60  # this is the client timeout, server still has internal timeouts

        self._maybe_create_keyspace(astra_db_id, astra_token)
        self.prepare(embedding_dim)

    def prepare(self, embedding_dim: int):
        """
        Prepare the database schema and query statements.  AstraCQL creates the keyspace if necessary;
        everything else is up to you.

        This method should be implemented by subclasses to set up the necessary
        database structure and prepare statements for querying.

        Args:
            embedding_dim (int): The dimension of the ColBERT embeddings.

        Expected implementations:
        1. Create required tables (if not exists)
        2. Create necessary indexes (if not exists)
        3. Prepare two main statements:
           a) query_ann_stmt: For approximate nearest neighbor search
              - Parameters: [query_embedding, query_embedding, limit]
              - Expected result: [(primary_key, similarity)]
              Example:
                SELECT pk, similarity_cosine(embedding, ?) AS similarity
                FROM table
                ORDER BY embedding ANN OF ?
                LIMIT ?

           b) query_chunks_stmt: For retrieving embeddings by primary key
              - Parameters: [primary_key]
              - Expected result: [embedding]
              Example:
                SELECT embedding
                FROM table
                WHERE pk = ?

        Note:
        - Ensure that compound primary keys are represented as tuples in the results.
        - The results of these queries will be processed by process_ann_rows and process_chunk_rows, respectively.
        """
        self.query_ann_stmt = None
        self.query_chunks_stmt = None
        raise NotImplementedError('Subclasses must implement prepare_statements')

    def process_ann_rows(self, result: ResultSet) -> list[tuple[Any, float]]:
        """
        Process the result of the ANN query into a list of (primary_key, similarity) tuples.

        Args:
            result (ResultSet): The result set from the ANN query.

        Returns:
            List[Tuple[Any, float]]: A list of tuples, each containing a primary key and its similarity score.

        Example implementation:
            return [(row.primary_key, row.similarity) for row in result]

        Note:
        - The primary_key should match the structure used in your database schema.
        - For compound primary keys, return them as tuples, e.g., (doc_id, page_num).
        """
        raise NotImplementedError('Subclasses must implement process_ann_rows')

    def process_chunk_rows(self, result: ResultSet) -> list[torch.Tensor]:
        """
        Process the result of the chunk query into a list of embedding tensors.

        Args:
            result (ResultSet): The result set from the chunk query.

        Returns:
            List[torch.Tensor]: A list of embedding tensors.

        Example implementation:
            return [torch.tensor(row.embedding) for row in result]

        Note:
        - Ensure that the returned tensors match the expected embedding dimension.
        - If your database stores embeddings in a different format, convert them to torch.Tensor here.
        """
        raise NotImplementedError('Subclasses must implement process_chunk_rows')

    def query_ann(self, embeddings: torch.Tensor, limit: int) -> list[list[tuple[Any, float]]]:
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

    def query_chunks(self, chunk_ids: list[Any]) -> list[torch.Tensor]:
        if self.verbose: print(f'Loading embeddings from {len(chunk_ids)} chunks for full ColBERT scoring')
        transformed_pks = [pk if isinstance(pk, tuple) else (pk,) for pk in chunk_ids]
        results = execute_concurrent_with_args(self.session, self.query_chunks_stmt, transformed_pks)
        return [torch.stack(self.process_chunk_rows(result))
                for success, result in results if success]

    def _connect_local(self):
        reconnection_policy = ExponentialReconnectionPolicy(base_delay=1, max_delay=60)
        self.cluster = Cluster(reconnection_policy=reconnection_policy)
        try:
            self.session = self.cluster.connect()
        except NoHostAvailable:
            raise ConnectionError("ASTRA_DB_TOKEN and ASTRA_DB_ID not set but Cassandra is not running locally")

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

    def _maybe_create_keyspace(self, db_id, token):
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


class ConcurrentExecutorFutureResults(ConcurrentExecutorListResults):
    def __init__(self, session, statements_and_params, execution_profile, future):
        super().__init__(session, statements_and_params, execution_profile)
        self.future = future

    def _put_result(self, result, idx, success):
        super()._put_result(result, idx, success)
        with self._condition:
            if self._current == self._exec_count:
                if self._exception and self._fail_fast:
                    self.future.set_exception(self._exception)
                else:
                    sorted_results = [r[1] for r in sorted(self._results_queue)]
                    self.future.set_result(sorted_results)


def execute_concurrent_async(
        session,
        statements_and_parameters,
        concurrency=100,
        raise_on_first_error=False,
        execution_profile=EXEC_PROFILE_DEFAULT
):
    """
    Asynchronously executes a sequence of (statement, parameters) tuples concurrently.

    Args:
        session: Cassandra session object.
        statement_and_parameters: Iterable of (prepared CQL statement, bind parameters) tuples.
        concurrency (int, optional): Number of concurrent operations. Default is 100.
        raise_on_first_error (bool, optional): If True, execution stops on the first error. Default is True.
        execution_profile (ExecutionProfile, optional): Execution profile to use. Default is EXEC_PROFILE_DEFAULT.

    Returns:
        A `Future` object that will be completed when all operations are done.
    """
    # Create a Future object and initialize the custom ConcurrentExecutor with the Future
    future = Future()
    executor = ConcurrentExecutorFutureResults(
        session=session,
        statements_and_params=statements_and_parameters,
        execution_profile=execution_profile,
        future=future
    )

    # Execute concurrently
    try:
        executor.execute(concurrency=concurrency, fail_fast=raise_on_first_error)
    except Exception as e:
        future.set_exception(e)

    return future
