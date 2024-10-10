# Colbert Live!

This documents how to implement a DB subclass for use with ColbertLive.  DB is an abstract class that looks like this:
## DB

### Methods

#### `query_ann(self, embeddings: torch.Tensor, limit: int, params: dict[str, typing.Any] = {}) -> list[list[tuple[typing.Any, float]]]`

Perform an approximate nearest neighbor (ANN) search of the ColBERT embeddings.
This informs the decision of which documents to fetch for full ColBERT scoring.


**Arguments:**

- `embeddings`: A 2D tensor of ColBERT embeddings to compare against.
- `limit`: The maximum number of results to return for each embedding.
- `params`: Additional search parameters, if any.
  

**Returns:**

  A list of lists, one per embedding, where each inner list contains tuples of (PK, similarity)
  for the chunks closest to each query embedding.

#### `query_chunks(self, pks: list[typing.Any]) -> Iterable[torch.Tensor]`

Retrieve all ColBERT embeddings for specific chunks so that ColBERT scores
can be computed.


**Arguments:**

- `pks`: A list of primary keys (of any object type) identifying the chunks.
- `params`: Additional search parameters, if any.
  

**Returns:**

  An iterable of 2D tensors representing the ColBERT embeddings for each of the specified chunks.




As an example, here is the source for AstraCQL, which implements DB for use with Astra, a hosted Cassandra database
that uses CQL.
```
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

    Subclasses must implement:
    - prepare: Set up necessary database statements and perform any required table manipulation.
    - get_query_ann: Set up prepared statement and bind vars for ANN queries.
    - get_query_chunks_stmt: Return prepared statement for chunk queries.
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

            def get_query_ann(self, embeddings, limit, params):
                params_list = [(emb, emb, limit) for emb in embeddings.tolist()]
                return self.query_ann_stmt, params_list

            def get_query_chunks_stmt(self, chunk_ids):
                return self.query_chunks_stmt

    Raises:
        Exception: If Astra credentials are incomplete or connection fails.
    """

    def __init__(self, keyspace: str, embedding_dim: int, astra_db_id: str | None=None, astra_token: str | None=None, astra_endpoint: str | None=None, verbose: bool=False):
        self.verbose = verbose
        self.embedding_dim = embedding_dim
        self.keyspace = keyspace
        if astra_db_id and astra_endpoint:
            raise ValueError('Both astra_db_id and astra_endpoint cannot be provided simultaneously.')
        if astra_endpoint:
            import re
            match = re.search('https://([0-9a-f-]{36})-', astra_endpoint)
            if not match:
                raise ValueError('Invalid astra_endpoint format. Expected UUID not found.')
            astra_db_id = match.group(1)
        if astra_db_id:
            try:
                uuid.UUID(astra_db_id)
            except ValueError:
                raise ValueError(f'Invalid astra_db_id: {astra_db_id}. It must be a valid UUID.')
        if not astra_token:
            if self.verbose:
                print('Connecting to local Cassandra')
            self._connect_local()
        else:
            if not astra_db_id:
                raise Exception('ASTRA_DB_ID not set')
            if self.verbose:
                print(f'Connecting to Astra db {astra_db_id}')
            self._connect_astra(astra_token, astra_db_id)
        self.session.default_timeout = 60
        self._maybe_create_keyspace(astra_db_id, astra_token)
        self.prepare(embedding_dim)

    @abstractmethod
    def prepare(self, embedding_dim: int):
        """
        Prepare the database schema and query statements. AstraCQL creates the keyspace if necessary;
        everything else is up to you.

        This method should be implemented by subclasses to set up the necessary
        database structure and prepare any additional statements for querying.

        Args:
            embedding_dim (int): The dimension of the ColBERT embeddings.

        Expected implementations:
        1. Create required tables (if not exists)
        2. Create necessary indexes (if not exists)
        3. Prepare any additional statements needed for your specific implementation

        Note:
        - The query_ann_stmt and query_chunks_stmt are now abstract methods and should be implemented separately.
        - Ensure that compound primary keys are represented as tuples in the results.
        """

    @abstractmethod
    def get_query_ann(self, embeddings: torch.Tensor, limit: int, params: dict[str, Any]) -> tuple[PreparedStatement, list[tuple]]:
        """
        Abstract method for setting up the ANN query.

        Args:
            embeddings: a 2D tensor of query embeddings to compare against.
            limit: The maximum number of results to return for each embedding.
            params: Additional parameters to pass to the query, if any.

        Returns:
            A prepared statement and a list of bind variables to pass to the query (one element per embedding).
        """

    @abstractmethod
    def get_query_chunks_stmt(self) -> PreparedStatement:
        """
        Abstract method for the chunks query.

        Returns:
            A prepared statement for retrieving embeddings by primary key.
        """

    @abstractmethod
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

    @abstractmethod
    def process_chunk_rows(self, result: ResultSet) -> list[torch.Tensor]:
        """
        Process the result of the chunk query into a list of embedding tensors.

        Args:
            result (ResultSet): The result set from the chunk query.

        Returns:
            List[torch.Tensor]: A list of embedding tensors.

        Example implementation:
            return [torch.tensor(row.embedding) for row in result]
        """

    def query_ann(self, embeddings: torch.Tensor, limit: int, params: dict[str, Any]={}) -> list[list[tuple[Any, float]]]:
        if self.verbose:
            print(f'Querying ANN with {len(embeddings)} embeddings')
        (stmt, params_list) = self.get_query_ann(embeddings, limit, params)
        results = execute_concurrent_with_args(self.session, stmt, params_list)
        ann_results = []
        for (success, result) in results:
            if not success:
                raise Exception('Failed to execute ANN query')
            ann_results.append(self.process_ann_rows(result))
        return ann_results

    def query_chunks(self, chunk_ids: list[Any]) -> list[torch.Tensor]:
        if self.verbose:
            print(f'Loading embeddings from {len(chunk_ids)} chunks for full ColBERT scoring')
        flattened_pks = [pk if isinstance(pk, tuple) else (pk,) for pk in chunk_ids]
        stmt = self.get_query_chunks_stmt()
        results = execute_concurrent_with_args(self.session, stmt, flattened_pks)
        chunk_results = []
        for (success, result) in results:
            if not success:
                raise Exception('Failed to execute chunk query')
            chunk_results.append(torch.stack(self.process_chunk_rows(result)))
        return chunk_results

    def _connect_local(self):
        reconnection_policy = ExponentialReconnectionPolicy(base_delay=1, max_delay=60)
        self.cluster = Cluster(reconnection_policy=reconnection_policy)
        try:
            self.session = self.cluster.connect()
        except NoHostAvailable:
            raise ConnectionError('ASTRA_DB_TOKEN and ASTRA_DB_ID not set but Cassandra is not running locally')

    def _connect_astra(self, token: str, db_id: str):
        scb_path = _get_secure_connect_bundle(token, db_id, self.verbose)
        cloud_config = {'secure_connect_bundle': scb_path}
        auth_provider = PlainTextAuthProvider('token', token)
        reconnection_policy = ExponentialReconnectionPolicy(base_delay=1, max_delay=60)
        self.cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider, reconnection_policy=reconnection_policy)
        self.session = self.cluster.connect()
        if self.verbose:
            print(f'Connected to Astra db {db_id}')

    def _maybe_create_keyspace(self, db_id, token):
        if token:
            url = f'https://api.astra.datastax.com/v2/databases/{db_id}/keyspaces/{self.keyspace}'
            headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
            data = json.dumps({'name': self.keyspace}).encode('utf-8')
            req = urllib.request.Request(url, method='POST', headers=headers, data=data)
            try:
                with urllib.request.urlopen(req) as response:
                    if response.status == 201:
                        if self.verbose:
                            print(f"Keyspace '{self.keyspace}' created or verified")
                        start_time = time.time()
                        while time.time() - start_time < 10:
                            try:
                                self.session.execute(f'USE {self.keyspace}')
                                break
                            except BaseException:
                                time.sleep(0.1)
                        else:
                            raise Exception(f"Keyspace '{self.keyspace}' creation successful, but still unavailable after 10 seconds")
                    else:
                        raise Exception(f'Failed to create keyspace: {response.read().decode()}')
            except urllib.error.HTTPError as e:
                raise Exception(f'Failed to create keyspace: {e.read().decode()}')
        else:
            create_keyspace_query = f"\n            CREATE KEYSPACE IF NOT EXISTS {self.keyspace}\n            WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}}\n            "
            self.session.execute(create_keyspace_query)
            if self.verbose:
                print(f"Keyspace '{self.keyspace}' created or verified")

```

Note how AstraCQL gives extension points for subclasses to adapt ColbertLive for their schema, but AstraCQL itself
handles best practices like dispatching queries concurrently.

Here is an example of how AstraCQL is used.  This is given purely to inform your writing of a new DB subclass,
not to incorporate into the subclass directly.  In particular, this example adds other convenience methods typical
of a DAL that are not part of the AstraCQL API.

```
import uuid
from typing import List, Any, Tuple

import torch
from cassandra.concurrent import execute_concurrent_with_args
from cassandra.query import PreparedStatement

from colbert_live.db.astra import AstraCQL
from cassandra.cluster import ResultSet

from colbert_live.db.sqlite import Sqlite3DB
import sqlite3
import numpy as np


class CmdlineAstraDB(AstraCQL):
    def __init__(self, keyspace: str, embedding_dim: int, astra_db_id: str, astra_token: str):
        super().__init__(keyspace, embedding_dim, astra_db_id, astra_token, verbose=True)

    def prepare(self, embedding_dim: int):
        # Create tables asynchronously
        futures = []

        # for simplicity, we don't actually have a records table, but if we
        # wanted to add things like title, creation date, etc., that's where it would go

        # Create pages table
        futures.append(self.session.execute_async(f"""
            CREATE TABLE IF NOT EXISTS {self.keyspace}.pages (
                record_id uuid,
                num int,
                body blob,
                PRIMARY KEY (record_id, num)
            )
        """))

        # Create page_embeddings table
        futures.append(self.session.execute_async(f"""
            CREATE TABLE IF NOT EXISTS {self.keyspace}.page_embeddings (
                record_id uuid,
                page_num int,
                embedding_id int,
                tags set<text>,
                embedding vector<float, {embedding_dim}>,
                PRIMARY KEY (record_id, page_num, embedding_id)
            )
        """))

        # Wait for all CREATE TABLE operations to complete
        for future in futures:
            future.result()

        # Create colbert_ann index
        i1 = self.session.execute_async(f"""
            CREATE CUSTOM INDEX IF NOT EXISTS colbert_ann 
            ON {self.keyspace}.page_embeddings(embedding) 
            USING 'StorageAttachedIndex'
            WITH OPTIONS = {{ 'source_model': 'bert' }}
        """)
        i2 = self.session.execute_async(f"""
            CREATE CUSTOM INDEX IF NOT EXISTS colbert_tags 
            ON {self.keyspace}.page_embeddings(tags) 
            USING 'StorageAttachedIndex'
        """)
        index_futures = [i1, i2]

        # Prepare statements
        self.insert_page_stmt = self.session.prepare(f"""
            INSERT INTO {self.keyspace}.pages (record_id, num, body) VALUES (?, ?, ?)
        """)
        self.insert_embedding_stmt = self.session.prepare(f"""
            INSERT INTO {self.keyspace}.page_embeddings (record_id, page_num, embedding_id, embedding, tags) VALUES (?, ?, ?, ?, ?)
        """)

        [index_future.result() for index_future in index_futures]
        self.query_ann_stmt = self.session.prepare(f"""
            SELECT record_id, page_num, similarity_cosine(embedding, ?) AS similarity
            FROM {self.keyspace}.page_embeddings
            ORDER BY embedding ANN OF ?
            LIMIT ?
        """)
        self.query_ann_with_tag_stmt = self.session.prepare(f"""
            SELECT record_id, page_num, similarity_cosine(embedding, ?) AS similarity
            FROM {self.keyspace}.page_embeddings
            WHERE tags CONTAINS ?
            ORDER BY embedding ANN OF ?
            LIMIT ?
        """)
        self.query_chunks_stmt = self.session.prepare(f"""
            SELECT embedding FROM {self.keyspace}.page_embeddings WHERE record_id = ? AND page_num = ?
        """)

        print("Schema ready")

    def add_record(self, pages: list[bytes], embeddings: list[torch.Tensor], tags: set[str] = set()):
        record_id = uuid.uuid4()
        L = [(record_id, num, body) for num, body in enumerate(pages, start=1)]
        execute_concurrent_with_args(self.session, self.insert_page_stmt, L)

        L = [(record_id, page_num, embedding_id, embedding, tags)
             for page_num in range(1, len(embeddings) + 1)
             for embedding_id, embedding in enumerate(embeddings[page_num - 1])]
        execute_concurrent_with_args(self.session, self.insert_embedding_stmt, L)

        return record_id

    def get_query_ann(self, embeddings: torch.Tensor, limit: int, params: dict[str, Any]) -> tuple[PreparedStatement, list[tuple]]:
        tag = params.get('tag')
        if tag:
            params_list = [(emb, tag, emb, limit) for emb in embeddings.tolist()]
            return self.query_ann_with_tag_stmt, params_list
        else:
            params_list = [(emb, emb, limit) for emb in embeddings.tolist()]
            return self.query_ann_stmt, params_list

    def get_query_chunks_stmt(self) -> PreparedStatement:
        return self.query_chunks_stmt

    def process_ann_rows(self, result: ResultSet) -> List[tuple[Any, float]]:
        return [((row.record_id, row.page_num), row.similarity) for row in result]

    def process_chunk_rows(self, result: ResultSet) -> List[torch.Tensor]:
        return [torch.tensor(row.embedding) for row in result]

    def get_page_body(self, chunk_pk: tuple) -> bytes:
        record_id, page_num = chunk_pk
        query = f"SELECT body FROM {self.keyspace}.pages WHERE record_id = %s AND num = %s"
        result = self.session.execute(query, (record_id, page_num))
        return result.one()[0]


class CmdlineSqlite3DB(Sqlite3DB):
    def __init__(self, db_path: str, embedding_dim: int):
        super().__init__(db_path, embedding_dim, verbose=True)

    def prepare(self, embedding_dim: int):
        self.cursor.executescript(f"""
            CREATE TABLE IF NOT EXISTS pages (
                record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                num INTEGER,
                body BLOB,
                UNIQUE (record_id, num)
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS page_embedding_vectors USING vec0(
                embedding FLOAT[{embedding_dim}]
            );

            CREATE TABLE IF NOT EXISTS page_embeddings (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id INTEGER,
                page_num INTEGER,
                embedding_id INTEGER,
                page_embedding_vector_rowid INTEGER,
                FOREIGN KEY (record_id, page_num) REFERENCES pages(record_id, num),
                FOREIGN KEY (page_embedding_vector_rowid) REFERENCES page_embedding_vectors(rowid)
            );
        """)

        # insert statements
        self.insert_page_stmt = "INSERT INTO pages (num, body) VALUES (?, ?) RETURNING record_id"
        self.insert_embedding_stmt = """
            INSERT INTO page_embeddings (record_id, page_num, embedding_id, page_embedding_vector_rowid)
            VALUES (?, ?, ?, ?)
        """
        self.insert_embedding_vector_stmt = "INSERT INTO page_embedding_vectors (embedding) VALUES (vec_f32(?))"

        # queries
        self.query_ann_stmt = f"""
            SELECT record_id, page_num, distance
            FROM (
                SELECT rowid, distance
                FROM page_embedding_vectors
                WHERE embedding MATCH ?
                ORDER BY distance
                LIMIT ?
            ) AS pev
            JOIN page_embeddings pe ON pe.page_embedding_vector_rowid = pev.rowid
        """
        self.query_chunks_stmt = """
            SELECT pev.embedding
            FROM page_embeddings pe
            JOIN page_embedding_vectors pev ON pe.page_embedding_vector_rowid = pev.rowid
            WHERE pe.record_id = ? AND pe.page_num = ?
        """

        print("Schema ready")

    def add_record(self, pages: list[bytes], embeddings: list[torch.Tensor]) -> int:
        assert pages
        for page_num, (page, page_embeddings) in enumerate(zip(pages, embeddings), start=1):
            # Insert page
            self.cursor.execute(self.insert_page_stmt, (page_num, page))
            record_id = self.cursor.fetchone()[0]

            # Insert embeddings for this page
            for embedding_id, embedding in enumerate(page_embeddings):
                # Insert the embedding vector
                self.cursor.execute(self.insert_embedding_vector_stmt, (embedding.cpu().numpy().tobytes(),))
                embedding_vector_rowid = self.cursor.lastrowid

                # Insert the embedding metadata
                self.cursor.execute(self.insert_embedding_stmt, 
                    (record_id, page_num, embedding_id, embedding_vector_rowid))

        self.conn.commit()
        return record_id

    def get_query_ann(self, embeddings: torch.Tensor, limit: int, params: dict[str, Any]) -> Tuple[str, List[Any]]:
        params_list = [(emb.cpu().numpy().tobytes(), limit) for emb in embeddings]
        return self.query_ann_stmt, params_list

    def get_query_chunks_stmt(self) -> str:
        return self.query_chunks_stmt

    def process_ann_rows(self, result: List[sqlite3.Row]) -> List[Tuple[Any, float]]:
        return [((row[0], row[1]), row[2]) for row in result]

    def process_chunk_rows(self, result: List[sqlite3.Row]) -> List[torch.Tensor]:
        return [torch.from_numpy(np.frombuffer(row[0], dtype=np.float32)) for row in result]

    def get_page_body(self, chunk_pk: tuple) -> bytes:
        record_id, page_num = chunk_pk
        self.cursor.execute("SELECT body FROM pages WHERE record_id = ? AND num = ?", (record_id, page_num))
        result = self.cursor.fetchone()
        return result[0]

```
