import os
import sys
import time
import json
from typing import Dict, Tuple, List
from collections import defaultdict

from openai import OpenAI
import google.generativeai as gemini
import tiktoken
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from more_itertools import chunked

from cassandra.cluster import Cluster, Session
from cassandra.auth import PlainTextAuthProvider
from colbert_live.db.astra import execute_concurrent_async

LOTTE_DATA_PATH = os.path.expanduser("~/datasets/lotte")

# Set up OpenAI client
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# Set up Gemini client
gemini.configure(api_key=(os.environ.get("GEMINI_API_KEY")))

# Global variables to store the chosen embedding provider and vector size
EMBEDDING_PROVIDER = None
VECTOR_SIZE = None
STELLA_MODEL = None

truncated_passages = 0

class LotteDPRDB:
    def __init__(self, keyspace: str, embedding_provider: str, vector_size: int):
        self.keyspace = f"{keyspace}_{embedding_provider}"
        self.vector_size = vector_size
        self.session = self._connect_astra()
        self._create_schema()

    def _connect_astra(self) -> Session:
        cluster = Cluster()
        return cluster.connect()

    def _create_schema(self):
        self.session.execute(f"""
            CREATE KEYSPACE IF NOT EXISTS {self.keyspace}
            WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}}
        """)
        self.session.execute(f"USE {self.keyspace}")
        self.session.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id text PRIMARY KEY,
                text text,
                embedding vector<float, %s>
            )
        """, [self.vector_size])
        self.session.execute(f"""
            CREATE CUSTOM INDEX IF NOT EXISTS ON documents (embedding) 
            USING 'StorageAttachedIndex'
        """)
        self.insert_stmt = self.session.prepare("""
            INSERT INTO documents (id, text, embedding)
            VALUES (?, ?, ?)
        """)
        self.search_stmt = self.session.prepare("""
            SELECT id, text, similarity_cosine(embedding, ?) AS similarity
            FROM documents
            ORDER BY embedding ANN OF ?
            LIMIT ?
        """)

    def insert_documents(self, documents: List[Tuple[str, str, List[float]]]):
        futures = execute_concurrent_async(self.session, [(self.insert_stmt, doc) for doc in documents])
        return futures

    def search(self, query_embedding: List[float], k: int) -> List[Tuple[str, float]]:
        rows = self.session.execute(self.search_stmt, (query_embedding, query_embedding, k))
        return [(row.id, row.similarity) for row in rows]

def load_dataset(dataset: str, split: str) -> Tuple[Dict, Dict, Dict]:
    print(f"Loading dataset {dataset} ({split} split)...")
    data_path = os.path.join(LOTTE_DATA_PATH, dataset, split)
    
    corpus = {}
    with open(os.path.join(data_path, "collection.tsv"), "r") as f:
        for line in f:
            pid, text = line.strip().split("\t")
            corpus[pid] = {"text": text}

    queries = {}
    qrels = defaultdict(dict)
    for query_type in ["search", "forum"]:
        with open(os.path.join(data_path, f"questions.{query_type}.tsv"), "r") as f:
            for line in f:
                qid, text = line.strip().split("\t")
                queries[f"{query_type}_{qid}"] = text

        with open(os.path.join(data_path, f"qas.{query_type}.jsonl"), "r") as f:
            for line in f:
                data = json.loads(line)
                qid = f"{query_type}_{data['qid']}"
                for pid in data['answer_pids']:
                    qrels[qid][pid] = 1

    print(f"Dataset loaded. Corpus size: {len(corpus)}, Queries: {len(queries)}, Relevance judgments: {len(qrels)}")
    return corpus, queries, qrels

def get_embeddings(texts: List[str], is_query: bool = False) -> List[List[float]]:
    if EMBEDDING_PROVIDER == "openai":
        tiktoken_model = tiktoken.encoding_for_model('text-embedding-3-small')
        def tokenize(text: str) -> List[int]:
            return tiktoken_model.encode(text, disallowed_special=())
        def token_length(text: str) -> int:
            return len(list(tokenize(text)))
        def truncate_to(text, max_tokens):
            truncated_tokens = list(tokenize(text))[:max_tokens]
            truncated_s = tiktoken_model.decode(truncated_tokens)
            return truncated_s
        truncated_texts = []
        for text in texts:
            if token_length(text) > 8000:
                global truncated_passages
                truncated_passages += 1
                text = truncate_to(text, 8000)
            truncated_texts.append(text)
        response = openai_client.embeddings.create(
            input=truncated_texts,
            model="text-embedding-3-small"
        )
        return [data.embedding for data in response.data]
    elif EMBEDDING_PROVIDER == "gemini":
        model = "models/text-embedding-004"
        result = gemini.embed_content(model=model, content=texts)
        time.sleep(1.0) # crude rate limit
        return result['embedding']
    elif EMBEDDING_PROVIDER == "stella":
        global STELLA_MODEL
        if STELLA_MODEL is None:
            STELLA_MODEL = SentenceTransformer("dunzhang/stella_en_1.5B_v5", trust_remote_code=True).cuda()
        if is_query:
            return STELLA_MODEL.encode(texts, prompt_name="s2p_query").tolist()
        else:
            return STELLA_MODEL.encode(texts).tolist()
    else:
        raise ValueError(f"Invalid embedding provider: {EMBEDDING_PROVIDER}")

def is_populated(db):
    result = db.session.execute(f"SELECT * FROM {db.keyspace}.documents LIMIT 1")
    return result.one() is not None

def compute_and_store_embeddings(corpus: Dict, db: LotteDPRDB):
    if is_populated(db):
        print("Database not empty, assuming embeddings already computed and stored.")
        return
    print("Computing and storing document embeddings...")
    start_time = time.time()
    
    batch_size = 20
    future = None
    for doc_batch in tqdm(chunked(corpus.items(), batch_size), total=len(corpus)//batch_size + 1, desc="Processing"):
        doc_ids, doc_data = zip(*doc_batch)
        doc_texts = [data['text'] for data in doc_data]
        embeddings = get_embeddings(doc_texts, is_query=False)
        
        documents = list(zip(doc_ids, doc_texts, embeddings))
        if future:
            future.result()
        future = db.insert_documents(documents)

    end_time = time.time()
    print(f"Embedding computation and storage completed. Time taken: {end_time - start_time:.2f} seconds")

def search_and_benchmark(queries: Dict, db: LotteDPRDB, k: int = 5) -> Dict[str, Dict[str, float]]:
    def search_batch(query_batch: List[Tuple[str, str]]) -> List[Tuple[str, Dict[str, float]]]:
        query_ids, query_texts = zip(*query_batch)
        query_embeddings = get_embeddings(query_texts, is_query=True)
        
        results = []
        for query_id, query_embedding in zip(query_ids, query_embeddings):
            search_results = db.search(query_embedding, k)
            results.append((query_id, dict(search_results)))
        return results

    start_time = time.time()
    batch_size = 20
    results = {}
    
    with tqdm(total=len(queries), desc="Retrieving") as pbar:
        for query_batch in chunked(queries.items(), batch_size):
            batch_results = search_batch(query_batch)
            results.update(batch_results)
            pbar.update(len(query_batch))

    end_time = time.time()
    print(f"  Time: {end_time - start_time:.2f} seconds = {len(queries) / (end_time - start_time):.2f} QPS")
    return results

def write_rankings(results: Dict[str, Dict[str, float]], output_file: str):
    with open(output_file, "w") as f:
        for qid, doc_scores in results.items():
            for rank, (pid, score) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True), start=1):
                qid_bare = qid.split("_")[1]
                f.write(f"{qid_bare}\t{pid}\t{rank}\t{score}\n")

def evaluate_lotte(dataset: str, split: str, query_type: str):
    ks_name = f"lotte_{dataset.replace('-', '')}"

    db = LotteDPRDB(ks_name, EMBEDDING_PROVIDER, VECTOR_SIZE)

    corpus, all_queries, all_qrels = load_dataset(dataset, split)
    compute_and_store_embeddings(corpus, db)

    queries = {qid: query for qid, query in all_queries.items() if qid.startswith(f"{query_type}_")}

    print(f"Evaluating {dataset} ({query_type} queries)")
    results = search_and_benchmark(queries, db)

    output_dir = f"lotte_rankings/{split}]/{EMBEDDING_PROVIDER}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{dataset}.{query_type}.ranking.tsv")
    write_rankings(results, output_file)

    print(f"Rankings written to {output_file}")

def main(embedding_provider, datasets):
    global EMBEDDING_PROVIDER, VECTOR_SIZE, truncated_passages
    EMBEDDING_PROVIDER = embedding_provider
    if embedding_provider == "openai":
        VECTOR_SIZE = 1536
    elif embedding_provider == "gemini":
        VECTOR_SIZE = 768
    elif embedding_provider == "stella":
        VECTOR_SIZE = 1024
    else:
        raise ValueError(f"Invalid embedding provider: {embedding_provider}")
    for dataset in datasets:
        truncated_passages = 0
        for query_type in ["search", "forum"]:
            evaluate_lotte(dataset, "test", query_type)
        print(f"Truncated passages: {truncated_passages} in {dataset}")

if __name__ == "__main__":
    all_datasets = ["writing", "recreation", "lifestyle"]

    if len(sys.argv) < 2:
        print("Usage: python lotte-dpr.py <embedding_provider> [dataset1 dataset2 ...]")
        print("Embedding provider must be either 'openai' or 'gemini'")
        sys.exit(1)

    embedding_provider = sys.argv[1].lower()
    if embedding_provider not in ["openai", "gemini", "stella"]:
        print("Invalid embedding provider. Must be 'openai', 'gemini', or 'stella'")
        sys.exit(1)

    if len(sys.argv) > 2:
        datasets_to_run = [d for d in sys.argv[2:] if d in all_datasets]
        unrecognized_datasets = [d for d in sys.argv[2:] if d not in all_datasets]
        if unrecognized_datasets:
            print(f"Skipping unrecognized datasets: {', '.join(unrecognized_datasets)}")
    else:
        datasets_to_run = all_datasets

    print(f"Using {embedding_provider} for embeddings")
    print(f"Evaluating datasets: {', '.join(datasets_to_run)}")
    main(embedding_provider, datasets_to_run)
