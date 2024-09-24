import os
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from typing import Dict, Tuple, List

from beir import util
from more_itertools import chunked, divide
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from example.util import execute_concurrent_async
from tqdm import tqdm

from colbert_live import ColbertLive
from .db import AstraDBBeir

#
# Because we're importing `util` from the parent example module, you should run this script
# from the main colbert-live directory using module syntax:
# `python -m example.beir.main`
#

def download_and_load_dataset(dataset) -> Tuple[dict, dict, dict]:
    print("Downloading and loading dataset...")
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)
    corpus, all_queries, all_qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    # Limit queries to the first 100 items
    queries = all_queries # dict(list(all_queries.items())[:100])
    qrels = {qid: all_qrels[qid] for qid in queries.keys() if qid in all_qrels}

    print(f"Dataset loaded. Corpus size: {len(corpus)}, Queries: {len(queries)}, Relevance judgments: {len(qrels)}")
    return corpus, queries, qrels


def process_document_batch(batch: List[Tuple[str, Dict]], db, colbert_live):
    chunk_raw_data = []
    chunk_texts = []
    insert_data = []

    # flatten the batch dict into chunk_raw_data
    for chunk_id, chunk in batch:
        title = chunk['title']
        content = chunk['text']
        chunk_texts.append(content)
        chunk_raw_data.append((chunk_id, title, content))
    # Encode the batch to a 2d tensor of embeddings per chunk
    embeddings = colbert_live.encode_chunks(chunk_texts)
    # Combine the embeddings with their chunk information
    for (chunk_id, _, _), embedding_tensor in zip(chunk_raw_data, embeddings):
        insert_data.extend((chunk_id, i, e) for i, e in enumerate(embedding_tensor))

    # Insert chunks and embeddings
    chunk_stmt_with_parms = list(zip(cycle([db.insert_chunk_stmt]), chunk_raw_data))
    embeddings_stmt_with_parms = list(zip(cycle([db.insert_embeddings_stmt]), insert_data))
    return execute_concurrent_async(db.session, chunk_stmt_with_parms + embeddings_stmt_with_parms)


def is_populated(db):
    result = db.session.execute(f"SELECT * FROM {db.keyspace}.chunks LIMIT 1")
    return result.one() is not None

def process_document_range(start_idx: int, end_idx: int, range_items: List[Tuple[str, Dict]], db_params: Dict, model_name: str):
    db = AstraDBBeir(db_params['keyspace'], model_name, db_params['astra_db_id'], db_params['astra_token'])
    colbert_live = ColbertLive(db, model_name)
    
    for doc_batch in chunked(range_items, 32):
        process_document_batch(doc_batch, db, colbert_live)

def compute_and_store_embeddings(corpus: dict, db, colbert_live):
    if is_populated(db):
        print("The chunks table is not empty. Skipping encoding and insertion.")
        return

    print("Encoding and inserting documents...")
    start_time = time.time()

    num_processes = multiprocessing.cpu_count()
    corpus_items = list(corpus.items())
    ranges = list(divide(num_processes, corpus_items))

    db_params = {
        'keyspace': db.keyspace,
        'astra_db_id': os.environ.get('ASTRA_DB_ID'),
        'astra_token': os.environ.get('ASTRA_DB_TOKEN')
    }

    with multiprocessing.Pool(processes=num_processes) as pool:
        tasks = [
            pool.apply_async(process_document_range, (0, len(list(range_items)), list(range_items), db_params, colbert_live.model_name))
            for range_items in ranges
        ]
        for task in tqdm(tasks, total=num_processes, desc="Processing document ranges"):
            task.get()  # This will raise an exception if the task failed

    end_time = time.time()
    print(f"Encoding and insertion completed. Time taken: {end_time - start_time:.2f} seconds")


def search_and_benchmark(queries: dict, n_ann_docs: int, n_colbert_candidates: int, colbert_live: ColbertLive) -> Dict[str, Dict[str, float]]:
    def search(query_item: Tuple[str, str]) -> Tuple[str, Dict[str, float]]:
        query_id, query = query_item
        return (query_id, dict(colbert_live.search(query, n_colbert_candidates, n_ann_docs, n_colbert_candidates)))

    start_time = time.time()
    num_threads = 8
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = dict(tqdm(executor.map(search, queries.items()), total=len(queries), desc="Retrieving"))
    end_time = time.time()

    print(f"  Time: {end_time - start_time:.2f} seconds = {len(queries) / (end_time - start_time):.2f} QPS")
    return results


def evaluate_model(qrels: dict, results: dict):
    evaluator = EvaluateRetrieval()
    metrics = evaluator.evaluate(qrels, results, [1, 5, 10, 100])
    metric_names = ["NDCG"]
    evaluation_results = {}
    for metric_name, scores in zip(metric_names, metrics):
        for k, score in scores.items():
            evaluation_results[f"{k}"] = score
    return evaluation_results

def test_all():
    for dataset, tokens_per_query in [
        ('webis-touche2020', 32),
        ('scifact', 48),
        ('nfcorpus', 32),
        ('scidocs', 48),
        ('trec-covid', 48),
        ('fiqa', 32),
        ('arguana', 64),
        ('quora', 32)
    ]:
        for doc_pool_factor in [1, 2, 3, 4]:
            model_name = 'answerdotai/answerai-colbert-small-v1'
            ks_name = dataset.replace('-', '') + 'aaiv1'
            if doc_pool_factor > 1:
                ks_name += f'pool{doc_pool_factor}'
            db = AstraDBBeir(ks_name, model_name, os.environ.get('ASTRA_DB_ID'), os.environ.get('ASTRA_DB_TOKEN'))

            colbert_live = ColbertLive(db, model_name, doc_pool_factor=doc_pool_factor)
            corpus, queries, qrels = download_and_load_dataset(dataset)
            compute_and_store_embeddings(corpus, db, colbert_live)


if __name__ == "__main__":
    test_all()
