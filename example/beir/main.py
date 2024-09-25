import os
import time
import multiprocessing
from itertools import cycle
from typing import Dict, Tuple, List

from beir import util
from more_itertools import chunked
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from example.util import execute_concurrent_async
from tqdm import tqdm

from colbert_live import ColbertLive
from .db import AstraDBBeir
import time

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


def connect(db_params: Dict, model_name: str):
    max_retries = 5
    for attempt in range(max_retries):
        try:
            return AstraDBBeir(db_params['keyspace'], model_name, db_params['astra_db_id'], db_params['astra_token'])
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"Connection attempt {attempt + 1} failed. Retrying in 5 seconds...")
            time.sleep(5)

def is_populated(db):
    result = db.session.execute(f"SELECT * FROM {db.keyspace}.chunks LIMIT 1")
    return result.one() is not None

def process_document_range(start_idx: int, end_idx: int, range_items: List[Tuple[str, Dict]], db_params: Dict, model_name: str, doc_pool_factor: int):
    db = connect(db_params, model_name)
    colbert_live = ColbertLive(db, model_name, doc_pool_factor=doc_pool_factor)
    
    for doc_batch in chunked(range_items[start_idx:end_idx], 32):
        process_document_batch(doc_batch, db, colbert_live)

def compute_and_store_embeddings(corpus: dict, db, model_name: str, doc_pool_factor):
    if is_populated(db):
        print("The chunks table is not empty. Skipping encoding and insertion.")
        return

    print("Encoding and inserting documents...")
    start_time = time.time()

    num_processes = multiprocessing.cpu_count()
    corpus_items = list(corpus.items())
    total_items = len(corpus_items)
    items_per_process = total_items // num_processes
    remainder = total_items % num_processes

    db_params = {
        'keyspace': db.keyspace,
        'astra_db_id': os.environ.get('ASTRA_DB_ID'),
        'astra_token': os.environ.get('ASTRA_DB_TOKEN')
    }

    with multiprocessing.Pool(processes=num_processes) as pool:
        tasks = []
        start_idx = 0
        for i in range(num_processes):
            end_idx = start_idx + items_per_process + (1 if i < remainder else 0)
            tasks.append(pool.apply_async(process_document_range, (start_idx, end_idx, corpus_items, db_params, model_name, doc_pool_factor)))
            start_idx = end_idx
        
        for task in tqdm(tasks, total=num_processes, desc="Processing document ranges"):
            task.get()  # This will raise an exception if the task failed

    end_time = time.time()
    print(f"Encoding and insertion completed. Time taken: {end_time - start_time:.2f} seconds")


def search_range(start_idx: int, end_idx: int, query_items: List[Tuple[str, str]], db_params: Dict, model_name: str, n_ann_docs: int, n_colbert_candidates: int, query_pool_distance: float, tokens_per_query: int) -> Dict[str, Dict[str, float]]:
    db = connect(db_params, model_name)
    colbert_live = ColbertLive(db, model_name, query_pool_distance=query_pool_distance, tokens_per_query=tokens_per_query)
    
    results = {}
    for query_id, query in query_items[start_idx:end_idx]:
        results[query_id] = dict(colbert_live.search(query, n_colbert_candidates, n_ann_docs, n_colbert_candidates))
    return results

def search_and_benchmark(queries: dict, n_ann_docs: int, n_colbert_candidates: int, db: AstraDBBeir, model_name: str, query_pool_distance: float, tokens_per_query: int) -> Dict[str, Dict[str, float]]:
    start_time = time.time()

    num_processes = multiprocessing.cpu_count()
    query_items = list(queries.items())
    total_queries = len(query_items)
    queries_per_process = total_queries // num_processes
    remainder = total_queries % num_processes

    db_params = {
        'keyspace': db.keyspace,
        'astra_db_id': os.environ.get('ASTRA_DB_ID'),
        'astra_token': os.environ.get('ASTRA_DB_TOKEN')
    }

    with multiprocessing.Pool(processes=num_processes) as pool:
        tasks = []
        start_idx = 0
        for i in range(num_processes):
            end_idx = start_idx + queries_per_process + (1 if i < remainder else 0)
            tasks.append(pool.apply_async(search_range, (start_idx, end_idx, query_items, db_params, model_name, n_ann_docs, n_colbert_candidates, query_pool_distance, tokens_per_query)))
            start_idx = end_idx
        
        results = {}
        for task in tqdm(tasks, total=num_processes, desc="Retrieving"):
            results.update(task.get())

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
        ('fiqa', 32),
        ('scifact', 48),
        ('nfcorpus', 32),
        ('scidocs', 48),
        ('trec-covid', 48),
        ('arguana', 64),
        ('quora', 32),
        ('webis-touche2020', 32),
    ]:
        for doc_pool_factor in [1, 2, 3, 4]:
            model_name = 'answerdotai/answerai-colbert-small-v1'
            ks_name = dataset.replace('-', '') + 'aaiv1'
            if doc_pool_factor > 1:
                ks_name += f'pool{doc_pool_factor}'
            db_params = {
                'keyspace': ks_name,
                'astra_db_id': os.environ.get('ASTRA_DB_ID'),
                'astra_token': os.environ.get('ASTRA_DB_TOKEN')
            }
            db = connect(db_params, model_name)

            corpus, queries, qrels = download_and_load_dataset(dataset)
            compute_and_store_embeddings(corpus, db, model_name, doc_pool_factor)

            for query_pool_distance in [0.03]:
                for n_ann_docs in [120, 240, 360]:
                    for n_maxsim_candidates in [20, 40, 60, 80]:
                        print(f'{dataset} @ {tokens_per_query} TPQ from keyspace {ks_name}, query pool distance {query_pool_distance}, CL {n_ann_docs}:{n_maxsim_candidates}')

                        results = search_and_benchmark(queries, n_ann_docs, n_maxsim_candidates, db, model_name, query_pool_distance, tokens_per_query)
                        evaluation_results = evaluate_model(qrels, results)
                        for k, score in evaluation_results.items():
                            print(f"  {k}: {score:.5f}")


if __name__ == "__main__":
    test_all()
