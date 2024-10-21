import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from typing import Dict, Tuple, List

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from more_itertools import chunked
from tqdm import tqdm

from colbert_live import ColbertLive
from colbert_live.db.astra import execute_concurrent_async
from colbert_live.models import ColbertModel
from .db import BenchDB


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

def compute_and_store_embeddings(corpus: dict, db, colbert_live):
    if is_populated(db):
        print("The chunks table is not empty. Skipping encoding and insertion.")
        return

    print("Encoding and inserting documents...")
    start_time = time.time()
    
    batch_size = 32
    insert_future = None
    for doc_batch in tqdm(chunked(corpus.items(), batch_size), total=len(corpus)//batch_size + 1, desc="Encoding and inserting"):
        next_insert_future = process_document_batch(doc_batch, db, colbert_live)
        if insert_future:
            insert_future.result() # wait for previous batch before inserting the next
        insert_future = next_insert_future

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

def test_all(datasets):
    for dataset, tokens_per_query in datasets:
        for doc_pool_factor in [2]:
            model_name = 'answerdotai/answerai-colbert-small-v1'
            ks_name = dataset.replace('-', '') + 'aaiv1'
            if doc_pool_factor > 1:
                ks_name += f'pool{doc_pool_factor}'

            model = ColbertModel(model_name, tokens_per_query=tokens_per_query)
            db = BenchDB(ks_name, model.dim, os.environ.get('ASTRA_DB_ID'), os.environ.get('ASTRA_DB_TOKEN'))
            colbert_live = ColbertLive(db, model, doc_pool_factor=doc_pool_factor)
            corpus, queries, qrels = download_and_load_dataset(dataset)
            compute_and_store_embeddings(corpus, db, colbert_live)

            for query_pool_distance in [0.03]:
                for n_ann_docs in [240]:
                    for n_maxsim_candidates in [20]:
                        print(f'{dataset} @ {tokens_per_query} TPQ from keyspace {ks_name}, query pool distance {query_pool_distance}, CL {n_ann_docs}:{n_maxsim_candidates}')

                        colbert_live = ColbertLive(db, model, query_pool_distance=query_pool_distance)
                        results = search_and_benchmark(queries, n_ann_docs, n_maxsim_candidates, colbert_live)
                        evaluation_results = evaluate_model(qrels, results)
                        for k, score in evaluation_results.items():
                            print(f"  {k}: {score:.5f}")


if __name__ == "__main__":
    all_datasets = [
        # "core" Q&A datasets that are a good fit for what colbert's designed for
        ('scifact', 48),
        ('trec-covid', 48),
        ('quora', 32),
        # other datasets
        # ('nfcorpus', 32),
        # ('scidocs', 48),
        # ('fiqa', 32),
        # ('arguana', 64),
    ]

    if len(sys.argv) > 1:
        dataset_dict = {name: tpq for name, tpq in all_datasets}
        requested_datasets = sys.argv[1:]
        datasets_to_run = [(name, dataset_dict[name]) for name in requested_datasets if name in dataset_dict]
        unrecognized_datasets = [name for name in requested_datasets if name not in dataset_dict]
        if unrecognized_datasets:
            print(f"Skipping unrecognized datasets: {', '.join(unrecognized_datasets)}")
    else:
        datasets_to_run = all_datasets

    print(f"Testing datasets: {', '.join(name for name, _ in datasets_to_run)}")
    test_all(datasets_to_run)
