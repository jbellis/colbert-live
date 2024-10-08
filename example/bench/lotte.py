import os
import sys
import time
import json
import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from typing import Dict, Tuple, List
from collections import defaultdict

from tqdm import tqdm
from more_itertools import chunked

from colbert_live import ColbertLive
from colbert_live.db.astra import execute_concurrent_async
from colbert_live.models import ColbertModel
from .db import BenchDB

LOTTE_DATA_PATH = os.path.expanduser("~/datasets/lotte")

def load_corpus(dataset: str, split: str, batch_size: int = 32):
    print(f"Loading corpus for dataset {dataset} ({split} split)...")
    data_path = os.path.join(LOTTE_DATA_PATH, dataset, split)
    
    batch = []
    with open(os.path.join(data_path, "collection.tsv"), "r") as f:
        for line in f:
            pid, text = line.strip().split("\t")
            batch.append((pid, text))
            if len(batch) == batch_size:
                yield batch
                batch = []
    
    if batch:  # Yield any remaining items
        yield batch

def load_queries(dataset: str, split: str, query_type: str) -> Dict:
    print(f"Loading {query_type} queries for dataset {dataset} ({split} split)...")
    data_path = os.path.join(LOTTE_DATA_PATH, dataset, split)
    
    queries = {}
    with open(os.path.join(data_path, f"questions.{query_type}.tsv"), "r") as f:
        for line in f:
            qid, text = line.strip().split("\t")
            queries[f"{query_type}_{qid}"] = text

    print(f"Queries loaded. Count: {len(queries)}")
    return queries

def process_document_batch(batch: List[Tuple[str, str]], db, colbert_live):
    chunk_raw_data = []
    chunk_texts = []
    insert_data = []

    for chunk_id, content in batch:
        chunk_texts.append(content)
        chunk_raw_data.append((chunk_id, "", content))
    
    embeddings = colbert_live.encode_chunks(chunk_texts)
    for (chunk_id, _, _), embedding_tensor in zip(chunk_raw_data, embeddings):
        insert_data.extend((chunk_id, i, e) for i, e in enumerate(embedding_tensor))

    chunk_stmt_with_parms = list(zip(cycle([db.insert_chunk_stmt]), chunk_raw_data))
    embeddings_stmt_with_parms = list(zip(cycle([db.insert_embeddings_stmt]), insert_data))
    return execute_concurrent_async(db.session, chunk_stmt_with_parms + embeddings_stmt_with_parms)

def is_populated(db):
    result = db.session.execute(f"SELECT * FROM {db.keyspace}.chunks LIMIT 1")
    return result.one() is not None

def compute_and_store_embeddings(corpus_generator, db, colbert_live, start_ordinal=0):
    if is_populated(db) and not start_ordinal:
        print("The chunks table is not empty. Skipping encoding and insertion.")
        return

    print("Encoding and inserting documents...")
    start_time = time.time()
    
    insert_future = None
    total_docs = 0
    for batch_num, doc_batch in tqdm(enumerate(corpus_generator, 1), desc="Encoding and inserting batches"):
        if (batch_num - 1) * len(doc_batch) < start_ordinal:
            continue
        next_insert_future = process_document_batch(doc_batch, db, colbert_live)
        if insert_future:
            insert_future.result()
        insert_future = next_insert_future
        total_docs += len(doc_batch)

    if insert_future:
        insert_future.result()

    end_time = time.time()
    print(f"Encoding and insertion completed. Total batches: {batch_num}, Total documents: {total_docs}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")

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

def write_rankings(results: Dict[str, Dict[str, float]], output_file: str):
    with open(output_file, "w") as f:
        for qid, doc_scores in results.items():
            for rank, (pid, score) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True), start=1):
                qid_bare = qid.split("_")[1]
                f.write(f"{qid_bare}\t{pid}\t{rank}\t{score}\n")

def evaluate_lotte(dataset: str, split: str, query_type: str, start_ordinal: int = 0):
    model_name = 'answerdotai/answerai-colbert-small-v1'
    doc_pool = 2
    n_ann_docs = 240
    n_maxsim_candidates = 15
    tokens_per_query = 32

    ks_name = f"lotte_{dataset.replace('-', '')}_colbert_{doc_pool}"

    model = ColbertModel(model_name, tokens_per_query=tokens_per_query)
    db = BenchDB(ks_name, model.dim, os.environ.get('ASTRA_DB_ID'), os.environ.get('ASTRA_DB_TOKEN'))
    colbert_live = ColbertLive(db, model, doc_pool_factor=doc_pool)

    corpus_generator = load_corpus(dataset, split)
    compute_and_store_embeddings(corpus_generator, db, colbert_live, start_ordinal)

    queries = load_queries(dataset, split, query_type)

    print(f"Evaluating {dataset} ({query_type} queries) @ {tokens_per_query} TPQ")
    results = search_and_benchmark(queries, n_ann_docs, n_maxsim_candidates, colbert_live)

    output_dir = f"lotte_rankings/{split}/colbert"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{dataset}.{query_type}.ranking.tsv")
    write_rankings(results, output_file)

    print(f"Rankings written to {output_file}")

def main(datasets, start_ordinal):
    for dataset in datasets:
        for query_type in ["search", "forum"]:
            evaluate_lotte(dataset, "test", query_type, start_ordinal)

if __name__ == "__main__":
    all_datasets = ["writing", "recreation", "science", "technology", "lifestyle"]

    parser = argparse.ArgumentParser(description="Evaluate LOTTE datasets")
    parser.add_argument("datasets", nargs="*", choices=all_datasets, default=all_datasets,
                        help="Datasets to evaluate (default: all)")
    parser.add_argument("--start", type=int, default=0,
                        help="Corpus ordinal to start from (default: 0)")
    args = parser.parse_args()

    datasets_to_run = args.datasets
    start_ordinal = args.start

    print(f"Evaluating datasets: {', '.join(datasets_to_run)}")
    print(f"Starting from corpus ordinal: {start_ordinal}")
    main(datasets_to_run, start_ordinal)
