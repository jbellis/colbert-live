import sys
import re

def extract_data(input_text):
    # Regular expressions to match the required fields
    dataset_re = r'^([\w-]+)\s*@'
    doc_pooling_re = r'keyspace \w+aai[vV]\d(?:pool(\d))?'
    query_pool_distance_re = r'query pool distance ([\d.]+)'
    ndcg_10_re = r'NDCG@10: ([\d.]+)'

    # Extract data
    dataset = re.search(dataset_re, input_text, re.MULTILINE)
    doc_pooling = re.search(doc_pooling_re, input_text)
    query_pool_distance = re.search(query_pool_distance_re, input_text)
    ndcg_10 = re.search(ndcg_10_re, input_text)

    # Return extracted data if all fields are found
    if all([dataset, doc_pooling, query_pool_distance, ndcg_10]):
        # If pool number is not found, default to 1
        pool_num = doc_pooling.group(1) if doc_pooling.group(1) else '1'
        return f"{dataset.group(1)},{pool_num},{query_pool_distance.group(1)},{ndcg_10.group(1)}"
    return None

# Read all input at once
all_input = sys.stdin.read()

# Split input into entries
entries = re.split(r'([\w-]+\s*@\s*\d+\s*TPQ)', all_input)[1:]  # [1:] to skip the first empty element

# Process entries in pairs (entry start + entry content)
for i in range(0, len(entries), 2):
    if i+1 < len(entries):
        entry = entries[i] + entries[i+1]
        result = extract_data(entry)
        if result:
            print(result)
