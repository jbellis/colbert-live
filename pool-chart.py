import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read data from stdin
data = pd.read_csv(sys.stdin, names=['dataset', 'doc_pool', 'query_pool', 'ndcg'])

# Prepare data for plotting
datasets = data['dataset'].unique()
conditions = ['baseline', 'docs', 'queries', 'docs+queries']

results = {dataset: {condition: 0 for condition in conditions} for dataset in datasets}

for _, row in data.iterrows():
    dataset = row['dataset']
    doc_pool = row['doc_pool']
    query_pool = row['query_pool']
    ndcg = row['ndcg']
    
    if doc_pool == 1 and query_pool == 0.0:
        results[dataset]['baseline'] = ndcg
    elif doc_pool == 2 and query_pool == 0.0:
        results[dataset]['docs'] = ndcg
    elif doc_pool == 1 and query_pool == 0.03:
        results[dataset]['queries'] = ndcg
    elif doc_pool == 2 and query_pool == 0.03:
        results[dataset]['docs+queries'] = ndcg

# Normalize values
for dataset in datasets:
    baseline = results[dataset]['baseline']
    for condition in conditions:
        results[dataset][condition] /= baseline

# Set up the plot (doubled size)
fig, ax = plt.subplots(figsize=(24, 12))

# Set the width of each bar and the positions of the bars
bar_width = 0.2
r1 = range(len(datasets))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]

# Create the bars
ax.bar(r1, [results[dataset]['baseline'] for dataset in datasets], color='blue', width=bar_width, label='baseline')
ax.bar(r2, [results[dataset]['docs'] for dataset in datasets], color='green', width=bar_width, label='docs')
ax.bar(r3, [results[dataset]['queries'] for dataset in datasets], color='red', width=bar_width, label='queries')
ax.bar(r4, [results[dataset]['docs+queries'] for dataset in datasets], color='purple', width=bar_width, label='docs+queries')

# Customize the plot
ax.set_ylabel('Normalized NDCG', fontsize=16)
ax.set_title('Normalized NDCG by Dataset and Condition (Baseline = 1.0)', fontsize=20)
ax.set_xticks([r + bar_width*1.5 for r in range(len(datasets))])
ax.set_xticklabels(datasets, fontsize=14)
ax.legend(fontsize=14)

# Increase font size for tick labels
ax.tick_params(axis='both', which='major', labelsize=14)

# Add a horizontal line at y=1.0 to represent the baseline
ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.0)

# Set y-axis to start from 0
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.show()
