import torch
from datasets import load_dataset

from embedder import query
from embedder import texts
from sentence_transformers.util import semantic_search


faqs_embeddings = load_dataset('lukablaskovic/dataset')
dataset_embeddings = torch.from_numpy(
    faqs_embeddings["train"].to_pandas().to_numpy()).to(torch.float)

question = ["How can Medicare help me?"]
output = query(question)

query_embeddings = torch.FloatTensor(output)

print("query_embeddings: ", query_embeddings)
# Find top 5
hits = semantic_search(query_embeddings, dataset_embeddings, top_k=5)

# Extract
print([texts[hits[0][i]['corpus_id']] for i in range(len(hits[0]))])


print(hits)
