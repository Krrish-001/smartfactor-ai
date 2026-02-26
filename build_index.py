from sentence_transformers import SentenceTransformer
import faiss
import os
import numpy as np

model = SentenceTransformer("BAAI/bge-large-en")

documents = []
for file in os.listdir("../data/sample_docs"):
    with open(f"../data/sample_docs/{file}", "r") as f:
        documents.append(f.read())

embeddings = model.encode(documents)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

faiss.write_index(index, "faiss_index.bin")
