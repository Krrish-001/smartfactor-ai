from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def hallucination_score(answer, context):
    emb1 = model.encode(answer)
    emb2 = model.encode(context)
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1)*np.linalg.norm(emb2))
    return similarity
