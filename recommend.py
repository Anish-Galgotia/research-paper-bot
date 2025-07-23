import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# Load corpus
def load_paper_corpus(csv_path="papers_db.csv"):
    df = pd.read_csv(csv_path)
    return df

# Embed corpus once
def embed_corpus(df, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    corpus_embeddings = model.encode(df['abstract'].tolist(), convert_to_tensor=True)
    return model, corpus_embeddings

# Recomment similar papers
def recommend_similar_papers(query_text, df, model, corpus_embeddings, top_k=3):
    query_embedding = model.encode(query_text, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)[0]
    recommendations = []
    for hit in hits:
        idx = hit['corpus_id']
        score = hit['score']
        recommendations.append((df.iloc[idx]['title'], df.iloc[idx]['abstract'], score))
    return recommendations 