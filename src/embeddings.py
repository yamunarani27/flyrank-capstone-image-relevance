from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str) -> list[float]:
    """Return the embedding vector for a piece of text."""
    embedding = model.encode(text)
    return embedding.tolist()

