from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2") 

class LocalEmbeddings:
    def embed_query(self, text):
        return model.encode(text).tolist()
    
    def embed_documents(self, texts):
        return model.encode(texts).tolist()

def get_embeddings_model():
    return LocalEmbeddings()

if __name__ == "__main__":
    m = get_embeddings_model()
    result = m.embed_query("Test")
    print(f"Uspesno! Vektor je duzine: {len(result)}")