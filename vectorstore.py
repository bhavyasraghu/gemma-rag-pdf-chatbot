import chromadb


class VectorStoreManager:
    def __init__(self, persist_dir="./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection("knowledge_base")

    def create_store(self, documents, embedding_model):
        texts = [doc.page_content for doc in documents]
        metadata = [doc.metadata for doc in documents]
        embeddings = embedding_model.embed_documents(texts)

        # Start fresh when rebuilding the local knowledge base.
        if self.collection.count() > 0:
            self.client.delete_collection("knowledge_base")
            self.collection = self.client.get_or_create_collection("knowledge_base")

        for i, text in enumerate(texts):
            self.collection.add(
                ids=[str(i)],
                documents=[text],
                metadatas=[metadata[i]],
                embeddings=[embeddings[i]]
            )

    def query(self, query_embedding, k=3):
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
