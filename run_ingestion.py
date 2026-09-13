from ingestion import DocumentLoader
from text_chunker import TextChunker
from embedding import EmbeddingModel
from vectorstore import VectorStoreManager


# Load docs

loader = DocumentLoader("data/")

documents = loader.load_documents()


# Chunk

chunker = TextChunker()

chunks = chunker.split(documents)


# Embedding

embedding_model = EmbeddingModel()


# Store

vector_store = VectorStoreManager()

vector_store.create_store(chunks, embedding_model)


print("Ingestion complete with MiniLM embeddings")  