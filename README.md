# Gemma 2B RAG PDF Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions from PDF documents using:

- **PyPDFLoader** for PDF ingestion
- **RecursiveCharacterTextSplitter** for chunking
- **all-MiniLM-L6-v2** for embeddings
- **ChromaDB** for vector storage and similarity retrieval
- **Gemma 2B** through Ollama for answer generation
- **Streamlit** for the chatbot interface
- **RAGAS** for evaluation

## Architecture

```text
PDF documents
     ↓
PDF Loader
     ↓
Text Chunking
     ↓
MiniLM Embeddings
     ↓
ChromaDB
     ↓
User Question
     ↓
Query Embedding
     ↓
Similarity Retrieval (Top 5)
     ↓
Top 1 Context
     ↓
Gemma 2B
     ↓
Final Answer
```

This project is **standard RAG**. It does **not** implement HyDE because the query is embedded directly; there is no hypothetical-answer generation step before retrieval.

## Project Structure

```text
.
├── app.py
├── embedding.py
├── ingestion.py
├── text_chunker.py
├── vectorstore.py
├── rag_pipeline.py
├── run_ingestion.py
├── evaluate_ragas.py
├── eval_dataset.json
├── ragas_eval_results.csv
├── requirements.txt
├── .gitignore
├── README.md
└── data/
    └── compressor1.pdf
```

`chroma_db/` is intentionally not included. It is generated locally by the ingestion step.

## Requirements

- Python 3.11+ recommended
- Ollama installed and available on your PATH
- Gemma 2B available locally
- Internet access is required the first time `all-MiniLM-L6-v2` is downloaded by Sentence Transformers.

Pull the model used by the chatbot:

```bash
ollama pull gemma:2b
```

For the RAGAS evaluation script, the default judge model is `llama3`:

```bash
ollama pull llama3
```

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Then:

```bash
pip install -r requirements.txt
```

## Add PDFs

Put the PDFs you want the chatbot to use inside:

```text
data/
```

## Build the Vector Database

Run ingestion before starting the chatbot:

```bash
python run_ingestion.py
```

This loads PDFs, splits them into chunks, creates MiniLM embeddings, and creates the local ChromaDB knowledge base.

## Run the Chatbot

```bash
streamlit run app.py
```

Open the Streamlit URL shown in the terminal and enter a question about the indexed PDF.

## RAGAS Evaluation

The evaluation dataset is stored in `eval_dataset.json`.

Run:

```bash
python evaluate_ragas.py
```

The results are written to:

```text
ragas_eval_results.csv
```

## Notes

- `chroma_db/` is generated and ignored by Git.
- Do not commit API keys, passwords, `.env` files, or confidential PDFs.
- If `compressor1.pdf` is confidential or belongs to an organization, remove it from the repository before making the repository public. Users can instead place their own PDF files in `data/`.
