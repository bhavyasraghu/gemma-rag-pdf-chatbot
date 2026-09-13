import os
from langchain_community.document_loaders import PyPDFLoader


class DocumentLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_documents(self):
        documents = []

        for file in os.listdir(self.data_path):
            if file.endswith(".pdf"):
                file_path = os.path.join(self.data_path, file)
                loader = PyPDFLoader(file_path)
                docs = loader.load()

                for doc in docs:
                    doc.metadata["source"] = file

                documents.extend(docs)

        return documents
