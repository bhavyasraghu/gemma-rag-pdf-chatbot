from ollama import chat
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



class RAGPipeline:


    def __init__(self, vector_store, embedding_model):

        self.vector_store = vector_store

        self.embedding_model = embedding_model



    def retrieve(self, query, k=5):


        # -------------------------
        # 1. Create query embedding
        # -------------------------

        query_embedding = self.embedding_model.embed_query(query)



        # -------------------------
        # 2. Retrieve top k chunks
        # -------------------------

        results = self.vector_store.collection.query(

            query_embeddings=[query_embedding],

            n_results=k,

            include=[
                "documents",
                "metadatas"
            ]

        )



        docs = results["documents"][0]

        metadatas = results["metadatas"][0]



        # -------------------------
        # 3. Recreate embeddings
        # for retrieved chunks
        # -------------------------

        doc_embeddings = self.embedding_model.embed_documents(
            docs
        )



        # -------------------------
        # 4. Similarity calculation
        # -------------------------

        query_vec = np.array(
            query_embedding
        ).reshape(1,-1)



        doc_vecs = np.array(
            doc_embeddings
        )



        similarity = cosine_similarity(

            query_vec,

            doc_vecs

        )[0]



        similarity_scores = [

            round(score * 100, 2)

            for score in similarity

        ]



        # -------------------------
        # 5. Sort by similarity
        # -------------------------

        combined = list(zip(

            docs,

            metadatas,

            similarity_scores

        ))



        combined.sort(

            key=lambda x: x[2],

            reverse=True

        )



        docs, metadatas, scores = zip(*combined)



        return (

            list(docs),

            list(metadatas),

            list(scores)

        )





    def generate_response(self, query):


        # Retrieve chunks

        docs, metadatas, scores = self.retrieve(
            query
        )



        # -------------------------
        # Only TOP 1 chunk to Gemma
        # -------------------------

        top_context = docs[0]



        prompt = f"""

You are answering questions from a technical PDF document.

Answer the question using the retrieved context.
The context can come from a specification document and may describe a topic through its components, functions and requirements or as process, procedure, components, steps, guidelines or explanations.

Do not say information is missing if the context describes the topic indirectly.

If any type of information is given in the pdf u must answer using it.
Use your understanding to explain the information clearly.
Always give answer in more than one sentence.
Do not say conetext is not enough use whatever is available to answer.
Just provide the answer no need for saying "sure" or "here's the answer"


Context:

{top_context}


Question:

{query}


Answer:

"""



        response = chat(

            model="gemma:2b",

            messages=[

                {

                "role":"user",

                "content":prompt

                }

            ]

        )



        answer = response[

            "message"

        ][

            "content"

        ]



        return (

            answer,

            docs,

            metadatas,

            scores

        )