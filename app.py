import streamlit as st
import pandas as pd

from embedding import EmbeddingModel
from vectorstore import VectorStoreManager
from rag_pipeline import RAGPipeline



# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="RAG PDF Chatbot",
    layout="wide"
)



st.title("🤖 RAG PDF Chatbot")



# ---------------- LOAD COMPONENTS ----------------

embedding_model = EmbeddingModel()

vector_store = VectorStoreManager()


rag = RAGPipeline(
    vector_store,
    embedding_model
)



# ---------------- INPUT FORM ----------------

with st.form(
    "query_form",
    clear_on_submit=False
):

    query = st.text_input(
        "Enter your question"
    )


    submitted = st.form_submit_button(
        "Search"
    )



# ---------------- PROCESS QUERY ----------------

if submitted and query:


    with st.spinner(
        "Retrieving relevant chunks..."
    ):


        answer, docs, metadata, scores = rag.generate_response(query)



        # ---------------- TABLE DATA ----------------


        table_data = []


        for i in range(len(docs)):


            table_data.append({

                "Rank": i + 1,


                "Chunk": docs[i][:200] + "...",


                "Page No": metadata[i].get(
                    "page",
                    "N/A"
                ),


                "Similarity %": scores[i]

            })



        df = pd.DataFrame(table_data)



        # Remove pandas index completely

        df.index = [""] * len(df)



        st.subheader(
            "📊 Top 5 Retrieved Chunks"
        )



        st.dataframe(

            df,

            hide_index=True,

            use_container_width=True

        )




    # ---------------- GEMMA ANSWER ----------------


    with st.spinner(
        "Generating answer using Gemma 2B..."
    ):


        st.subheader(
            "🤖 Gemma 2B Answer"
        )


        st.success(
            answer
        )