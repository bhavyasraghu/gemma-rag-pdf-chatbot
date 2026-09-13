
import json
import warnings

warnings.filterwarnings("ignore")


from langchain_core.embeddings import Embeddings
from langchain_ollama import ChatOllama


from ragas import evaluate, EvaluationDataset
from ragas.run_config import RunConfig


from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)


from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper


from embedding import EmbeddingModel
from rag_pipeline import RAGPipeline
from vectorstore import VectorStoreManager



# ---------------- CONFIG ----------------

EVAL_DATASET_PATH = "eval_dataset.json"

OUTPUT_CSV_PATH = "ragas_eval_results.csv"


TOP_K_FOR_GENERATION = 1


JUDGE_MODEL = "llama3"



# ---------------- EMBEDDING ADAPTER ----------------


class LangchainEmbeddingAdapter(Embeddings):

    def __init__(self, embedding_model):

        self.embedding_model = embedding_model


    def embed_documents(self, texts):

        return self.embedding_model.embed_documents(texts)


    def embed_query(self, text):

        return self.embedding_model.embed_query(text)



# ---------------- LOAD QUESTIONS ----------------


def load_eval_questions(path):

    with open(path, "r", encoding="utf-8") as f:

        return json.load(f)



# ---------------- RUN RAG PIPELINE ----------------


def run_pipeline_on_questions(rag, questions):

    rows = []


    for item in questions:


        question = item["question"]


        ground_truth = item.get(
            "ground_truth",
            None
        )


        answer, docs, metadata, scores = (
            rag.generate_response(question)
        )



        rows.append(
            {

                "user_input": question,


                "response": answer,


                "generation_context":
                    list(
                        docs[:TOP_K_FOR_GENERATION]
                    ),



                "retrieval_contexts":
                    list(docs),



                "reference":
                    ground_truth

            }
        )


        print(
            "Done:",
            question[:60]
        )


    return rows





# ---------------- MAIN ----------------


def main():


    print("Loading RAG components...")


    embedding_model = EmbeddingModel()


    vector_store = VectorStoreManager()


    rag = RAGPipeline(
        vector_store,
        embedding_model
    )



    print(
        "Loading evaluation dataset..."
    )


    questions = load_eval_questions(
        EVAL_DATASET_PATH
    )



    print(
        "Running RAG pipeline..."
    )


    rows = run_pipeline_on_questions(
        rag,
        questions
    )



    print(
        "Loading RAGAS judge..."
    )


    judge_llm = LangchainLLMWrapper(

        ChatOllama(

            model=JUDGE_MODEL,

            temperature=0

        )

    )



    judge_embeddings = (

        LangchainEmbeddingsWrapper(

            LangchainEmbeddingAdapter(
                embedding_model
            )

        )

    )



    # -------- FIX FOR TIMEOUT --------


    run_config = RunConfig(

        max_workers=1,

        timeout=300

    )



    # -------- Generation evaluation --------


    generation_dataset = EvaluationDataset.from_list(

        [

            {

                "user_input":
                    r["user_input"],


                "response":
                    r["response"],


                "retrieved_contexts":
                    r["generation_context"]

            }

            for r in rows

        ]

    )



    print(
        "Evaluating answer quality..."
    )



    generation_result = evaluate(

        dataset=generation_dataset,


        metrics=[

            Faithfulness(),

            AnswerRelevancy()

        ],


        llm=judge_llm,


        embeddings=judge_embeddings,


        run_config=run_config

    )



    generation_df = (
        generation_result.to_pandas()
    )




    # -------- Retrieval evaluation --------


    retrieval_rows = [

        r for r in rows

        if r["reference"]

    ]



    retrieval_df = None



    if retrieval_rows:


        retrieval_dataset = EvaluationDataset.from_list(

            [

                {

                    "user_input":
                        r["user_input"],


                    "response":
                        r["response"],


                    "retrieved_contexts":
                        r["retrieval_contexts"],


                    "reference":
                        r["reference"]

                }


                for r in retrieval_rows

            ]

        )



        print(
            "Evaluating retrieval..."
        )



        retrieval_result = evaluate(

            dataset=retrieval_dataset,


            metrics=[

                ContextPrecision(),

                ContextRecall()

            ],


            llm=judge_llm,


            embeddings=judge_embeddings,


            run_config=run_config

        )



        retrieval_df = (
            retrieval_result.to_pandas()
        )



    # -------- REPORT --------


    print("\n====================")

    print("RAGAS SUMMARY")

    print("====================")



    print(

        generation_df[

            [

            "faithfulness",

            "answer_relevancy"

            ]

        ]

        .mean()

        .round(3)

    )



    if retrieval_df is not None:


        print(

            retrieval_df[

                [

                "context_precision",

                "context_recall"

                ]

            ]

            .mean()

            .round(3)

        )



    final_df = generation_df



    if retrieval_df is not None:


        final_df = final_df.merge(

            retrieval_df[

                [

                "user_input",

                "context_precision",

                "context_recall"

                ]

            ],


            on="user_input",


            how="left"

        )



    final_df.to_csv(

        OUTPUT_CSV_PATH,


        index=False

    )


    print(
        "Saved:",
        OUTPUT_CSV_PATH
    )





if __name__ == "__main__":

    main()