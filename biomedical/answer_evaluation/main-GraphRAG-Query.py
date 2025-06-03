#%%
import os
import logging
import ollama
import numpy as np
from openai import AsyncOpenAI
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs
from sentence_transformers import SentenceTransformer

from biomedical.llm_utils import *

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)


######################################################
# Set WORKING_DIR
# os.chdir(r"/root/projects/nano-graphrag/biomedical")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())



#### Choose the model to use for the RAG #####
OLLAMA_MODELS = {
    0: "deepseek-v2",
    1: "gemma2",
    2: "gemma2:27b",
    3: "qwen2:7b",
    4: "llama3.1:8b",
    5: "qwen2.5:14b"
}

GROQ_MODELS = {
    0: "gemma2-9b-it",
    1: "llama-3.3-70b-versatile",
    2: "llama-3.1-8b-instant",
    3: "llama-guard-3-8b",
    4: "llama3-70b-8192",
    5: "llama3-8b-8192",
    6: "deepseek-r1-distill-llama-70b",
    7: "meta-llama/llama-4-maverick-17b-128e-instruct",
    8: "meta-llama/llama-4-scout-17b-16e-instruct",
    9: "mistral-saba-24b",
    10: "qwen-qwq-32b"
}

DEEP_MODELS = {
    0: "deepseek-chat"
}

# Choose a model from the GROQ_MODELS dictionary
os.environ['MODEL'] =  GROQ_MODELS[8]  # <===== Change this to select a different model
# os.environ['MODEL'] =  OLLAMA_MODELS[0] 

print(f"Using model: {os.environ['MODEL']}")

######################################################


##### Choose the embedding model #####
BERT_MODELS = {
    0: "dmis-lab/biobert-v1.1",
    1: "all-MiniLM-L6-v2",
    2: "all-mpnet-base-v2"
}
OPENAI_EMBEDDER = "text-embedding-3-small"  
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"

EMBEDDER = BERT_MODELS[2]  # <===== Change this to select a different embedding model

######################################################

#%%

# Set working Directory
project = "nutrig-graphrag"
model_name = os.environ['MODEL'].replace("/", "_").replace(":", "_")
WORKING_DIR = f"./{project}_{model_name}_{EMBEDDER}_1"  
WORKING_DIR = "./ablation_study/cache_gemma2_dmis-lab_biobert-v1.1" 
WORKING_DIR = "/root/projects/nano-graphrag/biomedical/answer_evaluation/cache_gemma2_all-mpnet-base-v2"  # For testing purposes, use a dummy cache directory


print(f"Working Directory: {WORKING_DIR}")
print(f"LLM Model: {os.environ['MODEL']}")
print(f"Embedding Model: {EMBEDDER}")
#%%
######################################################


def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)

def query(question):
    # rag = GraphRAG(
    #     working_dir=WORKING_DIR,
    #     best_model_func=USE_LLM,
    #     cheap_model_func=USE_LLM,
    #     embedding_func=local_embedding,
    # )
    print(f"Usint {EMBEDDER} embedding model")
    print(f"Using {os.environ['MODEL']} model for LLM")
    rag = NutrigGraphRAG(GraphRAG,
        working_dir=WORKING_DIR,
        llm_model=os.environ['MODEL'],
        embedding_model=EMBEDDER,
        )
    response = rag.query(
            question, param=QueryParam(mode="global")
        )
    print(response)
    return response


def query_local(question):

    rag = NutrigGraphRAG(GraphRAG,
        working_dir=WORKING_DIR,
        llm_model=os.environ['MODEL'],
        embedding_model=EMBEDDER,
        )
    print(f"Usint {EMBEDDER} embedding model")
    print(f"Using {os.environ['MODEL']} model for LLM")

    response = rag.query(
            question, param=QueryParam(mode="local")
        )
    print(response)
    return response



def query_naive(question):
    # rag = GraphRAG(
    #     working_dir=WORKING_DIR,
    #     best_model_func=USE_LLM,
    #     cheap_model_func=USE_LLM,
    #     enable_naive_rag =True,
    #     embedding_func=embedder,
    # )
    print(f"Usint {EMBEDDER} embedding model")
    print(f"Using {os.environ['MODEL']} model for LLM")
    rag = NutrigGraphRAG(GraphRAG,
        working_dir=WORKING_DIR,
        llm_model=os.environ['MODEL'],
        embedding_model=EMBEDDER,
        enable_naive_rag = True,
        )

    response = rag.query(
            question, param=QueryParam(mode="naive", top_k=100 )
        )
    print(response)
    return response

# Get Questions
import pickle
root = os.path.dirname(os.path.abspath(__file__))
questions_file = os.path.join(root, "questions_cache_gemma2_all-mpnet-base-v2.pkl")
questions = pickle.load(open(questions_file, "rb"))

questions = questions[:10]

question = "What are the top themes in this story?"
question = "What is the function of the protein encoded by the gene CDK2?"
question = "With what the gene CTLA4 is associated with?"
question = "Tell me hypertension-associated Mutations and genes"
question = "What is SOD and what are his main relationships in nutrition?"
# question = "tell me to what condition are Genetic variants in DLG5 associated"
question="What gene is associated with rs45500793 and what disease?"

import pandas as pd
answer_df = pd.DataFrame(columns=["User","Task", "Question"])
n = 0

import time
time1 = time.time()
for n in range(5):
    user = questions.User[n]
    task = questions.Task[n]
    print(f"User: {user}, Task: {task}")

    for question in questions.Questions[n]:
        print(f"Question: {question}")

        ########## RUN THE Queries ##########
        
        print("\n\n<<<<<<<<<<<<< GraphRAG Answer >>>>>>>>>>>>>>>")
        graphrag_response = query(question)


        print("<<< ----------------- >>>")
        
        # Naive RAG Query
        print("\n\n<<<<<<<<<<<<< Naive-RAG Answer >>>>>>>>>>>>>>>")
        # naive_response = query_naive(question)
        naive_response = QueryNaiveGraphRAG(
                        question,llm_model=os.environ["MODEL"], 
                        working_dir=WORKING_DIR, k=20, threshold=0.2, 
                        BERT_MODEL= EMBEDDER
                        )
        

        print("<<< ----------------- >>>")

        from groq import Groq

        print("\n\n<<<<<<<<<<<<< No-RAG Answer >>>>>>>>>>>>>>>")
        GROQ_API_KEY = api_keys["groq"]
        MODEL = os.environ['MODEL'] 
        
        # Test Native LLM response
        client = Groq(api_key=GROQ_API_KEY)

        norag_response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": question},
            ],
            stream=False
        )
        norag_response = norag_response.choices[0].message.content
        print(norag_response)
        print("<<< ----------------- >>>")


        # Dati da aggiungere
        new_row = {
            "User": user,
            "Task": task,
            "Question": question,
            "GraphRAG-Answer": graphrag_response,
            "Naive-RAG-Answer": naive_response,
            "No-RAG-Answer": norag_response
        }

        # Crea un DataFrame con la nuova riga
        new_df = pd.DataFrame([new_row])

        # Concatena i DataFrame
        answer_df = pd.concat([answer_df, new_df], ignore_index=True)

print(f"Time taken for question: {time.time() - time1:.2f} seconds")
# %%
answer_df
# %%

# try naive
replyes = []
for n in range(1):
    user = questions.User[n]
    task = questions.Task[n]
    print(f"User: {user}, Task: {task}")
    for question in questions.Questions[n]:
        print(f"Question: {question}")

        ########## RUN THE Queries ##########

        print("<<< ----------------- >>>")
        
        # Naive RAG Query
        print("\n\n<<<<<<<<<<<<< Naive-RAG Answer >>>>>>>>>>>>>>>")
        # naive_response = query_naive(question)
        naive_response = QueryNaiveGraphRAG(
                question,llm_model=os.environ["MODEL"], 
                working_dir=WORKING_DIR, k=20, threshold=0.2, 
                BERT_MODEL= EMBEDDER
                )
        replyes.append(naive_response)
# %%
set(replyes)
    
# %%
