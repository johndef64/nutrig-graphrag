#%%
import os
import logging
import ollama
import numpy as np
from openai import AsyncOpenAI
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs
from sentence_transformers import SentenceTransformer

# from biomedical.llm_utils import *

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)


######################################################

# Set WORKING_DIR
# os.chdir(r"/root/projects/nano-graphrag/biomedical")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())

# WORKING_DIR = "./nano_graphrag_cache_groq_biomed_TEST_300halftext_LLAMA4_BioPrompts_biobert"
# WORKING_DIR = "./nano_graphrag_cache_groq_biomed_TEST_200Results_LLAMA4_BioPrompts_biobert"
WORKING_DIR = "./cache_groqLLAMA4scout_biobert_bioprompt_20Results_TEST"
WORKING_DIR = "./cache_groqLLAMA4scout_openaiembed_bioprompt_20Results_TEST"
WORKING_DIR = "./dummy_cache_groqllama4_openaiemb"  # For testing purposes, use a dummy cache directory


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
os.environ['MODEL'] =  GROQ_MODELS[0]  # <===== Change this to select a different model
# os.environ['MODEL'] =  OLLAMA_MODELS[0] 

print(f"Using model: {os.environ['MODEL']}")

######################################################

# import llm_utils after MODEL Selection !
from biomedical.llm_utils import *

##### Choose the embedding model #####
BERT_MODELS = ["dmis-lab/biobert-v1.1",
               "all-MiniLM-L6-v2",
               "all-mpnet-base-v2"
               ]
OPENAI_EMBEDDER = "text-embedding-3-small"  
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"

embedder = openai_embedding
# embedder = ollama_embedding
# embedder = local_embedding


# ---> Change Default Embedding model in biomedical/llm_utils.py

######################################################

# Run Script
MODEL = os.environ['MODEL']
if MODEL in GROQ_MODELS.values():
    USE_LLM = groq_model_if_cache
elif MODEL in OLLAMA_MODELS.values():
    USE_LLM = ollama_model_if_cache
elif MODEL in DEEP_MODELS.values(): 
    USE_LLM = deepseepk_model_if_cache
else: 
    raise ValueError(f"Model {MODEL} is not recognized. Please choose a valid model from GROQ_MODELS or OLLAMA_MODELS.")

#%%

def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)

def query(question):
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=USE_LLM,
        cheap_model_func=USE_LLM,
        # embedding_func=local_embedding,
    )
    print(
        rag.query(
            question, param=QueryParam(mode="global")
        )
    )

def query_naive(question):
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=USE_LLM,
        cheap_model_func=USE_LLM,
        enable_naive_rag =True,
        #embedding_func=embedder,
    )
    print(
        rag.query(
            question, param=QueryParam(mode="naive")
        )
    )


question = "What are the top themes in this story?"
question = "What is the function of the protein encoded by the gene CDK2?"
question = "With what the gene CTLA4 is associated with?"
question = "Tell me hypertension-associated Mutations and genes"
question = "What is SOD and what are his main relationships in nutrition?"
# question = "tell me to what condition are Genetic variants in DLG5 associated"
question="What gene is associated with rs45500793 and what disease?"

#%%

########## RUN THE Queries ##########
print("\n\n<<<<<<<<<<<<< GraphRAG Answer >>>>>>>>>>>>>>>")
if __name__ == "__main__":
    query(question)

print("<<< ----------------- >>>")

#%% Naive RAG Query
print("\n\n<<<<<<<<<<<<< Naive-RAG Answer >>>>>>>>>>>>>>>")
if __name__ == "__main__":
    query_naive(question)

print("<<< ----------------- >>>")

#%%
from groq import Groq

if __name__ == "__main__":
    print("\n\n<<<<<<<<<<<<< No-RAG Answer >>>>>>>>>>>>>>>")
    GROQ_API_KEY = api_keys["groq"]
    MODEL = os.environ['MODEL'] 
    
    # Test Native LLM response
    client = Groq(api_key=GROQ_API_KEY)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": question},
        ],
        stream=False
    )

    print(response.choices[0].message.content)
    print("<<< ----------------- >>>")
# %%
