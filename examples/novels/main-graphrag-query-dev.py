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
WORKING_DIR = f"./{project}_{model_name}_{EMBEDDER}_1"  # For testing purposes, use a dummy cache directory
WORKING_DIR = "./ablation_study/cache_gemma2_dmis-lab_biobert-v1.1"
# WORKING_DIR = "biomedical/nutrig-graphrag_meta-llama_llama-4-scout-17b-16e-instruct_all-mpnet-base-v2_1"

print(f"Working Directory: {WORKING_DIR}")
print(f"LLM Model: {os.environ['MODEL']}")
print(f"Embedding Model: {EMBEDDER}")

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


############
def query_naive_original(question):

    EMBED_MODEL = SentenceTransformer(
    EMBEDDER, cache_folder= ".cache_huggingface", device="cpu"
    )

    # We're using Sentence Transformers to generate embeddings for the BGE model
    @wrap_embedding_func_with_attrs(
        embedding_dim=EMBED_MODEL.get_sentence_embedding_dimension(),
        max_token_size=EMBED_MODEL.max_seq_length,
    )
    async def local_embedding(texts: list[str]) -> np.ndarray:
        return EMBED_MODEL.encode(texts, normalize_embeddings=True)

    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=groq_model_if_cache,
        cheap_model_func=groq_model_if_cache,
        enable_naive_rag =True,
        embedding_func=local_embedding
        )
    print(f"Usint {EMBEDDER} embedding model")
    print(f"Using {os.environ['MODEL']} model for LLM")

    response = rag.query(
            question, param=QueryParam(mode="naive")
        )
    print(response)
    return response
############


def query_naive(question):

    rag = NutrigGraphRAG(GraphRAG,
        working_dir=WORKING_DIR,
        llm_model=os.environ['MODEL'],
        embedding_model=EMBEDDER,
        enable_naive_rag = True,
        )
    print(f"Usint {EMBEDDER} embedding model")
    print(f"Using {os.environ['MODEL']} model for LLM")

    response = rag.query(
            question, param=QueryParam(mode="naive")
        )
    print(response)
    return response


# Get Questions
import pickle
questions = pickle.load(open("./question_generation/questions_cache_gemma2_all-mpnet-base-v2.pkl", "rb"))

question = "What are the top themes in this story?"
question = "What is the function of the protein encoded by the gene CDK2?"
question = "With what the gene CTLA4 is associated with?"
question = "Tell me hypertension-associated Mutations and genes"
question = "What is SOD and what are his main relationships in nutrition?"
# question = "tell me to what condition are Genetic variants in DLG5 associated"
question="What gene is associated with rs45500793 and what disease?"

question = questions.Questions[2][1]
# print(f"Question: {question}")
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
