#%%
import os
import logging
import ollama
import numpy as np
import pandas as pd
import time
from openai import AsyncOpenAI
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs
from sentence_transformers import SentenceTransformer

from biomedical.llm_utils import *

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

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



# Set working Directory
project = "nutrig-graphrag"
model_name = os.environ['MODEL'].replace("/", "_").replace(":", "_")
WORKING_DIR = f"./{project}_{model_name}_{EMBEDDER}_1"  # For testing purposes, use a dummy cache directory

print(f"Working Directory: {WORKING_DIR}")



#%%


def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)


def insert(TEXT, ):
    from time import time
    #remove_if_exist(f"{WORKING_DIR}/vdb_entities.json")
    #remove_if_exist(f"{WORKING_DIR}/kv_store_full_docs.json")
    #remove_if_exist(f"{WORKING_DIR}/kv_store_text_chunks.json")
    #remove_if_exist(f"{WORKING_DIR}/kv_store_community_reports.json")
    #remove_if_exist(f"{WORKING_DIR}/graph_chunk_entity_relation.graphml")

    # rag = GraphRAG(
    #    working_dir=WORKING_DIR,
    #    enable_llm_cache=True,
    #    best_model_func=USE_LLM,
    #    cheap_model_func=USE_LLM,
    #    embedding_func=embedder,
    # )
    rag = NutrigGraphRAG(GraphRAG,
        working_dir=WORKING_DIR,
        llm_model=os.environ['MODEL'],
        embedding_model=EMBEDDER,
        )
    start = time()
    rag.insert(TEXT)
    print("indexing time:", time() - start)


##### LOAD DATASET #####

# df = pd.read_csv("datasets/fulltext_dataset.zip")
df = pd.read_csv("datasets/working_dataset_100.csv")

# Filter Dataset
# df.text = df["RESULTS"] + "\n\n\n\n" + df["DISCUSS"]
df.text = df["RESULTS"]
df.text = df.text.str.replace("<SEP>","\n\n")
# df.to_csv("datasets/halftext_dataset5000.csv", index=False)


#%%

########## RUN THE JOB ##########
start_id = 51
batch_size = 1


t1 = time.time() 

if __name__ == "__main__":
    #insert(TEXT)
    #insert(df.text[1])
    #insert("""  """)
    
    for i in range(start_id, start_id+batch_size):
        if len(df.text[i]) > 10: 
            # print(f"""Inserting text of len {len(df.text[i])}""")
            insert(df.text[i])
            print(f"""\n\n<<<<<<<<<  {i+1}/{batch_size+start_id}  >>>>>>>>>\n\n """)

            print("Sleeping...")
            time.sleep(10)
        
        else:
            print("\n\n\n\n\nNo text...\n\n\n\n\n")
        
        
    
print(f"\033[94mBatch runtime: [{round(time.time() - t1, 2)}s]\033[0m")


from biomedical.graphml_visualize import CreateGraphVisualization

CreateGraphVisualization(WORKING_DIR.split("/")[-1])



#%%

# Nota: Inserire Risultati e Discussione insieme fa bloccare il batch 

# InternalServerError: Error code: 503 - {'error': {'message': 'Service Unavailable', 'type': 'internal_server_error'}}

# Limite Grow : Error code: 429 - {'error': {'message': 'Rate limit reached for model `meta-llama/llama-4-scout-17b-16e-instruct` in organization `org_01jsh6rb0jfm88d0qyz3b7zr1k` service tier `on_demand` on tokens per minute (TPM): Limit 300000, Used 320646, Requested 3149. Please try again in 4.759s. Need more tokens? Visit https://groq.com/self-serve-support/ to request higher limits.', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
    
#%%
