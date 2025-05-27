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


from biomedical.llm_utils import *

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

# Set WORKING_DIR
# os.chdir(r"/root/projects/nano-graphrag/biomedical")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())

# WORKING_DIR = "./nano_graphrag_cache_groq_biomed_TEST_300halftext_LLAMA4_BioPrompts_biobert"
# WORKING_DIR = "./nano_graphrag_cache_groq_biomed_TEST_200Results_LLAMA4_BioPrompts_biobert"
WORKING_DIR = "./cache_groqLLAMA4scout_biobert_bioprompt_20Results_TEST"
WORKING_DIR = "./cache_groqLLAMA4scout_openaiembed_bioprompt_20Results_TEST"
WORKING_DIR = "./dummy_cache"  # For testing purposes, use a dummy cache directory


api_keys = load_api_keys()
# OPENAI EMBEDDER  [Non funziona]
os.environ['OPENAI_API_KEY'] = api_keys["openai"]
USING_OPENAI_EMBEDDER = True 


# HUGGINGFACE SETTING
#  "sentence-transformers/all-MiniLM-L6-v2", "dmis-lab/biobert-v1.1"
BERT_MODEL = "dmis-lab/biobert-v1.1"
BERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

if BERT_MODEL == "dmis-lab/biobert-v1.1":
    from huggingface_hub import login
    login(api_keys["huggingface"])


if not USING_OPENAI_EMBEDDER:
    EMBED_MODEL = SentenceTransformer(
        BERT_MODEL, cache_folder=WORKING_DIR, device="cpu"
    )

    # We're using Sentence Transformers to generate embeddings for the BGE model
    @wrap_embedding_func_with_attrs(
        embedding_dim=EMBED_MODEL.get_sentence_embedding_dimension(),
        max_token_size=EMBED_MODEL.max_seq_length,
    )
    async def local_embedding(texts: list[str]) -> np.ndarray:
        return EMBED_MODEL.encode(texts, normalize_embeddings=True)

else:

    @wrap_embedding_func_with_attrs(
        embedding_dim=1536,
        max_token_size=8191,
    )
    async def local_embedding(texts: list[str]) -> np.ndarray:
        client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        # Get embeddings from OpenAI API
        response = await client.embeddings.create(
            model="text-embedding-3-small",  # You can choose different models
            input=texts
        )
        # Extract embeddings from response
        embeddings = [resp.embedding for resp in response.data]
        # Convert to numpy array
        return np.array(embeddings, dtype=np.float32)
    


#%%

USE_LLM = groq_model_if_cache
# USE_LLM = ollama_model_if_cache
# USE_LLM = deepseepk_model_if_cache


def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)


def query(question):
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=USE_LLM,
        cheap_model_func=USE_LLM,
        embedding_func=local_embedding,
    )
    print(
        rag.query(
            question, param=QueryParam(mode="global")
        )
    )


# with open("datasets/abstract10.txt", encoding="utf-8-sig") as f:
#     TEXT = f.read()

def insert(TEXT):
    from time import time
    #remove_if_exist(f"{WORKING_DIR}/vdb_entities.json")
    #remove_if_exist(f"{WORKING_DIR}/kv_store_full_docs.json")
    #remove_if_exist(f"{WORKING_DIR}/kv_store_text_chunks.json")
    #remove_if_exist(f"{WORKING_DIR}/kv_store_community_reports.json")
    #remove_if_exist(f"{WORKING_DIR}/graph_chunk_entity_relation.graphml")

    rag = GraphRAG(
        working_dir=WORKING_DIR,
        enable_llm_cache=True,
        best_model_func=USE_LLM,
        cheap_model_func=USE_LLM,
        embedding_func=local_embedding,
    )
    start = time()
    rag.insert(TEXT)
    print("indexing time:", time() - start)


##### LOAD DATASET #####
import pandas as pd
# df = pd.read_csv("datasets/fulltext_dataset.zip")
df = pd.read_csv("datasets/chunks/0-3000pmc_fulltext.csv")

# Filter Dataset
df = df[~((df['INTRO'].isna() & df['METHODS'].isna() & df['RESULTS'].isna() & df['DISCUSS'].isna()))]
df = df[~((df['RESULTS'].isna() & df['DISCUSS'].isna()))].reset_index(drop=True)
df = df[~(df['RESULTS'].isna())].reset_index(drop=True)
df.fillna("", inplace=True)

df.text = df["RESULTS"] + "\n\n\n\n" + df["DISCUSS"]
df.text = df["RESULTS"]
df.text = df.text.str.replace("<SEP>","\n\n")
# df.to_csv("datasets/halftext_dataset5000.csv", index=False)

question = "What are the top themes in this story?"
question = "What is the function of the protein encoded by the gene CDK2?"
question = "With what the gene CTLA4 is associated with?"
question = "Tell me hypertension-associated Mutations and genes"
question = "What is SOD and what are his main relationships in nutrition?"
# question = "tell me to what condition are Genetic variants in DLG5 associated"

#%%

########## RUN THE JOB ##########
start_id = 0
batch_size = 1
import time

if __name__ == "__main__":
    #insert(TEXT)
    #insert(df.text[1])
    #insert("""  """)
    if True:
        for i in range(start_id, start_id+batch_size):
            if len(df.text[i]) > 10: 
                # print(f"""Inserting text of len {len(df.text[i])}""")
                insert(df.text[i])
                print(f"""\n\n<<<<<<<<<<<<<<<<<<<<<  {i+1}/{batch_size+start_id}  >>>>>>>>>>>>>>>>>>\n\n """)
            else:
                print("\n\n\n\n\nNo text...\n\n\n\n\n")
            
            print("Sleeping...")
            time.sleep(5)
    
    # query(question)


#%%
"""
Nota: Inserire Risultati e Discussione insieme fa bloccare il batch 

InternalServerError: Error code: 503 - {'error': {'message': 'Service Unavailable', 'type': 'internal_server_error'}}
"""
    
#%%
df.text

#%%
print("<<< ----------------- >>>")

#%%

if not __name__ == "__main__":
    GROQ_API_KEY = api_keys["groq"]
    MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
    # MODEL = "llama-3.1-8b-instant"

    # Test Native LLM response
    from groq import Groq

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
# %%
