#%%
import os
import json
import logging
import ollama
import numpy as np
from openai import AsyncOpenAI
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs
from sentence_transformers import SentenceTransformer

###### set parameters ########
# GROQ SETTING
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

# DEEPSEEK SETTING
DEEP_MODELS = {
    0: "deepseek-chat"
}

# OLLAMA SETTING
# !!! qwen2-7B maybe produce unparsable results and cause the extraction of graph to fail.
OLLAMA_MODELS = {
    0: "deepseek-v2",
    1: "gemma2",
    2: "gemma2:27b",
    3: "qwen2:7b",
    4: "llama3.1:8b",
    5: "qwen2.5:14b"
}

# Set MODEL if not set in environment variables
if not os.environ['MODEL']:
    os.environ['MODEL'] = GROQ_MODELS[7]  # <===== Change this to select a different model
    print(f"Using default model: {os.environ['MODEL']}")


# HUGGINGFACE SETTING
BERT_MODELS = {
    0: "dmis-lab/biobert-v1.1",
    1: "all-MiniLM-L6-v2",
    2: "all-mpnet-base-v2"
}
BERT_MODEL = BERT_MODELS[2]  # "all-mpnet-base-v2"

# OpenAI embedding model
OPENAI_EMBEDDER = "text-embedding-3-small"  

# Assumed embedding model settings
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_EMBEDDING_MODEL_DIM = 768
OLLAMA_EMBEDDING_MODEL_MAX_TOKENS = 8192

# Ollama Configuration
IP_ADDRESS = "0.0.0.0"
PORT = 11434
# https://ollama.com/library?sort=newest



###########################################



def load_json_from_lib(nome_file, local = False):
    # Usa __file__ per ottenere il percorso della directory del file corrente
    if not local:
        file_path = os.path.join(os.path.dirname(__file__), nome_file)
    else:
        file_path = nome_file
    with open(file_path, 'r') as file:
        return json.load(file)

def load_api_keys():
    root = os.path.dirname(os.path.abspath(__file__))
    # if not api_keys.json in cwd, save it in pkg dir
    print(os.path.join(root, "api_keys.json"))

    if not os.path.exists(os.path.join(root, "api_keys.json")):
        file_path = os.path.join(os.path.dirname(__file__), "api_keys.json")
        if os.path.exists(file_path):
            # load api keys from pkg
            api_keys = load_json_from_lib("api_keys.json")
        else:
            """
            Please, provide API keys to the system running function:
            in the file "api_keys.json"
            
            """
            api_keys = {
                "openai":   "miss",
                "deepseek": "miss",
                #"grok":     "miss",
                #"gemini":   "miss",
                #"aimlapi":   "miss",
                "groq":   "miss",
                "huggingface": "miss",
            }
    else:
        # if api_keys.json in cwd, take them from here
        api_keys = load_json_from_lib("api_keys.json", local=True)

    return api_keys


api_keys = load_api_keys()


DEEPSEEK_API_KEY = api_keys["deepseek"]
GROQ_API_KEY = api_keys["groq"]
os.environ['OPENAI_API_KEY'] = api_keys["openai"]

if BERT_MODEL == "dmis-lab/biobert-v1.1":
    from huggingface_hub import login
    login(api_keys["huggingface"])



def connect_ollama_server(ip_address: str, port: int = 11434):
    base_url = f"http://{ip_address}:{str(port)}"
    print(f"Ollama base url {base_url}")
    api_key = "ollama"
    #ollama_client = ollama.AsyncClient(
    ollama_client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key
    )

    return ollama_client

OLLAMA_CLIENT = connect_ollama_server(ip_address=IP_ADDRESS, port = PORT)




######## LLM API CALLS ########

MODEL = os.environ['MODEL']  # Get the model from environment variables

async def deepseepk_model_if_cache(
    prompt, system_prompt=None, history_messages=[],  **kwargs
) -> str:
    openai_async_client = AsyncOpenAI(
        api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com"
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------

    response = await openai_async_client.chat.completions.create(
        model=MODEL, messages=messages, **kwargs
    )

    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": MODEL}}
        )
    # -----------------------------------------------------
    return response.choices[0].message.content



async def groq_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI(
        api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1"
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------

    response = await openai_async_client.chat.completions.create(
        model=MODEL, messages=messages, **kwargs
    )

    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": MODEL}}
        )
    # -----------------------------------------------------
    return response.choices[0].message.content



async def ollama_model_server_if_cache(
    prompt, system_prompt=None, history_messages=[],  **kwargs
) -> str:
    # remove kwargs that are not supported by ollama
    kwargs.pop("max_tokens", None)
    kwargs.pop("response_format", None)

    #ollama_client = ollama.AsyncClient()
    ollama_client = OLLAMA_CLIENT 
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------
    response = await ollama_client.chat(model=MODEL, messages=messages, **kwargs)

    result = response["message"]["content"]
    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": result, "model": MODEL}})
    # -----------------------------------------------------
    return result


async def ollama_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # remove kwargs that are not supported by ollama
    kwargs.pop("max_tokens", None)
    kwargs.pop("response_format", None)

    ollama_client = ollama.AsyncClient()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------
    response = await ollama_client.chat(model=MODEL, messages=messages, **kwargs)

    result = response["message"]["content"]
    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": result, "model": MODEL}})
    # -----------------------------------------------------
    return result



####### Embedders #######

@wrap_embedding_func_with_attrs(
    embedding_dim=1536,
    max_token_size=8191,
)
async def openai_embedding(texts: list[str]) -> np.ndarray:
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    # Get embeddings from OpenAI API
    response = await client.embeddings.create(
        model=OPENAI_EMBEDDER,  # You can choose different models
        input=texts
    )
    # Extract embeddings from response
    embeddings = [resp.embedding for resp in response.data]
    # Convert to numpy array
    return np.array(embeddings, dtype=np.float32)


# We're using Ollama to generate embeddings for the BGE model
@wrap_embedding_func_with_attrs(
    embedding_dim=OLLAMA_EMBEDDING_MODEL_DIM,
    max_token_size=OLLAMA_EMBEDDING_MODEL_MAX_TOKENS,
)
async def ollama_embedding(texts: list[str]) -> np.ndarray:
    embed_text = []
    for text in texts:
        data = ollama.embeddings(model=OLLAMA_EMBEDDING_MODEL, prompt=text)
        embed_text.append(data["embedding"])

    return embed_text

# os.mkdir(".cache_huggingface", exist_ok=True)
EMBED_MODEL = SentenceTransformer(
    BERT_MODEL, cache_folder= ".cache_huggingface", device="cpu"
)

# We're using Sentence Transformers to generate embeddings for the BGE model
@wrap_embedding_func_with_attrs(
    embedding_dim=EMBED_MODEL.get_sentence_embedding_dimension(),
    max_token_size=EMBED_MODEL.max_seq_length,
)
async def local_embedding(texts: list[str]) -> np.ndarray:
    return EMBED_MODEL.encode(texts, normalize_embeddings=True)
# %%
