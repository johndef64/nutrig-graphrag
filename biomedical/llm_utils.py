#%%
import os
import json
import logging
import ollama
import numpy as np
from openai import AsyncOpenAI
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs
from sentence_transformers import SentenceTransformer



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


# GROQ SETTING
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
# MODEL = "llama-3.1-8b-instant"

# DEEPSEEK SETTING
DEEP_MODEL = "deepseek-chat"

# OLLAMA SETTING
# !!! qwen2-7B maybe produce unparsable results and cause the extraction of graph to fail.
OLLAMA_MODEL = "qwen2"
OLLAMA_MODEL = "gemma2" # solo per test



IP_ADDRESS = "0.0.0.0"
PORT = 11434
# https://ollama.com/library?sort=newest


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
# OLLAMA_CLIENT = OpenAI(
#     base_url="https://jxbwpvkneg5alo-11434.proxy.runpod.net/",
#     api_key="ollama",
# )

if False:
    response = OLLAMA_CLIENT.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello World!"},
        ],
    )
    print(response.choices[0].message.content)


######## LLM API CALLS ########


async def deepseepk_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
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
        args_hash = compute_args_hash(DEEP_MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------

    response = await openai_async_client.chat.completions.create(
        model=DEEP_MODEL, messages=messages, **kwargs
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
        args_hash = compute_args_hash(GROQ_MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------

    response = await openai_async_client.chat.completions.create(
        model=GROQ_MODEL, messages=messages, **kwargs
    )

    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": GROQ_MODEL}}
        )
    # -----------------------------------------------------
    return response.choices[0].message.content



async def ollama_model_server_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
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
        args_hash = compute_args_hash(OLLAMA_MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------
    response = await ollama_client.chat(model=OLLAMA_MODEL, messages=messages, **kwargs)

    result = response["message"]["content"]
    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": result, "model": OLLAMA_MODEL}})
    # -----------------------------------------------------
    return result

if False:
    dd = ollama_model_if_cache("Hello world")
    print(dd)



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
        args_hash = compute_args_hash(OLLAMA_MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------
    response = await ollama_client.chat(model=OLLAMA_MODEL, messages=messages, **kwargs)

    result = response["message"]["content"]
    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": result, "model": OLLAMA_MODEL}})
    # -----------------------------------------------------
    return result
