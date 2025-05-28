import ollama
from openai import AsyncOpenAI
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash

def connect_ollama_server(ip_address: str, port: int = 11434):
    base_url = f"http://{ip_address}:{str(port)}"
    print(f"Ollama base url {base_url}")
    api_key = "ollama"
    ollama_client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key
    )
    return ollama_client

def get_ollama_model_server_fun(ollama_client, llm_model):
    async def ollama_model_server_if_cache(
        prompt, system_prompt=None, history_messages=[], **kwargs
    ) -> str:
        kwargs.pop("max_tokens", None)
        kwargs.pop("response_format", None)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
        if hashing_kv is not None:
            args_hash = compute_args_hash(llm_model, messages)
            if_cache_return = await hashing_kv.get_by_id(args_hash)
            if if_cache_return is not None:
                return if_cache_return["return"]

        response = await ollama_client.chat(model=llm_model, messages=messages, **kwargs)
        result = response["message"]["content"]

        if hashing_kv is not None:
            await hashing_kv.upsert({args_hash: {"return": result, "model": llm_model}})

        return result

    return ollama_model_server_if_cache

def get_ollama_model_fun(ll_model):
    async def ollama_model_if_cache(
        prompt, system_prompt=None, history_messages=[], **kwargs
    ) -> str:
        kwargs.pop("max_tokens", None)
        kwargs.pop("response_format", None)

        ollama_client = ollama.AsyncClient()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        if hashing_kv is not None:
            args_hash = compute_args_hash(ll_model, messages)
            if_cache_return = await hashing_kv.get_by_id(args_hash)
            if if_cache_return is not None:
                return if_cache_return["return"]
        response = await ollama_client.chat(model=ll_model, messages=messages, **kwargs)

        result = response["message"]["content"]
        if hashing_kv is not None:
            await hashing_kv.upsert({args_hash: {"return": result, "model": ll_model}})
        return result
    return ollama_model_if_cache