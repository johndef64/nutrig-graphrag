
#%%
import os

# Imposta una variabile d'ambiente in Linux prima di eseguire questo script
# export MY_VAR="il_mio_valore"
with open("openai-key.txt") as f:
    os.environ['OPENAI_API_KEY'] = f.read().strip()

try:
    my_var = os.environ['OPENAI_API_KEY']
    print(f"Il valore della variabile d'ambiente è: {my_var}")
except KeyError:
    print("La variabile d'ambiente OPENAI_API_KEY= non è stata impostata.")
# Error fix!
# https://stackoverflow.com/questions/46827007/error-runtimeerror-this-event-loop-is-already-running-in-python

#%%
from nano_graphrag import GraphRAG, QueryParam
from biomedical.llm_utils import NutrigGraphRAG

graph_func = NutrigGraphRAG(GraphRAG, 
                            working_dir="./carrol_2", 
                            llm_model="openai/gpt-oss-20b",)
book = "./book.txt"
# book = "./j-r-r-tolkien-lord-of-the-rings-01-the-fellowship-of-the-ring-retail-pdf.txt"
with open(book) as f:
    print(f.read().split(".")[1])
    graph_func.insert(f.read())


#%%
with open(book) as f:
    print(f.read())
#%%
len(open(book).read().split("."))
open(book).read().split(".")[100]


#%%

from nano_graphrag import GraphRAG, QueryParam

graph_func = GraphRAG(working_dir="./talkien_cache_deepseek_TEST")

#%%
# Perform global graphrag search
query = "What are the top themes in this story?"
query = "What is the relationship between Aragorn and Arwen?"

print(graph_func.query(query))
#%%
# Perform local graphrag search (I think is better and more scalable one)
print(graph_func.query("What are the top themes in this story?", param=QueryParam(mode="local")))
# %%


WORKING_DIR = "./talkien_cache_deepseek_TEST"

# Perform global graphrag search
query = "What are the top themes in this story?"
query = "What is the relationship between Aragorn and Arwen?"
def query():
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=deepseepk_model_if_cache,
        cheap_model_func=deepseepk_model_if_cache,
    )
    print(
        rag.query(
            query, param=QueryParam(mode="global")
        )
    )