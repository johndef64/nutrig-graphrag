#%%
from litellm import dataclass
from nutrig_graphrag.nano_graphrag import GraphRAG, QueryParam
from nutrig_graphrag.biomedical.llm_utils import NutrigGraphRAG

# Initialize
ngrag = NutrigGraphRAG(GraphRAG,
    working_dir="test_cache",
    llm_model="gemma2-9b-it",
    embedding_model="all-MiniLM-L6-v2",
)

# Ingest documents
for doc in ["data/pubmed_1.txt", "data/pubmed_2.txt"]:
    with open(doc) as f:
        ngrag.insert(f.read())

# Query knowledge graph
print(ngrag.query(
    "How does the MTHFR C677T variant affect folate metabolism?",
    param=QueryParam(mode="global")
))
# %%

from typing import Literal
from dataclasses import dataclass

@dataclass
class MyParam:
    mode: Literal["local", "global", "naive"] = "global"
    print_context: bool = False

def test(param: MyParam):
    if param.print_context:
        print(f"Mode is {param.mode} and print_context is {param.print_context}")
    else:
        print(f"Mode is {param.mode} and print_context is False")

param = MyParam(mode="local", print_context=True)
print(param)    

test(param)
# %%

