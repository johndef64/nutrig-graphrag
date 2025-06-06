#%%
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
