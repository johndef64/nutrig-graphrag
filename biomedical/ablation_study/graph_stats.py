#%%
import networkx as nx
import os
# Load graph from GraphML file
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())
import glob
import pprint
graphml_files = glob.glob('*')
folders = [f for f in glob.glob('*') if os.path.isdir(f)]

from gpt import C, pen, men
m= "make this a numbered dict:"
# C.cp(m,4)

folders = [f for f in folders if f.startswith('cache')]#
pprint.pprint(folders)
cache_list = [
    'cache_llama3.1_8b_all-mpnet-base-v2',
    'cache_gemma2_all-mpnet-base-v2',
    'cache_qwen2.5_14b_all-mpnet-base-v2',
    'cache_qwen2_7b_dmis-lab_biobert-v1.1',
    'cache_qwen2_7b_all-mpnet-base-v2',
    'cache_qwen2.5_14b_dmis-lab_biobert-v1.1',
    'cache_llama3.1_8b_dmis-lab_biobert-v1.1',
    'cache_gemma2_dmis-lab_biobert-v1.1'
]
numbered_dict = {i + 1: cache_list[i] for i in range(len(cache_list))}
numbered_dict
cache_dict ={1: 'cache_llama3.1_8b_all-mpnet-base-v2',
             2: 'cache_gemma2_all-mpnet-base-v2',
             3: 'cache_qwen2.5_14b_all-mpnet-base-v2',
             4: 'cache_qwen2_7b_dmis-lab_biobert-v1.1',
             5: 'cache_qwen2_7b_all-mpnet-base-v2',
             6: 'cache_qwen2.5_14b_dmis-lab_biobert-v1.1',
             7: 'cache_llama3.1_8b_dmis-lab_biobert-v1.1',
             8: 'cache_gemma2_dmis-lab_biobert-v1.1'}
#%%
file = os.path.join(os.getcwd(), cache_dict[1], 'graph_chunk_entity_relation.graphml')
G = nx.read_graphml(file)

# Graph statistics
def print_graph_info(G):
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Is directed: {G.is_directed()}")
    print(f"Graph density: {nx.density(G)}")
    if not G.is_directed():
        largest_cc = max(nx.connected_components(G), key=len)
        print(f"Largest connected component size: {len(largest_cc)}")
    else:
        largest_cc = max(nx.weakly_connected_components(G), key=len)
        print(f"Largest weakly connected component size: {len(largest_cc)}")

def get_graph_info(G):
    info = {"degrees": [d for n, d in G.degree()]}
    degrees = info["degrees"]

    info = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "is_directed": G.is_directed(),

        # Degree statistics
        "avg_degree" : sum(degrees) / len(degrees),
        "max_degree" : max(degrees),
        "min_degree" : min(degrees),

        "density": nx.density(G),
        "num_connected_components": nx.number_connected_components(G) if not G.is_directed() else nx.number_weakly_connected_components(G),
    }

    if not G.is_directed():
        largest_cc = max(nx.connected_components(G), key=len)
        info["largest_cc_size"] = len(largest_cc)
    else:
        largest_cc = max(nx.weakly_connected_components(G), key=len)
        info["largest_weakly_cc_size"] = len(largest_cc)

    # Print all statistics
    print(f"Number of nodes: {info["num_nodes"]}")
    print(f"Number of edges: {info["num_edges"]}")
    print(f"Average degree:  {info["avg_degree"]}")
    print(f"Maximum degree:  {info["max_degree"]}")
    print(f"Minimum degree:  {info["min_degree"]}")
    print(f"Density: {info["density"]}")
    print(f"Number of connected components: {info["num_connected_components"]}")
    if not G.is_directed():
        print(f"Largest connected component size: {info["largest_cc_size"]}")
    else:   
        print(f"Largest weakly connected component size: {info["largest_weakly_cc_size"]}")
    print(f"Is directed: {info["is_directed"]}")

    return info


info = get_graph_info(G)

# %%
# %%

# Get Component Statistics
# Trova tutti i componenti connessi
components = list(nx.connected_components(G))

# Trova il componente più grande
largest_component = max(components, key=len)

# Stampa il risultato
print("Largest Connected Component (LCC):")
print("Numero di nodi nell'LCC:", len(largest_component))
lcc_info = get_graph_info(G.subgraph(largest_component))

# %%

###python
import networkx as nx
import matplotlib.pyplot as plt

# Compute the Largest Connected Component (LCC)
lcc = max(nx.connected_components(G), key=len)  # Finding the LCC as a set of nodes
LCC_G = G.subgraph(lcc).copy()  # Create a subgraph from the LCC

# Save LCC graph to GraphML file
file = os.path.join(os.getcwd(), cache_dict[1], 'LCC_graph.graphml')
nx.write_graphml(LCC_G, file)

#%%

# Visualize the LCC
plt.figure(figsize=(10, 8))
nx.draw(LCC_G, with_labels=True, node_size=500, node_color="lightblue", edge_color="gray")
plt.title("Largest Connected Component")
plt.show()

# Access properties of nodes and edges
node_properties = LCC_G.nodes(data=True)  # List of nodes with their properties
edge_properties = LCC_G.edges(data=True)  # List of edges with their properties
node_properties = list(LCC_G.nodes.data())#
edge_properties = list(LCC_G.edges.data())

n = 8
# Output node and edge properties
print("Node Properties:")
for node, properties in node_properties[:n]:
    print(f"Node {node}: {properties}")
    print("<<<<<<<<<<>>>>>>>>>>>")

print("\nEdge Properties:")
for u, v, properties in edge_properties[:n]:
    print(f"Edge ({u}, {v}): {properties}")
    print("<<<<<<<<<<>>>>>>>>>>>")
###
#%%
node_properties[0][1]
edge_properties[0][2]
#%%
node_properties.keys()
# %%
import plotly.graph_objects as go
# Compute the Largest Connected Component (LCC)
lcc = max(nx.connected_components(G), key=len)  # Finding the LCC as a set of nodes
LCC_G = G.subgraph(lcc).copy()  # Create a subgraph from the LCC

GRAPH = LCC_G

# Posiciones de los nodos
pos = nx.spring_layout(G)

# Crear el gráfico
fig = go.Figure(data=[go.Scatter(
    x=[pos[node][0] for node in GRAPH.nodes],
    y=[pos[node][1] for node in GRAPH.nodes],
    mode="markers",
    hoverinfo="text",
    hovertext=[node for node in GRAPH.nodes],
    marker=dict(size=10)
)])

# Agregar aristas
for edge in GRAPH.edges:
    x = [pos[edge[0]][0], pos[edge[1]][0]]
    y = [pos[edge[0]][1], pos[edge[1]][1]]
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines",
        line=dict(width=2)
    ))

# Configuración del gráfico
fig.update_layout(
    title="Grafo de ejemplo",
    showlegend=False,
    hovermode="x",
    margin=dict(b=20, l=5, r=5, t=40),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))

# Mostrar el gráfico
fig.show()
# %%
