import networkx as nx

# Load graph from GraphML file
file = r"BLAP/graph_chunk_entity_relation.graphml"
G = nx.read_graphml(file)

# Number of nodes
num_nodes = G.number_of_nodes()

# Number of edges
num_edges = G.number_of_edges()

# Degree statistics
degrees = [d for n, d in G.degree()]
avg_degree = sum(degrees) / len(degrees)
max_degree = max(degrees)
min_degree = min(degrees)

# Density
density = nx.density(G)

# Connected components
num_connected_components = nx.number_connected_components(G) if not G.is_directed() else nx.number_weakly_connected_components(G)

# Largest connected component size
if not G.is_directed():
    largest_cc = max(nx.connected_components(G), key=len)
else:
    largest_cc = max(nx.weakly_connected_components(G), key=len)
largest_cc_size = len(largest_cc)

# Is the graph directed?
is_directed = G.is_directed()

# Print all statistics
print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {num_edges}")
print(f"Average degree: {avg_degree}")
print(f"Maximum degree: {max_degree}")
print(f"Minimum degree: {min_degree}")
print(f"Density: {density}")
print(f"Number of connected components: {num_connected_components}")
print(f"Largest connected component size: {largest_cc_size}")
print(f"Is directed: {is_directed}")

"""
Number of nodes: 267
Number of edges: 7
Average degree: 0.052434456928838954
Maximum degree: 3
Minimum degree: 0
Density: 0.00019712201852946975
Number of connected components: 260
Largest connected component size: 4
Is directed: False
"""

"""
Number of nodes: 431
Number of edges: 36
Average degree: 0.16705336426914152
Maximum degree: 6
Minimum degree: 0
Density: 0.00038849619597474777
Number of connected components: 401
Largest connected component size: 8
Is directed: False
"""