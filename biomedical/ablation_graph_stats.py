import os
import json
import networkx as nx
from tqdm.auto import tqdm

def list_cache_directories():
    current_dir = os.getcwd()
    cache_dirs = []
    
    for item in os.listdir(current_dir):
        if os.path.isdir(item) and item.startswith('cache_'):
            cache_dirs.append(item)
    
    return cache_dirs

def get_graph_info(directory):
    graph_file = os.path.join(directory, 'graph_chunk_entity_relation.graphml')
    if not os.path.exists(graph_file):
        print(f"Graph file not found in {directory}. Skipping...")
        return None
    
    G = nx.read_graphml(graph_file)
    
    # Basic statistics
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    is_directed = G.is_directed()
    
    # Degree statistics
    degrees = dict(G.degree())
    avg_degree = sum(degrees.values()) / num_nodes if num_nodes > 0 else 0
    max_degree = max(degrees.values()) if degrees else 0
    min_degree = min(degrees.values()) if degrees else 0
    
    # Graph density
    density = nx.density(G)
    
    # Connected components
    if is_directed:
        components = list(nx.weakly_connected_components(G))
    else:
        components = list(nx.connected_components(G))
    
    num_components = len(components)
    lcc_size = len(max(components, key=len)) if components else 0
    
    # Isolated nodes (degree = 0)
    isolated_nodes = sum(1 for node, degree in degrees.items() if degree == 0)
    
    # Self-loops
    num_self_loops = nx.number_of_selfloops(G)
    
    # Small components (< 3 nodes)
    small_components = sum(1 for comp in components if len(comp) < 3)
    
    # Entity types from node attribute 'd0' (entity_type)
    entity_types = []
    lcc_entity_types = []
    
    if components:
        lcc = max(components, key=len)
        
        for node in G.nodes(data=True):
            node_id, node_data = node
            if 'entity_type' in node_data:
                entity_type = node_data['entity_type'].strip('"')
                entity_types.append(entity_type)
                if node_id in lcc:
                    lcc_entity_types.append(entity_type)
    
    # Get unique entity types
    unique_entity_types = list(set(entity_types))
    unique_lcc_entity_types = list(set(lcc_entity_types))
    
    # Additional advanced statistics
    stats = {}
    
    # Only compute expensive metrics for non-trivial graphs
    if num_nodes > 1 and num_edges > 0:
        # Centrality measures (compute on LCC if disconnected)
        if num_components > 1 and lcc_size > 1:
            lcc_subgraph = G.subgraph(max(components, key=len))
            betweenness = nx.betweenness_centrality(lcc_subgraph)
            closeness = nx.closeness_centrality(lcc_subgraph)
            eigenvector = nx.eigenvector_centrality(lcc_subgraph, max_iter=1000)
        else:
            betweenness = nx.betweenness_centrality(G)
            closeness = nx.closeness_centrality(G)
            try:
                eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
            except:
                eigenvector = {}
        
        # Centrality statistics
        stats['avg_betweenness'] = sum(betweenness.values()) / len(betweenness) if betweenness else 0
        stats['avg_closeness'] = sum(closeness.values()) / len(closeness) if closeness else 0
        stats['avg_eigenvector'] = sum(eigenvector.values()) / len(eigenvector) if eigenvector else 0
        
        # Clustering coefficient
        stats['avg_clustering'] = nx.average_clustering(G)
        
        # Assortativity by degree
        try:
            stats['degree_assortativity'] = nx.degree_assortativity_coefficient(G)
        except:
            stats['degree_assortativity'] = None
            
        # Diameter (on LCC if disconnected)
        try:
            if num_components == 1:
                stats['diameter'] = nx.diameter(G)
            elif lcc_size > 1:
                lcc_subgraph = G.subgraph(max(components, key=len))
                stats['diameter'] = nx.diameter(lcc_subgraph)
            else:
                stats['diameter'] = 0
        except:
            stats['diameter'] = None
            
        # Homophily by entity type
        if entity_types:
            entity_edges = 0
            same_type_edges = 0
            for u, v in G.edges():
                u_type = G.nodes[u].get('entity_type')
                v_type = G.nodes[v].get('entity_type')
                if u_type and v_type:
                    entity_edges += 1
                    if u_type == v_type:
                        same_type_edges += 1
            stats['entity_homophily'] = same_type_edges / entity_edges if entity_edges > 0 else 0
        else:
            stats['entity_homophily'] = 0
            
        # Number of hubs (nodes with degree > 2 * avg_degree)
        hub_threshold = 2 * avg_degree
        stats['num_hubs'] = sum(1 for degree in degrees.values() if degree > hub_threshold)
        
        # Disconnected hubs (hubs not in the LCC)
        if num_components > 1 and components:
            lcc = max(components, key=len)
            disconnected_hubs = []
            for node, degree in degrees.items():
                if degree > hub_threshold and node not in lcc:
                    disconnected_hubs.append(node)
            stats['num_disconnected_hubs'] = len(disconnected_hubs)
        else:
            stats['num_disconnected_hubs'] = 0
    
    else:
        # Default values for trivial graphs
        stats.update({
            'avg_betweenness': 0, 'avg_closeness': 0, 'avg_eigenvector': 0,
            'avg_clustering': 0, 'degree_assortativity': None, 'diameter': None,
            'entity_homophily': 0, 'num_hubs': 0, 'num_disconnected_hubs': 0
        })
    
    return {
        'directory': directory,
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'is_directed': is_directed,
        'graph_density': density,
        'avg_degree': avg_degree,
        'max_degree': max_degree,
        'min_degree': min_degree,
        'num_connected_components': num_components,
        'lcc_size': lcc_size,
        'num_isolated_nodes': isolated_nodes,
        'num_self_loops': num_self_loops,
        'num_small_components': small_components,
        'entity_types': unique_entity_types,
        'lcc_entity_types': unique_lcc_entity_types,
        **stats
    }

def save_graph_stats(stats, filename):
    """Save graph statistics to JSON file in graphs_stats/ directory"""
    # Create directory if it doesn't exist
    os.makedirs('graphs_stats', exist_ok=True)
    
    # Ensure .json extension
    if not filename.endswith('.json'):
        filename += '.json'
    
    filepath = os.path.join('graphs_stats', filename)
    
    # Save to JSON
    with open(filepath, 'w') as f:
        json.dump(stats, f, indent=2)
    
    return filepath


if __name__ == "__main__":
    cache_directories = list_cache_directories()
    
    if cache_directories:
        for directory in tqdm(cache_directories):
            stats = get_graph_info(directory)
            if stats:
                filename = f"graph_stats_{directory.replace('cache_', '')}.json"
                save_graph_stats(stats, filename)

    else:
        print("No directories starting with 'cache_' found.")