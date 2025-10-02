# preprocess_kg_networkx.py
import networkx as nx
import os
import pickle
from tqdm import tqdm

# --- CONFIGURATION ---
RAW_DATA_DIR = "/run/media/sakhil/sakhil/IIIT H/ANLP/Project/freebase-easy-14-04-14/"
OUT_DIR = "/run/media/sakhil/sakhil/IIIT H/ANLP/Project/NetworkXGraph/"

FACTS_FILE = os.path.join(RAW_DATA_DIR, "facts.txt")

# ---------------------
# THE DUMMY FILE CREATION BLOCK HAS BEEN REMOVED FROM HERE
# ---------------------

os.makedirs(OUT_DIR, exist_ok=True)

print("--- Step 1: Building Mappings (Entity/Relation -> Integer ID) ---")
# This part is identical to the DGL version, as it's library-agnostic.
entity_map = {}
relation_map = {}

def add_to_map(item, item_map):
    if item not in item_map:
        item_map[item] = len(item_map)

# This will now read your actual large Facts.txt file
with open(FACTS_FILE, 'r', encoding='utf-8', errors='ignore') as f:
    for line in tqdm(f, desc="Mapping entities and relations"):
        try:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                subject, predicate, obj = parts[0], parts[1], parts[2]
                add_to_map(subject, entity_map)
                add_to_map(predicate, relation_map)
                add_to_map(obj, entity_map)
        except Exception:
            pass

print(f"Found {len(entity_map)} unique entities and {len(relation_map)} unique relations.")

map_path = os.path.join(OUT_DIR, "mappings.pkl")
with open(map_path, 'wb') as f:
    pickle.dump({'entities': entity_map, 'relations': relation_map}, f)
print(f"Mappings saved to {map_path}")


print("\n--- Step 2: Building the NetworkX Graph ---")
# Use a MultiDiGraph to handle multiple different edges between two nodes
G = nx.MultiDiGraph()

with open(FACTS_FILE, 'r', encoding='utf-8', errors='ignore') as f:
    for line in tqdm(f, desc="Creating graph edges"):
        try:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                subject, predicate, obj = parts[0], parts[1], parts[2]
                src_id = entity_map[subject]
                dest_id = entity_map[obj]
                
                # Add an edge with the relation name as an attribute
                G.add_edge(src_id, dest_id, relation=predicate)
        except Exception:
            pass

# Add all nodes to the graph, including those that might not be a source node
G.add_nodes_from(range(len(entity_map)))

print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# Save the graph using Python's pickle module
graph_path = os.path.join(OUT_DIR, "kg_graph.gpickle")
with open(graph_path, 'wb') as f:
    pickle.dump(G, f)
print(f"NetworkX graph saved to {graph_path}")
print("\nPreprocessing with NetworkX is complete!")