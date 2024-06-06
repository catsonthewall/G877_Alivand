import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from shapely.geometry import MultiLineString, Point, LineString


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.pos = (x, y)
        self.visited = False


# Graph implemented by adjacency list
class Graph:
    def __init__(self):
        self.adj = {}
        self.nodes = {}

    def add_edge(self, v, w, length):
        self.adj.setdefault(v, []).append((w, length))
        self.adj.setdefault(w, []).append((v, length))

    def add_node(self, node_idx, node):
        self.nodes[node_idx] = node

    def get_node(self, node_idx):
        return self.nodes[node_idx]

    def get_adjacent(self, v):
        return self.adj[v]

    def remove_node(self, node_idx):
        if node_idx in self.adj:
            del self.adj[node_idx]
        if node_idx in self.nodes:
            del self.nodes[node_idx]
        for neighbors in self.adj.values():
            neighbors[:] = [n for n in neighbors if n[0] != node_idx]


def preprocess_gdf(gdf, graph=Graph()):
    points_dict = {}
    node_index = 0

    for idx, row in gdf.iterrows():
        multilines = row['geometry']
        for points in multilines.geoms:
            xy = points.xy
            if len(xy[0]) >= 2:  # simplify the multilines to lines
                x1, y1 = xy[0][0], xy[1][0]
                x2, y2 = xy[0][-1], xy[1][-1]

                v = points_dict.get((x1, y1))
                if v is None:
                    v = node_index
                    node = Node(x1, y1)
                    graph.add_node(v, node)
                    points_dict[(x1, y1)] = node_index
                    node_index += 1

                w = points_dict.get((x2, y2))
                if w is None:
                    w = node_index
                    node = Node(x2, y2)
                    graph.add_node(w, node)
                    points_dict[(x2, y2)] = w
                    node_index += 1

                # add length
                length = row['SHAPE_Length']
                graph.add_edge(v, w, length)
                print(f"Added edge from {v} to {w} with length {length}")

    return graph


def find_important_nodes(graph):
    important_nodes = {node_idx: i for i, (node_idx, edges) in enumerate(graph.adj.items()) if len(edges) >= 3}
    return important_nodes


def build_new_graph(graph, important_nodes):
    new_graph = Graph()
    node_map = {}

    # Add important nodes to the new graph
    for old_idx in important_nodes:
        new_idx = len(new_graph.nodes)
        node = graph.get_node(old_idx)
        new_graph.add_node(new_idx, node)
        node_map[old_idx] = new_idx
        print(f"Added important node {old_idx} as new node {new_idx}")

    # Create a list to store nodes to be removed
    nodes_to_remove = []

    # Traverse the graph and merge nodes
    for node_idx in list(graph.nodes):  # Convert keys to a list to avoid runtime error
        if node_idx not in important_nodes:
            neighbors = graph.get_adjacent(node_idx)
            if len(neighbors) == 2:
                (n1, l1), (n2, l2) = neighbors
                new_length = l1 + l2
                graph.add_edge(n1, n2, new_length)
                graph.adj[n1] = [(n, l) for n, l in graph.adj[n1] if n != node_idx]
                graph.adj[n2] = [(n, l) for n, l in graph.adj[n2] if n != node_idx]
                nodes_to_remove.append(node_idx)
                print(f"Merged node {node_idx} between nodes {n1} and {n2}")

    # Remove nodes after iteration
    for node_idx in nodes_to_remove:
        graph.remove_node(node_idx)
        print(f"Removed node {node_idx}")

    # Add remaining edges to the new graph
    for old_idx, new_idx in node_map.items():
        for neighbor, length in graph.get_adjacent(old_idx):
            if neighbor in node_map:
                new_neighbor_idx = node_map[neighbor]
                if new_idx != new_neighbor_idx:
                    new_graph.add_edge(new_idx, new_neighbor_idx, length)
                    print(f"Added edge from {new_idx} to {new_neighbor_idx} with length {length}")

    return new_graph


def remove_isolated_nodes(graph):
    nodes_to_remove = [node_idx for node_idx, neighbors in graph.adj.items() if len(neighbors) == 0]
    for node_idx in nodes_to_remove:
        graph.remove_node(node_idx)
        print(f"Removed isolated node {node_idx}")


def plot_graph(graph):
    G = nx.Graph()
    pos = {}

    for node_idx, node in graph.nodes.items():
        G.add_node(node_idx)
        pos[node_idx] = node.pos

    for node_idx, edges in graph.adj.items():
        for neighbor, length in edges:
            G.add_edge(node_idx, neighbor, weight=length)

    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, with_labels=True, node_size=50, font_size=8)
    plt.show()


if __name__ == "__main__":
    gdf = gpd.read_file('/Users/baoyuliu/Documents/GitHub/G877_Alivand/Roads_small.gpkg')

    # Generate the graph stored in the adjacency list
    road_graph = preprocess_gdf(gdf)
    print(f"Original graph has {len(road_graph.nodes)} nodes.")

    important_nodes = find_important_nodes(road_graph)
    print(f"Found {len(important_nodes)} important nodes.")

    new_graph = build_new_graph(road_graph, important_nodes)
    print(f"Reduced graph has {len(new_graph.nodes)} nodes.")

    # Draw original GeoDataFrame
    gdf.plot()
    plt.title('Original Road Data')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

    # Print and plot original graph
    for v, w in road_graph.adj.items():
        print("Node:", v, "Edges:", w)
        node = road_graph.get_node(v)
        print("Latitude:", node.x, "Longitude:", node.y)

    plot_graph(road_graph)

    # Print and plot reduced graph
    for v, w in new_graph.adj.items():
        print("Node:", v, "Edges:", w)
        node = new_graph.get_node(v)
        print("Latitude:", node.x, "Longitude:", node.y)

    plot_graph(new_graph)

    # Remove isolated nodes from new graph
    remove_isolated_nodes(new_graph)
    print(f"New graph after removing isolated nodes has {len(new_graph.nodes)} nodes.")

    # Print and plot final graph
    for v, w in new_graph.adj.items():
        print("Node:", v, "Edges:", w)
        node = new_graph.get_node(v)
        print("Latitude:", node.x, "Longitude:", node.y)

    plot_graph(new_graph)
