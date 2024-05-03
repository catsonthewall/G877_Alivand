import geopandas as gpd
import numpy as np


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

    def adj(self, v):
        return self.adj[v]

# Graph implemented by adjacency matrix
class Graph_matrix:
    def __init__(self):
        self.dis_matrix= np.zeros((4965, 4956))
        self.nodes = {}

    def add_edge(self, v, w, length):
        self.dis_matrix[v][w] = length
        self.dis_matrix[w][v] = length

    def add_node(self, node_idx, node):
        self.nodes[node_idx] = node

    def get_node(self, node_idx):
        return self.nodes[node_idx]

    def adj(self, v):
        return self.matirx[v]


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

    return graph

# Graph implemented by edge list::
    def __init__(self):
        self.edges = []
        self.nodes = set()

    def add_edge(self, source, target, length):
        self.edges.append((source, target, length))
        self.nodes.add(source)
        self.nodes.add(target)

def preprocess_gdf_to_edge_list(gdf):
    edge_list_graph = GraphFromEdgeList()

    for idx, row in gdf.iterrows():
        multilines = row['geometry']
        for points in multilines.geoms:
            xy = points.xy
            if len(xy[0]) >= 2:  # simplify the multilines to lines
                x1, y1 = xy[0][0], xy[1][0]
                x2, y2 = xy[0][-1], xy[1][-1]

                edge_list_graph.add_edge((x1, y1), (x2, y2), row['SHAPE_Length'])

    return edge_list_graph


if __name__ == "__main__":
    gdf = gpd.read_file('../data/Roads_small.gpkg')
    m=0

    '''
    #generate the graph stored in the adjacency list
    road_graph = preprocess_gdf(gdf)
    for v, w in road_graph.adj.items():
        print("Node:", v, "Edges:", w)
        node = road_graph.get_node(v)
        print("Latitude:", node.x, "Longitude:", node.y)
        m+=1
    print(m)

    edge_list_graph = preprocess_gdf_to_edge_list(gdf)
    for edge in edge_list_graph.edges[:10]:
        print("Edge:", edge)
    '''

    matrix_graph = preprocess_gdf(gdf, graph=Graph_matrix())
    print(matrix_graph.dis_matrix)
    print(matrix_graph.nodes[0].pos)