import geopandas as gpd


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.pos = (x, y)
        self.visited = False


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


def preprocess_gdf(gdf):
    points_dict = {}
    graph = Graph()
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


if __name__ == "__main__":
    gdf = gpd.read_file('D:\\anaconda\\envs\\NLP\\spatialalgorithm\\data\\Roads_small.gpkg')
    road_graph = preprocess_gdf(gdf)
    for v, w in road_graph.adj.items():
        if len(w) > 2:
            print("Node:", v, "Edges:", w)
            node = road_graph.get_node(v)
            print("Latitude:", node.x, "Longitude:", node.y)
    print(road_graph.nodes[0].pos)
