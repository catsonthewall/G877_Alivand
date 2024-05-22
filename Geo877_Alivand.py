# Geo877 Spatial Algorithms - Group Alivand: Extracting Scenic Routes from VGI Data Sources 

# Use this python file for all the "background" code of the project such as functions and classes.

# This is the full graph-routing.py file which includes the different graph classes with their respective algorithms 
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
        self.adj_list = {} # I had to change the name here
        # It was named .adj before and it can't have the same name as a function...
        self.nodes = {}

    def add_edge(self, v, w, length):
        # This does not check if the adge is already in the graph...
        self.adj_list.setdefault(v, []).append((w, length)) 
        self.adj_list.setdefault(w, []).append((v, length))

    def add_node(self, node_idx, node):
        self.nodes[node_idx] = node

    def get_node(self, node_idx):
        return self.nodes[node_idx]
    
    def get_edge_length(self, v, w):
        for node, length in self.adj_list[v]:
            if node == w:
                return length
        return None

    def adj(self, v):
        return self.adj[v]
    
    def dijkstra(graph, start_node): # simple dijkstra, that outputs distances to all nodes
        distance = {node: float('inf') for node in graph.nodes}
        distance[start_node] = 0
        visited = set()
        pq = [(0, start_node)]

        while pq: 
            dist_u, u = min(pq)
            pq.remove((dist_u, u))
            if u in visited:
                continue
            visited.add(u)
            if u in graph.adj_list:
                for v, _ in graph.adj_list[u]:
                    alt = distance[u] + graph.get_edge_length(u, v)
                    if alt < distance.get(v, float('inf')):
                        distance[v] = alt
                        pq.append((alt, v))          

        return distance

    def dijkstra_with_end_node(graph, start_node, end_node): # still same dijkstra
        # calculates distance to all nodes, but shows only the distance to the defined end node
        distance = {node: float('inf') for node in graph.nodes}
        distance[start_node] = 0
        visited = set()
        pq = [(0, start_node)]    

        while pq: 
            dist_u, u = min(pq)
            pq.remove((dist_u, u))
            if u in visited:
                continue
            visited.add(u)
            if u in graph.adj_list:
                for v, _ in graph.adj_list[u]:
                    alt = distance[u] + graph.get_edge_length(u, v)
                    if alt < distance.get(v, float('inf')):
                        distance[v] = alt
                        pq.append((alt, v))          

        return distance[end_node] 
    
    def dijkstra_with_end_node_and_path(self, start_node, end_node):
        distance = {node: float('inf') for node in self.nodes}
        distance[start_node] = 0
        predecessors = {node: None for node in self.nodes}  # Predecessor nodes
        visited = set()
        pq = [(0, start_node)]
        
        predecessors[start_node] = None

        while pq: 
            dist_u, u = min(pq)
            pq.remove((dist_u, u))
            if u in visited:
                continue
            visited.add(u)
            if u in self.adj_list: 
                for v, _ in self.adj_list[u]:
                    alt = distance[u] + self.get_edge_length(u, v)                  
                    if alt < distance.get(v, float('inf')):
                        distance[v] = alt
                        predecessors[v] = u  # Update predecessor node for node v
                        pq.append((alt, v))

        # Extract shortest path
        shortest_path = []
        current_node = end_node
        while current_node is not None:
            shortest_path.append(current_node)
            current_node = predecessors[current_node]

        shortest_path.reverse()  # Reverse to get path from start_node to target

        return distance[end_node], shortest_path


# Graph implemented by adjacency matrix
# I did not do a Dijkstra algorithm for the adjacency matrix, since I could not get it to work.
class Graph_matrix:
    def __init__(self):
        self.dis_matrix= np.zeros((4965, 4956)) 
        # It would be better if the size was automatically updated based on the size of the input data...
        self.nodes = {}

    def add_edge(self, v, w, length):
        self.dis_matrix[v][w] = length
        self.dis_matrix[w][v] = length

    def add_node(self, node_idx, node):
        self.nodes[node_idx] = node

    def get_node(self, node_idx):
        return self.nodes[node_idx]

    def adj(self, v):
        return self.matirx[v] # Have you tested this? I think there is at least a spelling error. 


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

# Graph from Edge List
class GraphFromEdgeList:
    def __init__(self):
        self.edges = []
        self.nodes = set()

    def add_edge(self, source, target, length):
        self.edges.append((source, target, length))
        self.nodes.add(source)
        self.nodes.add(target)

    def dijkstra(self, start_node): 
        distance = {node: float('inf') for node in self.nodes}
        distance[start_node] = 0
        visited = set()
        pq = [(0, start_node)]

        while pq: 
            dist_u, u = min(pq)
            pq.remove((dist_u, u))
            if u in visited:
                continue
            visited.add(u)
            for edge in self.edges:
                if edge[0] == u:
                    v, length = edge[1], edge[2]
                    alt = distance[u] + length
                    if alt < distance.get(v, float('inf')):
                        distance[v] = alt
                        pq.append((alt, v))

        return distance

    def dijkstra_with_end_node(self, start_node, end_node):
        # Again, same comment as for the graph with adjacency list... 
        distance = {node: float('inf') for node in self.nodes}
        distance[start_node] = 0
        visited = set()
        pq = [(0, start_node)]

        while pq: 
            dist_u, u = min(pq)
            pq.remove((dist_u, u))
            if u in visited:
                continue
            visited.add(u)
            for edge in self.edges:
                if edge[0] == u:
                    v, length = edge[1], edge[2]
                    alt = distance[u] + length
                    if alt < distance.get(v, float('inf')):
                        distance[v] = alt
                        pq.append((alt, v))

        return distance[end_node]
    
    # It might be interesting or necessary if we have the path as well...
    def dijkstra_with_end_node_and_path(self, start_node, end_node):
        distance = {node: float('inf') for node in self.nodes}
        distance[start_node] = 0
        predecessors = {node: None for node in self.nodes}  # Predecessor nodes
        visited = set()
        pq = [(0, start_node)]

        while pq: 
            dist_u, u = min(pq)
            pq.remove((dist_u, u))
            if u in visited:
                continue
            visited.add(u)
            for edge in self.edges:
                if edge[0] == u:
                    v, length = edge[1], edge[2]
                    alt = distance[u] + length
                    if alt < distance.get(v, float('inf')):
                        distance[v] = alt
                        predecessors[v] = u  # Update predecessor node for node v
                        pq.append((alt, v))

        # Extract shortest path
        shortest_path = []
        current_node = end_node
        while current_node is not None:
            shortest_path.append(current_node)
            current_node = predecessors[current_node]

        shortest_path.reverse()  # Reverse to get path from start_node to target

        return distance[end_node], shortest_path
    

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

# Point class for scenicness scoring

class Point():
    # initialise
    def __init__(self, x=None, y=None, cat=None):
        self.x = x
        self.y = y
        self.cat = cat

     # representation
    def __repr__(self):
        return f'Point(x={self.x}, y={self.y})'

    # calculate Euclidean distance between two points
    def distEuclidean(self, other):
        return np.sqrt((self.x-other.x)**2 + (self.y-other.y)**2)
    
    # method to find the closest pixel center
    def find_closest_pixel(self, pixel_centers):
        distances = [self.distEuclidean(center) for center in pixel_centers]
        closest_index = distances.index(min(distances))
        closest_pixel_center = pixel_centers[closest_index]
        return closest_pixel_center
    
    # method to find the neighbors of a given pixel center
    def find_neighbors(self):
        neighbors = []
        for dx in [-100, 0, 100]:
            for dy in [-100, 0, 100]:
                neighbor_point = Point(self.x + dx, self.y + dy, self.cat)
                neighbors.append(neighbor_point)
        return neighbors
    
    # static method to get landcover values for neighborhood
    @staticmethod
    def get_landcover(neighborhood):
        values = [point.cat for point in neighborhood]
        return values