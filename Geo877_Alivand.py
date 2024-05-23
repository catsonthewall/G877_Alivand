# Geo877 Spatial Algorithms - Group Alivand: Extracting Scenic Routes from VGI Data Sources 

# Use this python file for all the "background" code of the project such as functions and classes.

# This is the full graph-routing.py file which includes the different graph classes with their respective algorithms 
import geopandas as gpd
import numpy as np

def dfs_component(graph, node, component):
    # Add the current node to the component
    component.add(node)
    print(node)

    # Iterate over all neighbors of the current node
    for neighbor in graph.adj[node]:
        print(neighbor)
        # If the neighbor is not already in the component
        if neighbor not in component:
            # Recursively perform DFS on the neighbor
            dfs_component(graph, neighbor, component)



def find_connected_components(graph):
    # Initialize a set to keep track of visited nodes
    visited = set()
    # Initialize a list to store the connected components
    components = []

    # Iterate over each node in the graph
    for node in graph.nodes:
        # If the node has not been visited
        if node not in visited:
            # Initialize a new component set
            component = set()
            # Perform a depth-first search to find all nodes in this component
            dfs_component(graph, node, component)
            # Add the discovered component to the list of components
            components.append(component)
            # Mark all nodes in this component as visited
            visited.update(component)

    # Return the list of connected components
    return components



def get_largest_component(components):
    # Return the largest component by length
    return max(components, key=len)



def filter_graph(components, largest_component, gdf, graph):
    # Initialize a list to store nodes to remove
    nodes_to_remove = []

    # Iterate over each subgraph in the components list
    for subgraph in components:
        # If the subgraph is not the largest component
        if subgraph != largest_component:
            # Add the subgraph to the nodes to remove list
            nodes_to_remove.append(subgraph)

    # Flatten the list of nodes to remove
    delete_list = [item for subset in nodes_to_remove for item in subset]

    # Iterate over each row in the GeoDataFrame
    for idx, row in gdf.iterrows():
        # Get the geometry (MultiLineString) of the current row
        multilines = row['geometry']
        # Iterate over each line (LineString) in the MultiLineString
        for points in multilines.geoms:
            # Get the start and end points of the line
            xy = points.xy
            x1, y1 = xy[0][0], xy[1][0]
            x2, y2 = xy[0][-1], xy[1][-1]
            # Check if either end of the line is a node in the delete list
            for node in delete_list:
                node_pos = graph.nodes[node]['pos']
                if (x1, y1) == node_pos or (x2, y2) == node_pos:
                    # If so, drop the row from the GeoDataFrame
                    gdf.drop(idx, inplace=True)
                    break

    # Reset the index of the GeoDataFrame after dropping rows
    gdf.reset_index(drop=True, inplace=True)
    # Define the output file path
    output_file = '../data/clean_graph.gpkg'
    # Save the filtered GeoDataFrame to a file in GeoPackage format
    gdf.to_file(output_file, driver='GPKG')



class MergeMitnodes:
    @staticmethod
    def explore_node(graph, node_idx, mitnodes, pointslist, linklist):
        # Remove the current node from mitnodes
        mitnodes.remove(node_idx)
        # Mark the current node as visited
        graph.get_node(node_idx).visited = True
        # Add the current node to pointslist
        pointslist.append(node_idx)

        # Explore neighbors of the current node
        for edge in graph.adj[node_idx]:
            neighbor_idx = edge[0]
            if neighbor_idx in mitnodes:
                linklist.append(neighbor_idx)
                MergeMitnodes.explore_node(graph, neighbor_idx, mitnodes, pointslist, linklist)

    def combine_mitnodes(graph, mitnodes):
        while mitnodes:
            node_idx = mitnodes.pop(0)
            pointslist = []
            linklist = []
            MergeMitnodes.explore_node(graph, node_idx, mitnodes, pointslist, linklist)
            print(pointslist)

    def visit_node(graph, mitnodes, node_idx):
        if node_idx not in mitnodes:
            return
        mitnodes.remove(node_idx)
        pointslist.append(node_idx)

        for edge in graph.adj[node_idx]:
            if edge[0] not in pointslist:  # Add this check
                pointslist.append(edge[0])
                MergeMitnodes.visit_node(graph, mitnodes, edge[0])

    # Function to find the multilinestring between two nodes
    def find_multilinestring(graph, node1, node2, gdf):
        for idx, row in gdf.iterrows():
            multilines = row['geometry']
            length = row['SHAPE_Length']
            for points in multilines.geoms:
                xy = points.xy
                # Check if the start and end points of the multilinestring match the positions of node1 and node2
                if ((graph.get_node(node1).pos == (xy[0][0], xy[1][0]) or graph.get_node(node1).pos == (
                    xy[0][-1], xy[1][-1])) and
                        (graph.get_node(node2).pos == (xy[0][0], xy[1][0]) or graph.get_node(node2).pos == (
                        xy[0][-1], xy[1][-1]))):
                    return multilines, idx, length
        return None, None, None

    def merge_multilinestrings(geometries):
        # Loop to merge each multilinestring
        geom1 = geometries.pop(0)

        while len(geometries) > 0:
            # Flag to indicate whether a connected multilinestring was found
            found_connection = False

            # Attempt to find a multilinestring among the remaining ones that connects to the start or end of the current one
            for m, geom2 in enumerate(geometries):
                for line1, line2 in zip(geom1.geoms, geom2.geoms):
                    startpoint1 = line1.xy[0][0], line1.xy[1][0]
                    startpoint2 = line2.xy[0][0], line2.xy[1][0]
                    endpoint1 = line1.xy[0][-1], line1.xy[1][-1]
                    endpoint2 = line2.xy[0][-1], line2.xy[1][-1]
                    pointslist = []

                    if startpoint1 == startpoint2:
                        for i in range(1, len(line2.xy[0])):
                            pointslist.append(Point(line2.xy[0][-i], line2.xy[1][-i]))
                        for i in range(len(line1.xy[0])):
                            pointslist.append(Point(line1.xy[0][i], line1.xy[1][i]))
                        linestring = LineString(pointslist)
                        multilinestring = MultiLineString([linestring])
                        geom1 = multilinestring
                        geometries.pop(m)
                        found_connection = True
                        break
                    elif startpoint1 == endpoint2:
                        for i in range(len(line2.xy[0])):
                            pointslist.append(Point(line2.xy[0][i], line2.xy[1][i]))
                        for i in range(1, len(line1.xy[0])):
                            pointslist.append(Point(line1.xy[0][i], line1.xy[1][i]))
                        linestring = LineString(pointslist)
                        multilinestring = MultiLineString([linestring])
                        geom1 = multilinestring
                        geometries.pop(m)
                        found_connection = True
                        break
                    elif endpoint1 == startpoint2:
                        for i in range(len(line1.xy[0])):
                            pointslist.append(Point(line1.xy[0][i], line1.xy[1][i]))
                        for i in range(1, len(line2.xy[0])):
                            pointslist.append(Point(line2.xy[0][i], line2.xy[1][i]))
                        linestring = LineString(pointslist)
                        multilinestring = MultiLineString([linestring])
                        geom1 = multilinestring
                        geometries.pop(m)
                        found_connection = True
                        break
                    elif endpoint1 == endpoint2:
                        for i in range(len(line1.xy[0])):
                            pointslist.append(Point(line1.xy[0][i], line1.xy[1][i]))
                        for i in range(2, len(line2.xy[0]) + 1):
                            pointslist.append(Point(line2.xy[0][-i], line2.xy[1][-i]))
                        linestring = LineString(pointslist)
                        multilinestring = MultiLineString([linestring])
                        geom1 = multilinestring
                        geometries.pop(m)
                        found_connection = True
                        break

            if not found_connection:
                break

        return geom1

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
