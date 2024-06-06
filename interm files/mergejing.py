import geopandas as gpd
import numpy as np
from shapely.geometry import MultiLineString, Point, LineString
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv


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



if __name__ == "__main__":
    gdf = gpd.read_file('../data/clean_graph.gpkg')
    graph = preprocess_gdf(gdf, graph=Graph())
    geom = []
    lengths = []
    delete_index = []

    mitnodes = [node_idx for node_idx, edges in graph.adj.items() if len(edges) == 2]

    '''
    def visit_node(graph, mitnodes, node_idx):
        if node_idx not in mitnodes:
            return
        mitnodes.remove(node_idx)
        pointslist.append(node_idx)

        for edge in graph.adj[node_idx]:
            if edge[0] not in pointslist:  # Add this check
                pointslist.append(edge[0])
                visit_node(graph, mitnodes, edge[0])
    '''

    for node_idx in tqdm(mitnodes, desc="processing"):
        pointslist = []
        MergeMitnodes.visit_node(graph, mitnodes, node_idx)
        pointslist = list(set(pointslist))
        print(pointslist)
        geometries = []

        lines_length = []


        for i in range(len(pointslist)):
            for j in range(i + 1, len(pointslist)):
                node1 = pointslist[i]
                node2 = pointslist[j]
                multilinestring, idx, length = MergeMitnodes.find_multilinestring(graph, node1, node2, gdf)
                if multilinestring is not None:
                    geometries.append(multilinestring)
                    delete_index.append(idx)
                    lines_length.append(length)


        total_length = sum(lines_length)
        lengths.append(total_length)
        print(len(geometries))
        merged_multilinestring= MergeMitnodes.merge_multilinestrings(geometries)
        print(merged_multilinestring)
        geom.append(merged_multilinestring)


    gdf = gpd.GeoDataFrame(geometry=geom, crs='EPSG:2056')
    gdf['SHAPE_Length'] = lengths
    output_file = '../data/new_clean_combined.gpkg'
    gdf.to_file(output_file, driver='GPKG')
    print(f"GeoPackage 文件已保存到: {output_file}")
    print(delete_index)
    delete_index_file = '../data/new_clean_delete_index.csv'
    with open(delete_index_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for index in delete_index:
            writer.writerow([index])
    print(f"Delete index 已保存到: {delete_index_file}")









