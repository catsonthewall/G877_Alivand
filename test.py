import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
import pandas as pd
from shapely.geometry import MultiLineString, Point, LineString

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

def find_connected_components(graph):
    def dfs_component(node, component):
        component.add(node)
        for neighbor, _ in graph.adj[node]:
            if neighbor not in component:
                dfs_component(neighbor, component)

    visited = set()
    components = []

    for node in graph.nodes:
        if node not in visited:
            component = set()
            dfs_component(node, component)
            components.append(component)
            visited.update(component)

    return components

def get_largest_component(components):
    return max(components, key=len)

def filter_graph(components, largest_component, gdf, graph):
    nodes_to_remove = [subgraph for subgraph in components if subgraph != largest_component]
    delete_list = [item for subset in nodes_to_remove for item in subset]

    for idx, row in gdf.iterrows():
        multilines = row['geometry']
        for points in multilines.geoms:
            xy = points.xy
            x1, y1 = xy[0][0], xy[1][0]
            x2, y2 = xy[0][-1], xy[1][-1]
            for node in delete_list:
                node_pos = graph.nodes[node].pos
                if (x1, y1) == node_pos or (x2, y2) == node_pos:
                    gdf.drop(idx, inplace=True)
                    break

    gdf.reset_index(drop=True, inplace=True)
    output_file = '../data/clean_graph.gpkg'
    gdf.to_file(output_file, driver='GPKG')

class MergeMitnodes:
    @staticmethod
    def explore_node(graph, node_idx, mitnodes, pointslist, linklist):
        mitnodes.remove(node_idx)
        graph.get_node(node_idx).visited = True
        pointslist.append(node_idx)

        for neighbor_idx, _ in graph.get_adjacent(node_idx):
            if neighbor_idx in mitnodes:
                linklist.append(neighbor_idx)
                MergeMitnodes.explore_node(graph, neighbor_idx, mitnodes, pointslist, linklist)

    @staticmethod
    def visit_node(graph, mitnodes, node_idx):
        if node_idx not in mitnodes:
            return
        mitnodes.remove(node_idx)
        pointslist.append(node_idx)

        for neighbor_idx, _ in graph.get_adjacent(node_idx):
            if neighbor_idx not in pointslist:
                pointslist.append(neighbor_idx)
                MergeMitnodes.visit_node(graph, mitnodes, neighbor_idx)

    @staticmethod
    def find_multilinestring(graph, node1, node2, gdf):
        for idx, row in gdf.iterrows():
            multilines = row['geometry']
            length = row['SHAPE_Length']
            for points in multilines.geoms:
                xy = points.xy
                if ((graph.get_node(node1).pos == (xy[0][0], xy[1][0]) or graph.get_node(node1).pos == (
                        xy[0][-1], xy[1][-1])) and
                    (graph.get_node(node2).pos == (xy[0][0], xy[1][0]) or graph.get_node(node2).pos == (
                        xy[0][-1], xy[1][-1]))):
                    return multilines, idx, length
        return None, None, None

    @staticmethod
    def merge_multilinestrings(geometries):
        geom1 = geometries.pop(0)

        while len(geometries) > 0:
            found_connection = False

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

def print_graph_structure(graph):
    for node_idx, node in graph.nodes.items():
        print(f"Node {node_idx}: Position {node.pos}")
    for node_idx, edges in graph.adj.items():
        for neighbor, length in edges:
            print(f"Edge from {node_idx} to {neighbor} with length {length}")

if __name__ == "__main__":
    #gdf = gpd.read_file('../data/Roads_small.gpkg')
    gdf = gpd.read_file('/Users/baoyuliu/Documents/GitHub/G877_Alivand/Roads_small.gpkg')
    road_graph = preprocess_gdf(gdf)
    print(f"Original graph has {len(road_graph.nodes)} nodes.")

    components = find_connected_components(road_graph)
    largest_component = get_largest_component(components)
    filter_graph(components, largest_component, gdf, road_graph)

    gdf = gpd.read_file('../data/clean_graph.gpkg')
    graph = preprocess_gdf(gdf, graph=Graph())

    geom = []
    lengths = []
    delete_index = []
    mitnodes = [node_idx for node_idx, edges in graph.adj.items() if len(edges) == 2]
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
        merged_multilinestring = MergeMitnodes.merge_multilinestrings(geometries)
        print(merged_multilinestring)
        geom.append(merged_multilinestring)

    gdf = gpd.GeoDataFrame(geometry=geom, crs='EPSG:2056')
    gdf['SHAPE_Length'] = lengths
    output_file = '../data/new_clean_combined.gpkg'
    gdf.to_file(output_file, driver='GPKG')
    print(f"GeoPackage file has already been saved to : {output_file}")

    delete_index_file = '../data/new_clean_delete_index.csv'
    with open(delete_index_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for index in delete_index:
            writer.writerow([index])
    print(f"Delete index has already been saved to: {delete_index_file}")

    to_delete = pd.read_csv('../data/new_clean_delete_index.csv')
    input_file = '../data/new_clean_combined.gpkg'
    combined_gdf = gpd.read_file(input_file)
    gdf = gpd.read_file('../data/clean_graph.gpkg')

    fig, ax = plt.subplots()
    to_delete_list = to_delete.iloc[:, 0].tolist()
    gdf.drop(to_delete_list, inplace=True)
    gdf.reset_index(drop=True, inplace=True)
    gdf = gdf[['SHAPE_Length', 'geometry']]
    combined_gdf.rename(columns={'length': 'SHAPE_Length'}, inplace=True)
    new_gdf = pd.concat([gdf.reset_index(drop=True), combined_gdf.reset_index(drop=True)], axis=0, ignore_index=True)
    output_file = '../data/final_clean_combined.gpkg'
    new_gdf.to_file(output_file, driver='GPKG')
    new_gdf.reset_index(drop=True, inplace=True)
    print(f"GeoPackage file has already been saved to : {output_file}")

    new_gdf.plot(ax=ax)
    plt.show()

    print_graph_structure(road_graph)
    print_graph_structure(new_gdf)
