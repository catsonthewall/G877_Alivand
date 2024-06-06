# G877_Alivand

I guess we need a plan on what needs to be done...
This is a draft (proposal)...

1. Find hiking paths vector data (Annika, already done)
2. Transform this data into a network (Baoyu, Jing, done)
4. Create a Dijkstras Algorithm for shortest distance, with some dummy data for testing (Annika)
   I have now added the file graph_routing.py, which is basically an extension of graph.py. There are dijkstra algorithms for the graph   implemented through the adjacency list and the edge list. For these both I implemented a basic Dijkstra Algorithm, that outputs the distances to all nodes. Then a dijkstra with an end node which also produces the shortest path distances to all nodes but does only output the distance to the specified end node. Then there is the dijkstra with end node and path. This is the same algorithm as before, but it also outputs the path taken in order to get to the end_node with the shortest distance.
6. Calculate a variable (for example viewshed...), that determines the scenicness (Fabio, Dominik) *done*
7. Create a Dijkstras with the scenicness as weights (to do later)
8. Choose some sample points to calculate our routes (to do later)
=> Compare these routes (to do later)

Network usage instructions:
In the process of creating the graph, we utilize three methods: adjacency list, edge list, and adjacent matrix. As Ross mentioned, if we have additional time later, we can conduct experiments to test these different methods, which will serve as a significant milestone for us. Now, you can choose one method you prefer.

I have organized the contents of the GitHub repository and moved all the old files to the "interm files" folder. Everyone can now organize the code in the "geo877" folder. 
