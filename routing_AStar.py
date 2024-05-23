import math 

## A* Search

## Priority queue algorithm
class PriorityQueue:
    def __init__(self):
        self.elements = []

    def is_empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        self.elements.append((priority, item))
        self.elements.sort(key=lambda x: x[0])

    def get(self):
        return self.elements.pop(0)[1] 

# g: movement cost to move from the starting point to a given square on the grid
# h: estimated movement cost from the given square to the final destination
# f = g + h

# A* seach algorithm using a grid
class Cell:
    def __init__(self):
        self.parent_i = 0
        self.parent_j = 0
        self.f = float('inf')
        self.g = float('inf')
        self.h = 0
        

class AStarSearch_grid:
    def __init__(self, grid):
        self.grid = grid
        self.ROW = len(grid)
        self.COL = len(grid[0])
        self.closed_list = None
        self.cell_details = None
        
    #check if the cell is valid within the grid
    def is_valid(self, row, col):
        return (row >= 0) and (row < self.ROW) and (col >= 0) and (col < self.COL)
    #check if the cell is unblocked
    def is_unblocked(self, row, col):
        return self.grid[row][col] == 1
    #check if a cell is the destination
    def is_destination(self, row, col, dest):
        return row == dest[0] and col == dest[1]
    
    #heuristic value of a cell (Manhattan distance)
    def calculate_h_value(row, col, dest):
        return abs(row - dest[0]) + abs(col - dest[1])
    
    #trace the path
    def trace_path(cell_details, dest):
        print("The Path is ")
        path = []
        row = dest[0]
        col = dest[1]
        
        while not (self.cell_details[row][col].parent_i == row and 
                   self.cell_details[row][col].parent_j == col):
            path.append((row, col))
            temp_row = self.cell_details[row][col].parent_i
            temp_col = self.cell_details[row][col].parent_j
            row = temp_row
            col = temp_col
            
        path.append((row, col))
        path.reverse()
 
        #print the path
        for i in path:
            print("->", i, end=" ")
            print()
            
    #implement the A* Search algorithm
    def a_star_search(self, src, dest):
        if not self.is_valid(src[0], src[1]) or not 
        self.is_valid(dest[0], dest[1]):
            print("Source or destination is invalid")
            return

        if not self.is_unblocked(src[0], src[1]) or not 
        self.is_unblocked(dest[0], dest[1]):
            print("Source or the destination is blocked")
            return

        if self.is_destination(src[0], src[1], dest):
            print("We are already at the destination")
            return
        #initialize the closed_list (visited)
        self.closed_list = [[False for _ in range(self.COL)] for _ in range(self.ROW)]
        #detail of each cell
        self.cell_details = [[Cell() for _ in range(self.COL)] for _ in range(self.ROW)]
        #initialize the start cell details
        i, j = src
        self.cell_details[i][j].f = 0
        self.cell_details[i][j].g = 0
        self.cell_details[i][j].h = 0
        self.cell_details[i][j].parent_i = i
        self.cell_details[i][j].parent_j = j
        #implement priority queue
        open_list = PriorityQueue()
        open_list.put((i, j), 0.0)
        
        #flag for finding the destination
        found_dest = False
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        #main loop for A* Seach 
        while not open_list.is_empty():
            #implement the priority queue class to pop the cell with the smallest f
            p = open_list.get()
            #mark visited cell
            i, j = p
            self.closed_list[i][j] = True
            
            #check successors for each direction 
            for dir in directions:
                new_i = i + dir[0]
                new_j = j + dir[1]

                if self.is_valid(new_i, new_j) and self.is_unblocked(new_i, new_j) and not
                self.closed_list[new_i][new_j]:
                    if self.is_destination(new_i, new_j, dest):
                        self.cell_details[new_i][new_j].parent_i = i
                        self.cell_details[new_i][new_j].parent_j = j
                        print("The destination cell is found")
                        #trace and print the path
                        self.trace_path(dest)
                        found_dest = True
                        return

                    g_new = self.cell_details[i][j].g + 1.0
                    h_new = self.calculate_h_value(new_i, new_j, dest)
                    f_new = g_new + h_new
                    
                    if self.cell_details[new_i][new_j].f == float('inf') or
                    self.cell_details[new_i][new_j].f > f_new:
                        open_list.put((new_i, new_j), f_new)
                        self.cell_details[new_i][new_j].f = f_new
                        self.cell_details[new_i][new_j].g = g_new
                        self.cell_details[new_i][new_j].h = h_new
                        self.cell_details[new_i][new_j].parent_i = i
                        self.cell_details[new_i][new_j].parent_j = j

        if not found_dest:
            print("Failed to find the destination cell")
        
    
    
## A* seach algorithm using a graph (not working yet)

class AStarSearch:
    def __init__(self, graph):
        self.graph = graph

    def search(self, start, goal):
        open_set = PriorityQueue()
        open_set.put(start, 0)
        
        came_from = {}
        g_score = {node: float('inf') for node in self.graph}
        g_score[start] = 0
        f_score = {node: float('inf') for node in self.graph}
        # Manhattan distance
        f_score[start] = abs(start[0] - goal[0]) + abs(start[1] - goal[1])  
        
        while not open_set.is_empty():
            current = open_set.get()
            
            if current == goal:
                # Reconstruct path
                total_path = [current]
                while current in came_from:
                    current = came_from[current]
                    total_path.append(current)
                total_path.reverse()
                return total_path
            
            for neighbor, cost in self.graph[current].items():
                tentative_g_score = g_score[current] + cost
                
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
                    open_set.put(neighbor, f_score[neighbor])
        
        return None

