# Classes for the GEO 877 Alivand group project

from numpy import sqrt, argmin

#from numpy import sqrt

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
        return sqrt((self.x-other.x)**2 + (self.y-other.y)**2)
    
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