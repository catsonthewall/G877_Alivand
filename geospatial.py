# Classes for the GEO 877 Alivand group project

from numpy import sqrt, argmin

#from numpy import sqrt

class Point():
    # initialise
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y

     # representation
    def __repr__(self):
        return f'Point(x={self.x}, y={self.y})'

    # method to find the closest pixel center
    def find_closest_pixel(self, pixel_centers):
        distances = [sqrt((center.x - self.x)**2 + (center.y - self.y)**2) for center in pixel_centers]
        closest_index = distances.index(min(distances))
        closest_pixel_center = pixel_centers[closest_index]
        return closest_pixel_center