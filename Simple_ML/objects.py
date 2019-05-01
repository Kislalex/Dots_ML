import sys
import random
import math
import time
import numpy
def point_vector(first, second):
    vector = [y - x for (x,y) in list(zip(first,second))]
    return vector


def norm(vector):
    d = 0
    for x in vector:
        d = d + x ** 2
    return numpy.sqrt(d)

def dist(first,second):
    return norm(point_vector(first,second))

class MyPhysicsObject:        
    def __init__(self, weight, start=[0,0,0], speed = [0,0,0]):
        self.weight = weight
        self.position = list(start)
        self.velocity = list(speed)
        self.forces = [0,0,0]
    def compute_force(self, objects):
        self.forces = [0,0,0]
        for current_obj in objects:
            vector = point_vector(self.position, current_obj.position)
            distance = norm(vector)
            if (distance < 10): 
                continue
            newton_coefficient = 10000 * self.weight * current_obj.weight / (distance ** 3)# gmM/R^2
            self.forces = [x + vector[i] * newton_coefficient for i,x in enumerate(self.forces)]
    
    def update(self, time = 0.01):
        self.position = [x + time * self.velocity[i] for i,x in enumerate(self.position)]
        self.velocity = [x + (time * self.forces[i]) / self.weight for i,x in enumerate(self.velocity)]


#test = MyPhysicsObject(1000,[0,0,0],[1,10,0])
#while True:
#    print(test.position)
#    test.update(1)
#    time.sleep(1)
