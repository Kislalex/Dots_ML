import sys
import random
import math
import time

import numpy as np
from shapely import geometry as gm

from brain import Brain


class Dot:
    def __init__(self, start, size = 1000):
        self.dot_brain = Brain(size)
        
        self.pos = np.array(start)
        self.vel = np.array([0, 0])
        self.acc = np.array([0, 0])
        
        self.dead = False
        self.reached_goal = False
        
        self.time = 0
        self.score = 0

    def move(self, field, ttl):
        action = self.dot_brain.signal(self.time)
        self.acc = np.copy(action)
        self.vel = np.add(self.vel, ttl * action)
        if (np.linalg.norm(self.vel) > 10.0):
            self.vel = self.vel * (10.0 / np.linalg.norm(self.vel))
        self.pos = np.add(self.pos, ttl * self.vel)
        self.time = self.time + 1

    def update(self, field):
        if ((not self.dead) and (not self.reached_goal)):
            self.move(field, 1)
            if (field.crashed(self.pos)):
                self.dead = True
            if (field.goal(self.pos)):
                self.reached_goal = True

    def computeScore(self, field):
        if (self.reached_goal):
            self.score = 1000.0 + 10000.0 / (self.time ** 2)
        else:
            dist = np.linalg.norm(np.subtract(self.pos, field.finish))
            self.score = 1000.0 / (dist ** 3)

    def reproduce(self, start):
        baby = Dot(start, 0)
        baby.dot_brain = self.dot_brain.copy()
        return baby





