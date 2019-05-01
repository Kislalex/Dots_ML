import sys
import random
import time
import numpy as np
from dot import Dot

def poolUpdate(dot, field_):
    dot.update(field_)
    return dot

class Population:
    TTL = 900
    def __init__(self, start, size, rule):
        self.dots = [Dot(start, rule) for i in range(size)]
        self.total_score = 0
        self.best_dot_index = 0
        self.gen = 1
        self.start = start

    def update(self, field):
        #p = Pool(cpu_count())
        #poolFunction = partial(poolUpdate, field_ = field)
        #self.dots = p.map(poolFunction, self.dots)
        for dot in self.dots:
            dot.update(field)
        #p.close()
        #p.join()
           
    def computeScore(self, field):
        max_score = -1
        self.best_dot_index = 0
        self.total_score = 0
        for (ind,dot) in enumerate(self.dots):
            dot.computeScore(field)
            self.total_score += dot.score
            if (dot.score > max_score):
                max_score = dot.score
                self.best_dot_index = ind
        if (self.dots[self.best_dot_index].score < self.dots[0].score * 1.5):
            self.best_dot_index = 0
        if self.best_dot_index == 0:
            print("The leader still best")
        else:
            print("New leader found")
    
    def allDotsStopped(self):
        non_dead = 0
        for dot in self.dots:
            if (dot.time > self.TTL): 
                return 0
            if ((not dot.dead) and (not dot.reached_goal)):
                non_dead = non_dead + 1
        return non_dead

    def naturalSelection(self):
        new_dots = []
        new_dots.append(self.dots[self.best_dot_index].reproduce(self.start))
        for i in range(len(self.dots) - 1):
            parent_index = self.selectParent()
            new_dots.append(self.dots[parent_index].reproduce(self.start))
        self.dots = new_dots
        self.gen += 1

    def selectParent(self):
        best_rate = 0.9
        if (np.random.random_sample() > best_rate):
            return self.best_dot_index
        chance = np.random.random_sample() * self.total_score
        current_sum = 0
        for (index,dot) in enumerate(self.dots):
            current_sum += dot.score
            if (current_sum > chance):
                return index
        return 0

    def mutation(self):
        rate = 0.9
        for obj in self.dots[1:]:
            if (np.random.random_sample() < rate):
                obj.dot_brain.mutate()
