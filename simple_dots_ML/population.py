import sys
import random
import time
import numpy as np
from dot import Dot
from itertools import islice


class Population:
    TTL = 500

    def __init__(self, start, size, rule):
        self.dots = [Dot(start, rule) for i in range(size)]
        self.total_score = 0
        self.best_dot_index = 0
        self.gen = 1
        self.start = start

    def update(self, field):
        for dot in self.dots:
            dot.update(field)

    def compute_score(self, field):
        max_score = -1
        self.best_dot_index = 0
        self.total_score = 0
        for (ind, dot) in enumerate(self.dots):
            dot.compute_score(field)
            self.total_score += dot.score
            if dot.score > max_score:
                max_score = dot.score
                self.best_dot_index = ind
        return max_score

    def are_all_dots_stopped(self):
        non_dead = 0
        for dot in self.dots:
            if dot.time > self.TTL:
                return 0
            if (not dot.dead) and (not dot.reached_goal):
                non_dead = non_dead + 1
        return non_dead

    def natural_selection(self):
        new_dots = []
        new_dots.append(self.dots[self.best_dot_index].reproduce(self.start))
        for i in range(len(self.dots) - 1):
            parent_index = self.select_parent()
            new_dots.append(self.dots[parent_index].reproduce(self.start))
        self.dots = new_dots
        self.gen += 1

    def select_parent(self):
        chance = np.random.random_sample() * self.total_score
        current_sum = 0
        for (index, dot) in enumerate(self.dots):
            current_sum += dot.score
            if current_sum > chance:
                return index
        return 0

    def mutation(self):
        rate = 0.9
        for obj in islice(self.dots, 1, None):
            if np.random.random_sample() < rate:
                obj.dot_brain.mutate()
