import sys
import random
import time
import numpy as np
from dot import Dot
from itertools import islice

# from multiprocessing import Pool
from multiprocessing import Process


def dot_score_cmp(a):
    return -a.score


class FieldUpdate:
    def __init__(self, field):
        self.field = field

    def update(self, x):
        x.update(self.field)


class Population:
    TTL = 1000

    def __init__(self, start, size, rule):
        self.dots = [Dot(start, rule) for i in range(size)]
        self.total_score = 0
        self.best_dot_index = 0
        self.gen = 1
        self.start = start

    def update(self, field):
        for dot in self.dots:
            dot.update(field)

    def update_score(self, field):
        for dot in self.dots:
            dot.update_score(field)
        self.dots = sorted(self.dots, key=dot_score_cmp)
        return self.dots[0].score

    def clear_score(self):
        for dot in self.dots:
            dot.clear_score()

    def restart(self):
        for dot in self.dots:
            dot.pos = self.start
            dot.vel = np.array([0, 0])
            dot.acc = np.array([0, 0])
            dot.dead = False
            dot.reached_goal = False
            dot.checkpoints = 0
            dot.min_distance = 5000
            dot.time = 0

    def count_alive_dots(self):
        non_dead = 0
        for dot in self.dots:
            if dot.time > self.TTL:
                return 0
            if (not dot.dead) and (not dot.reached_goal):
                non_dead = non_dead + 1
        return non_dead

    def natural_selection(self):
        old_gen_count = len(self.dots) // 10
        new_dots = []
        for i in range(old_gen_count):
            new_dots.append(self.dots[i].reproduce(self.start))
            for j in range(9):
                new_born = self.dots[i].reproduce(self.start)
                new_born.dot_brain.mutate(0.1, 0.5)
                new_dots.append(new_born)
        self.dots = new_dots
        self.gen += 1
