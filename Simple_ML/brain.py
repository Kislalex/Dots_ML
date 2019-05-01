import sys
import random
import math
import time
import numpy


class Brain:
    def __init__(self, directions):
        self.ttl = directions
        self.coef = numpy.zeros((self.ttl, 2))  
        self.randomize(3)

    def randomize(self, max_acc):
        self.coef = max_acc * 2 * numpy.random.random_sample((self.ttl,2)) - max_acc

    def copy(self):
        copied = Brain(self.ttl)
        copied.coef = numpy.copy(self.coef)
        return copied

    def mutate(self, mutation_rate = 0.05):
        with numpy.nditer(self.coef, op_flags=['readwrite']) as it:
            for x in it:
                chance = numpy.random.random_sample()
                if (chance < mutation_rate):
                    x[...] = 6 * numpy.random.random_sample() - 3

    def signal(self, data):
        if (data < self.ttl):
            return self.coef[data]
        return self.coef[0]

