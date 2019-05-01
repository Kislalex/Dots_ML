import sys
import random
import math
import time
import numpy as np

def sigmaGsigmaF(transformations, shifts, data):
    result = data
    for (transform,shift) in zip(transformations,shifts):
        result = np.add(np.dot(transform, result), shift)
        result = np.tanh(result)
    return result

class Brain:
    def __init__(self, rule):
        self.rule = np.copy(rule)
        self.sizes = list(zip(rule[1:], rule[:-1]))
        x = lambda a:2 * np.random.random_sample(a) - 1
        self.mat_transformations = [x(size) for size in self.sizes]
        self.shifts =               [x(size[0]) for size in self.sizes]
        #self.randomize(10)

    def randomize(self, max_coeff):
        self.transformations = list(zip([np.random.random_sample(size)    for size in self.sizes],
                                        [np.random.random_sample(size[0]) for size in self.sizes]))

    def copy(self):
        copied = Brain((1,1))
        copied.mat_transformations = [np.copy(tr) for tr in self.mat_transformations ]
        copied.shifts = [np.copy(tr) for tr in self.shifts ]
        copied.sizes = list(self.sizes)
        return copied

    def mutate(self, mutation_rate = 0.04, stable_rate = 0.08):
        for transform in self.mat_transformations: 
            with np.nditer(transform, op_flags=['readwrite']) as it:
                for x in it:
                    chance = np.random.random_sample()
                    if (chance < mutation_rate):
                        x[...] = 2 * np.random.random_sample() - 1
                    elif (chance < stable_rate):
                        x[...] *= (0.9 + 0.2 * np.random.random_sample())
        for shift in self.shifts: 
            with np.nditer(shift, op_flags=['readwrite']) as it:
                for x in it:
                    chance = np.random.random_sample()
                    if (chance < mutation_rate):
                        x[...] = 2 * np.random.random_sample() - 1
                    elif (chance < stable_rate):
                        x[...] *= (0.9 + 0.2 * np.random.random_sample())
                        
    def signal(self, data):
        return sigmaGsigmaF(self.mat_transformations, self.shifts, data)

