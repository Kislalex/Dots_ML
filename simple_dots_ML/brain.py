import sys
import random
import math
import time
import numpy as np


def applySeriesOfLinearTransformations(transformations, shifts, data):
    result = data
    for (transform, shift) in zip(transformations, shifts):
        result = np.add(np.dot(transform, result), shift)
        result = np.tanh(result)
    return result


class Brain:
    def __init__(self, rule):
        self.rule = np.copy(rule)
        self.sizes = list(zip(rule[1:], rule[:-1]))
        x = lambda a: 2 * np.random.random(a) - 1
        self.mat_transformations = [x(size) for size in self.sizes]
        self.shifts = [x(size[0]) for size in self.sizes]

    def copy(self):
        copied = Brain((1, 1))
        copied.mat_transformations = [np.copy(tr) for tr in self.mat_transformations]
        copied.shifts = [np.copy(tr) for tr in self.shifts]
        copied.sizes = list(self.sizes)
        return copied

    def mutate(self, mutation_rate=0.04, stable_rate=0.08):
        for transform in self.mat_transformations:
            with np.nditer(transform, op_flags=["readwrite"]) as it:
                for x in it:
                    chance = np.random.random()
                    if chance < mutation_rate:
                        x[...] = 2 * np.random.random() - 1
                    elif chance < stable_rate:
                        x[...] *= 0.9 + 0.2 * np.random.random()
        for shift in self.shifts:
            with np.nditer(shift, op_flags=["readwrite"]) as it:
                for x in it:
                    chance = np.random.random()
                    if chance < mutation_rate:
                        x[...] = 2 * np.random.random() - 1
                    elif chance < stable_rate:
                        x[...] *= 0.9 + 0.2 * np.random.random()

    def signal(self, data):
        return applySeriesOfLinearTransformations(
            self.mat_transformations, self.shifts, data
        )

    def save_to_file(self, file_to_save):
        file_to_save.write(",".join(str(number) for number in self.rule))
        file_to_save.write("\n")
        for i in range(len(self.shifts)):
            file_to_save.write(
                ",".join(str(number) for number in self.mat_transformations[i].ravel())
            )
            file_to_save.write("\n")
            file_to_save.write(",".join(str(number) for number in self.shifts[i]))
            file_to_save.write("\n")

    def read_from_file(self, file_to_read):
        self.rule = tuple(map(int, file_to_read.readline().split(",")))
        self.mat_transformations = []
        self.shifts = []
        for i in range(len(self.rule) - 1):
            line = file_to_read.readline()
            array_ = np.array(list(map(float, line.split(","))))
            # print(array_)
            self.mat_transformations.append(
                np.reshape(array_, (self.rule[i + 1], self.rule[i]))
            )
            line = file_to_read.readline()
            array_ = np.array(list(map(float, line.split(","))))
            self.shifts.append(array_)
