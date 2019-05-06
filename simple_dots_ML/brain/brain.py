import sys
import random
import math
import time
import numpy as np


def applySeriesOfLinearTransformations(transformations, shifts, data, history=[]):
    result = data
    history.append(np.copy(data))
    for (transform, shift) in zip(transformations, shifts):
        # Compute A_MAT times data_VECT
        afine_ = np.dot(transform, result)
        # Compute data_VECT + add_VECT
        translation_ = np.add(afine_, shift)
        # Apply activation function
        result = np.tanh(translation_)
        # Save to history
        history.append(np.copy(result))
    return result


def calculate_full_gradient(history, transformations, shifts):
    gradient_shifts = []
    gradient_transformations = []
    range_ = len(shifts)
    for index in range(range_):
        # using the followin formula's to find new grarients:
        # f(a,b) = th(b + ax)
        # df = 1 - f ** 2
        # df/db = df
        # df/da = df * x

        # x:
        previos_value = history[index]

        # f(a,b):
        value = history[index + 1]

        # df = (df1,df2,df3,...)
        derivative = np.subtract(
            np.ones(len(value)),  # 1 - ...
            np.tensordot(np.diag(value), value, 1),  # f * f
        )

        # df = (df1, 0, 0, 0, ..
        #       0, df2, 0, 0, ..
        #       0, 0, df3, 0,..
        #            ...
        diagonal_derivative = np.diag(derivative)

        # diagonal df is already a new gradien for shift
        gradient_shifts.append(np.copy(diagonal_derivative))

        # gradient for transformation:
        # inside derivative is x , but
        # f_1 [a,b] [x]
        # f_2 [c,d] [y]
        #
        # df/dA =    [[x,y]
        #             [0,0]],
        #            [[0,0]
        #             [x,y]]
        #
        gradient_transformations.append(
            np.tensordot(diagonal_derivative, previos_value, 0)
        )
        for update_index in range(index):
            # now lets inductevely reconstuct gradients for previous parameters:
            # g(a,b,c) = th(a + b*f(c))
            # dg_i/dc = dg_i * b * df/dc
            #
            # dg/dc = diag_dg .*. (b * df/dc)
            # multiply by the current transformation
            gradient_transformations[update_index] = np.tensordot(
                transformations[index], gradient_transformations[update_index], 1
            )
            gradient_shifts[update_index] = np.tensordot(
                transformations[index], gradient_shifts[update_index], 1
            )
            # multiply by the current derivative
            gradient_transformations[update_index] = np.tensordot(
                diagonal_derivative, gradient_transformations[update_index], 1
            )
            gradient_shifts[update_index] = np.tensordot(
                diagonal_derivative, gradient_shifts[update_index], 1
            )

    return gradient_transformations, gradient_shifts


class Brain:
    def __init__(self, rule):
        self.rule = np.copy(rule)
        sizes = list(zip(rule[1:], rule[:-1]))
        x = lambda a: 2 * np.random.random(a) - 1
        self.mat_transformations = [x(size) for size in sizes]
        self.shifts = [x(size[0]) for size in sizes]
        self.history = []

    def copy(self):
        copied = Brain((1, 1))
        copied.rule = np.copy(self.rule)
        copied.mat_transformations = [np.copy(tr) for tr in self.mat_transformations]
        copied.shifts = [np.copy(tr) for tr in self.shifts]
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
        self.history = []
        return applySeriesOfLinearTransformations(
            self.mat_transformations, self.shifts, data, self.history
        )

    def compute_last_gradient(self):
        return calculate_full_gradient(
            self.history, self.mat_transformations, self.shifts
        )

    def apply_gradient_decent(self, delta_transformations, delta_shifts):
        for transform, delta in list(zip(self.transformations, delta_transformations)):
            transform = np.add(transform, delta)
        for shift, delta in list(zip(self.shifts, delta_shifts)):
            shift = np.add(shift, delta)

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
            self.mat_transformations.append(
                np.reshape(array_, (self.rule[i + 1], self.rule[i]))
            )
            line = file_to_read.readline()
            array_ = np.array(list(map(float, line.split(","))))
            self.shifts.append(array_)
