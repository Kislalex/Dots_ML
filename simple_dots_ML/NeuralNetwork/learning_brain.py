# from brain import Brain
import numpy as np
from multiprocessing import Pool

Eps = 0.0001


def compute_test_error(brain, inputs_data, outputs_data):
    error = 0.0
    for input_, output_ in zip(inputs_data, outputs_data):
        # compute approx result
        brain_output = brain.compute(input_)
        # compute multipiers - d(x-a) ** 2 = 2*(a-x)*dx
        df = output_ - brain_output[0]
        # update error
        current_error = np.linalg.norm(df)
        error += current_error
    return error


def apply_gradient_decent(brain, inputs_data, outputs_data, dt):
    dt *= 0.3 / len(inputs_data)
    for input_, output_ in zip(inputs_data, outputs_data):
        # compute approx result
        brain_output = brain.compute(input_)
        # compute multipiers - d(x-a) ** 2 = 2*(a-x)*dx
        df = brain_output[0] - output_
        # update gradient
        brain.compute_gradient(df, input_)
        brain.apply_gradient(-dt)


def find_the_best_brain_update(brain, inputs_data, outputs_data):
    min_error = 100000000000.0
    best_brain = brain.copy()
    for x in range(-10, 5):
        dt = 2.0 ** x
        new_brain = brain.copy()
        apply_gradient_decent(new_brain, inputs_data, outputs_data, dt)
        current_error = compute_test_error(new_brain, inputs_data, outputs_data)
        if current_error < min_error:
            min_error = current_error
            best_brain = new_brain.copy()
    return best_brain


def learning_brain_by_g_d(
    brain, inputs_data, outputs_data, step_number=100, subspace=None
):
    previous_error = 1
    delta_error = 1
    for step in range(step_number):
        brain = find_the_best_brain_update(brain, inputs_data, outputs_data)
        error = compute_test_error(brain, inputs_data, outputs_data)
        delta_error = previous_error - error
        previous_error = error
        print(inputs_data[0], outputs_data[0], brain.compute(inputs_data[0]))
        print("step     : ", step)
        print("total err: ", error)
        print("del   err: ", delta_error)
        print("dlog  err: ", delta_error / error)
