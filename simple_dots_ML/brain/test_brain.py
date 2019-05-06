from brain import Brain
import numpy as np
import pytest


Eps = 0.0001


def test_empty_brain():
    rule = (1,)
    test_brain = Brain(rule)
    y = test_brain.signal(np.array([1]))
    assert test_brain.shifts == []
    assert y[0] == pytest.approx(1, Eps)


def test_number_tivial_nonzero_brain():
    rule = (1, 1)
    # create a basic brain f(x) = th(a + 0 * x)
    test_shift = 0.12345
    test_brain = Brain(rule)
    test_brain.mat_transformations[0][0] = 0
    test_brain.shifts[0] = test_shift
    y = test_brain.signal(np.array([1]))
    assert len(test_brain.shifts) == 1
    assert y[0] == pytest.approx(np.tanh(test_shift), Eps)


def test_matrix_sizes_agree():
    rule = (3, 10, 10, 5)
    test_brain = Brain(rule)
    y = test_brain.signal(np.array([1, 2, 3]))
    assert len(y) == 5


def test_simple_brain_returns_expected_values():
    rule = (1, 1)
    # create a basic brain f(x) = th(a + 0 * x)
    test_shift = 0.12345
    test_brain = Brain(rule)
    test_brain.mat_transformations[0][0] = 0
    test_brain.shifts[0] = test_shift
    y = test_brain.signal(np.array([1]))[0]
    dA, db = test_brain.compute_last_gradient()
    assert len(dA) == 1
    assert dA[0].shape == (1, 1, 1)
    assert db[0].shape == (1, 1)
    assert dA[0][0][0][0] == db[0][0][0]


def test_brain_returns_correct_shapes():
    rule = (3, 10, 20, 4, 5)
    # create a basic brain f(x) = th(a + 0 * x)
    test_data = np.array([1, 2, 3])
    test_brain = Brain(rule)
    y = test_brain.signal(test_data)
    dA, db = test_brain.compute_last_gradient()
    assert len(dA) == 4
    for i in range(len(dA)):
        assert dA[i].shape == (5,) + (rule[i + 1], rule[i])
        assert db[i].shape == (5,) + (rule[i + 1],)


def test_brain_returns_correctly_last_gradients():
    rule = (3, 4, 5)
    # create a basic brain f(x) = th(a + 0 * x)
    test_data = np.array([1, 2, 3])
    test_brain = Brain(rule)
    y = test_brain.signal(test_data)
    dA, db = test_brain.compute_last_gradient()
    np.testing.assert_array_equal(db[1][0], np.array([1 - y[0] * y[0], 0, 0, 0, 0]))


def test_brain_compute_gradient_formulas_correct():
    rule = (2, 2, 2)
    #
    # g_1 = th( |v_1| + |a_11 a_12|                              )
    #           |   |               * th(w_1 + |b_11 b_12||x_1|)
    #           |   |               * th(w_2 + |b_21 b_22||x_2|)
    # g_2 = th( |v_2| + |a_21 a_22|                              )
    #
    # assign variables
    a_mat = np.array([[0.1, 0.2], [0.3, 0.4]])
    b_mat = np.array([[0.6, 0.5], [0.4, 0.3]])
    v_vect = np.array([-0.1, -0.2])
    w_vect = np.array([-0.3, 0.4])
    x_vect = np.array([0.5, 0.2])
    # Build correspondent Brain
    test_brain = Brain(rule)
    test_brain.mat_transformations[0] = b_mat
    test_brain.mat_transformations[1] = a_mat
    test_brain.shifts[0] = w_vect
    test_brain.shifts[1] = v_vect
    g_vect = test_brain.signal(x_vect)
    # compute the signal value by hand:
    h_1 = np.tanh(w_vect[0] + b_mat[0][0] * x_vect[0] + b_mat[0][1] * x_vect[1])
    h_2 = np.tanh(w_vect[1] + b_mat[1][0] * x_vect[0] + b_mat[1][1] * x_vect[1])
    g_1 = np.tanh(v_vect[0] + a_mat[0][0] * h_1 + a_mat[0][1] * h_2)
    g_2 = np.tanh(v_vect[1] + a_mat[1][0] * h_1 + a_mat[1][1] * h_2)
    # check the result:
    assert g_vect[0] == pytest.approx(g_1)
    assert g_vect[1] == pytest.approx(g_2)
    dA, db = test_brain.compute_last_gradient()
    # now computing the gradients

    # dg_dv matrix
    dg1_dv1 = 1 - g_1 ** 2
    dg1_dv2 = 0
    dg2_dv2 = 1 - g_2 ** 2
    dg2_dv1 = 0
    dg_dv = np.array([[dg1_dv1, dg1_dv2], [dg2_dv1, dg2_dv2]])

    # dg_dA matrix
    dg1_da11 = (1 - g_1 ** 2) * h_1
    dg1_da12 = (1 - g_1 ** 2) * h_2
    dg1_da21 = 0
    dg1_da22 = 0
    dg2_da11 = 0
    dg2_da12 = 0
    dg2_da21 = (1 - g_2 ** 2) * h_1
    dg2_da22 = (1 - g_2 ** 2) * h_2
    dg_da = np.array(
        [
            [[dg1_da11, dg1_da12], [dg1_da21, dg1_da22]],
            [[dg2_da11, dg2_da12], [dg2_da21, dg2_da22]],
        ]
    )

    # dg_dw matrix
    dg1_dw1 = (1 - g_1 ** 2) * a_mat[0][0] * (1 - h_1 ** 2)
    dg1_dw2 = (1 - g_1 ** 2) * a_mat[0][1] * (1 - h_2 ** 2)
    dg2_dw1 = (1 - g_2 ** 2) * a_mat[1][0] * (1 - h_1 ** 2)
    dg2_dw2 = (1 - g_2 ** 2) * a_mat[1][1] * (1 - h_2 ** 2)
    dg_dw = np.array([[dg1_dw1, dg1_dw2], [dg2_dw1, dg2_dw2]])

    # dg_dB matrix
    dg1_db11 = (1 - g_1 ** 2) * a_mat[0][0] * (1 - h_1 ** 2) * x_vect[0]
    dg1_db12 = (1 - g_1 ** 2) * a_mat[0][0] * (1 - h_1 ** 2) * x_vect[1]
    dg1_db21 = (1 - g_1 ** 2) * a_mat[0][1] * (1 - h_2 ** 2) * x_vect[0]
    dg1_db22 = (1 - g_1 ** 2) * a_mat[0][1] * (1 - h_2 ** 2) * x_vect[1]
    dg2_db11 = (1 - g_2 ** 2) * a_mat[1][0] * (1 - h_1 ** 2) * x_vect[0]
    dg2_db12 = (1 - g_2 ** 2) * a_mat[1][0] * (1 - h_1 ** 2) * x_vect[1]
    dg2_db21 = (1 - g_2 ** 2) * a_mat[1][1] * (1 - h_2 ** 2) * x_vect[0]
    dg2_db22 = (1 - g_2 ** 2) * a_mat[1][1] * (1 - h_2 ** 2) * x_vect[1]
    dg_db = np.array(
        [
            [[dg1_db11, dg1_db12], [dg1_db21, dg1_db22]],
            [[dg2_db11, dg2_db12], [dg2_db21, dg2_db22]],
        ]
    )

    np.testing.assert_allclose(dg_dv, db[1])
    np.testing.assert_allclose(dg_da, dA[1])
    np.testing.assert_allclose(dg_dw, db[0])
    np.testing.assert_allclose(dg_db, dA[0])
