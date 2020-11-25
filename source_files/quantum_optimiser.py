"""
Created on Tue Nov 24 20:00:51 2020.

# -*- coding: utf-8 -*-

@author: CHEE
"""

import quantum_report as qrep
import numpy as np

class optimiser:
    def __init__(self):
        self.call = {"GD": standard_gradient_descent,
                     "SDG": stochastic_gradient_descent,
                     }


def standard_gradient_descent(teacher_model, quantum_model, loss_function, config_store):
    """
    Applies the vanilla gradient descent without any modifications.

    Parameters
    ----------
    teacher_model : teacher_model class from quantum_engine.py
    quantum_model : quantum_model class from quantum_engine.py
    loss_function : loss_function class from quantum_engine.py
    config_store : config_storage class local under quantum_trainer class from quantum_engine.py
    """
    my_report = qrep.quantum_report()

    training_batch = teacher_model.generate_training_batch()
    assert isinstance(training_batch, np.ndarray), "training_batch is not numpy array"

    for _ in range(config_store.max_training_sets):
        quantum_result = []
        quantum_gradient_plus = []
        quantum_gradient_minus = []

        for i in range(len(training_batch)):
            quantum_model.input_data(training_batch[i])
            output_state = quantum_model.run_model()
            assert isinstance(output_state, np.ndarray), "output_state is not numpy array"

            quantum_result.append(quantum_model.measure_Z(output_state))
            output_state_plus = quantum_model.run_model_plus()
            assert isinstance(output_state_plus, np.ndarray), "output_state_plus is not numpy array"

            quantum_gradient_plus.append(quantum_model.measure_Z(output_state_plus))
            output_state_minus = quantum_model.run_model_minus()
            assert isinstance(output_state_minus, np.ndarray), "output_state_minus is not numpy array"

            quantum_gradient_minus.append(quantum_model.measure_Z(output_state_minus))

        loss = loss_function.calculate_loss(np.asarray(quantum_result), training_batch)
        assert isinstance(loss, float), "loss is not a float"
        assert loss >= 0, "loss is negative."

        my_report.append_loss(loss)
        quantum_gradient_plus = np.sum(np.stack(quantum_gradient_plus), axis=0)
        assert isinstance(quantum_gradient_plus, np.ndarray), "quantum_gradient_plus is not numpy array"

        quantum_gradient_minus = np.sum(np.stack(quantum_gradient_minus), axis=0)
        assert isinstance(quantum_gradient_minus, np.ndarray), "quantum_gradient_minus is not numpy array"

        quantum_gradient = 0.5 * (quantum_gradient_plus - quantum_gradient_minus) # parameter shift rule
        assert isinstance(quantum_gradient, np.ndarray), "quantum_gradient is not numpy array"

        loss_gradient = loss_function.calculate_gradient(quantum_gradient)
        assert isinstance(loss_gradient, np.ndarray), "loss_gradient is not numpy array"

        old_quantum_params = quantum_model.get_params()
        assert isinstance(old_quantum_params, np.ndarray), "old_quantum_params is not numpy array"

        new_quantum_params = old_quantum_params - loss_gradient * config_store.learning_rate # Gradient descent
        assert isinstance(new_quantum_params, np.ndarray), "new_quantum_params is not numpy array"

        quantum_model.updata_params(new_quantum_params)

    return my_report #report


def stochastic_gradient_descent():
    return






