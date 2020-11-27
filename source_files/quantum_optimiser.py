"""
Created on Tue Nov 24 20:00:51 2020.

# -*- coding: utf-8 -*-

@author: CHEE
"""

import quantum_report as qrep
import numpy as np


def optimiser(teacher_model, quantum_computer, loss_function, trainer_params):
    """
    Applies the vanilla gradient descent without any modifications.

    Parameters
    ----------
    teacher_model : teacher_model class from quantum_engine.py
    quantum_computer : quantum_computer class from quantum_engine.py
    loss_function : loss_function class from quantum_engine.py
    trainer_params : config_storage class local under quantum_trainer class from quantum_engine.py
    """
    if teacher_model.is_train_data_gen is False:
        print("Training data is not generated. Please use 'teacher_model.generate_training_batch() inputting it into me.'")
        return None

    my_report = qrep.quantum_report()
    model_selection = get_optimise_models()
    model = model_selection[trainer_params["select_model"]]
    model(teacher_model, quantum_computer, loss_function, trainer_params, my_report)

    return my_report


def get_optimise_models():
    dict_of_models = {"GD": standard_gradient_descent,
                      "SDG": stochastic_gradient_descent,
                      }
    return dict_of_models


def standard_gradient_descent(teacher_model, quantum_computer, loss_function, trainer_params, my_report):
    """
    Applies the vanilla gradient descent without any modifications.

    Parameters
    ----------
    teacher_model : 'teacher_model' class from quantum_engine.py
    quantum_computer : 'quantum_computer' class from quantum_engine.py
    loss_function : 'loss_function' class from quantum_engine.py
    trainer_params : 'trainer_config' under 'quantum_trainer' class from quantum_engine.py
    my_report: quantum_report from quantum_report.py
    """

    for _ in range(trainer_params["max_training_sets"]):
        for i in range(len(teacher_model.teacher_params["no_of_train_points"])):
            quantum_computer.input_data(teacher_model.train_data)
            quantum_result_data = quantum_computer.run_model()

        training_loss_info = loss_function.calculate_loss_info(quantum_result_data, teacher_model.train_data)
        my_report.store_loss_info(training_loss_info)
        new_quantum_params = gradient_descent(quantum_computer, trainer_params, training_loss_info)
        quantum_computer.update_params(new_quantum_params)


def stochastic_gradient_descent():

    return

## Helper Functions ##

def gradient_descent(quantum_computer, trainer_params, training_loss_info):
    # Gradient descent
    new_quantum_params = quantum_computer.get_current_params() - training_loss_info['loss_gradient'] * trainer_params["learning_rate"]
    assert isinstance(new_quantum_params, np.ndarray), "new_quantum_params is not a numpy array."
    assert isinstance(new_quantum_params.ndim != 1), "new_quantum_params is not a 1D array."

    return new_quantum_params