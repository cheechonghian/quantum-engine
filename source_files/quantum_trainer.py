# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 19:48:25 2020

@author: CHEE
"""

import time
import numpy as np
import matplotlib.pyplot as plt

class QuantumTrainer:
    """
    Attributes
    ----------

    """
    def __init__(self):
        """
        Parameters
        ----------

        """
        return None

    def config(self, select_optimiser="GD", select_loss="quad_loss", total_training_batch_size=100, max_training_set=10, learning_rate=0.5):
        """
        Parameters
        ----------

        """

        self.select_optimiser = select_optimiser
        self.select_loss = select_loss
        self.max_training_set = max_training_set
        self.learning_rate = learning_rate
        self.total_training_batch_size = total_training_batch_size
        self.optimiser_params = {"max_training_set": max_training_set,
                                 "learning_rate": learning_rate,
                                 "total_training_batch_size": total_training_batch_size}

    def inputs(self, teacher_model, quantum_computer):
        """
        Parameters
        ----------
        teacher_model: teacher_model class
        quantum_computer: quantum_computer class
        """
        self.teacher_model = teacher_model
        self.quantum_computer = quantum_computer


    def train(self):
        """
        Applies the vanilla gradient descent without any modifications.

        Parameters
        ----------
        teacher_model : teacher_model class from quantum_engine.py
        quantum_computer : quantum_computer class from quantum_engine.py
        """
        self.teacher_model.create_training_data(self.total_training_batch_size)

        loss_functions_selection = get_loss_functions()
        loss_functions = loss_functions_selection[self.select_loss]

        optimiser_model_selection = get_optimise_models()
        optimiser_model = optimiser_model_selection[self.select_optimiser]
        optimiser_model(self.teacher_model, self.quantum_computer, loss_functions, self.optimiser_params)


def get_optimise_models():
    dict_of_models = {"GD": standard_gradient_descent,
                      }
    return dict_of_models

def get_loss_functions():
    dict_of_models = {"quad_loss": quadratic_loss,
                      }
    return dict_of_models


def quadratic_loss(B, training_data, predict_y, predict_y_parameter_gradient, a):
    loss = np.sum(np.square(training_data["y_data"] - a * np.asarray(predict_y)))
    a_gradient = np.sum(-2*(training_data["y_data"] - a * np.asarray(predict_y)) * np.asarray(predict_y))
    loss_gradient_parameter_dict = {}
    for depth_iter in range(B.total_depth):

        loss_gradient_parameter_dict[depth_iter+1] = {}
        for qubit_iter in range(B.number_of_qubits):

            loss_gradient_parameter_dict[depth_iter+1][qubit_iter+1] = {}
            for rotate_gate_iter in range(3):

                parameter_gradient = []
                for data_iter in range(len(predict_y)):
                    parameter_gradient.append(predict_y_parameter_gradient[data_iter][depth_iter+1][qubit_iter+1][rotate_gate_iter+1])

                loss_gradient_parameter_dict[depth_iter+1][qubit_iter+1][rotate_gate_iter+1] = np.sum((training_data["y_data"] - a * np.asarray(predict_y)) * (-2*a*np.asarray(parameter_gradient)))

    loss_result = {"loss": loss,
                   "loss_gradient_parameter_dict": loss_gradient_parameter_dict,
                   "a_gradient": a_gradient}
    return loss_result


def standard_gradient_descent(teacher_model, quantum_computer, loss_functions, optimiser_params):
    """
    Apply the vanilla gradient descent without any modifications.

    Parameters
    ----------
    teacher_model : teacher_model class from quantum_engine.py
    quantum_computer : quantum_computer class from quantum_engine.py
    select_loss :
    max_training_set :
    learning_rate :
    """
    a = 1
    for training_iter in range(optimiser_params["max_training_set"]):
        predict_y = []
        predict_y_parameter_gradient = {}
        for data_iter in range(optimiser_params["total_training_batch_size"]):
            predict_result = quantum_computer.run_qc(teacher_model.training_data["x_data"][data_iter])
            predict_y.append(predict_result["output_data"])
            predict_y_parameter_gradient[data_iter] = predict_result["gradient_parameter_dict"]

        loss_result = loss_functions(quantum_computer.B, teacher_model.training_data, predict_y, predict_y_parameter_gradient, a)
        print(loss_result["loss"])
        a += - optimiser_params["learning_rate"] * loss_result['a_gradient']
        new_params = gradient_descent(quantum_computer.B, loss_result, optimiser_params)
        quantum_computer.update_params(new_params)
    print(loss_result["loss"])
    print(a)


def gradient_descent(B, loss_result, optimiser_params):
    loss_gradient_parameter_dict = loss_result["loss_gradient_parameter_dict"]
    old_params = B.parameter_dict
    new_params = {}
    for depth_iter in range(B.total_depth):
        new_params[depth_iter+1] = {}
        for qubit_iter in range(B.number_of_qubits):
            new_params[depth_iter+1][qubit_iter+1] = {}
            for rotate_gate_iter in range(3):
                new_params[depth_iter+1][qubit_iter+1][rotate_gate_iter+1] = old_params[depth_iter+1][qubit_iter+1][rotate_gate_iter+1] - optimiser_params["learning_rate"] * loss_gradient_parameter_dict[depth_iter+1][qubit_iter+1][rotate_gate_iter+1]

    return new_params

