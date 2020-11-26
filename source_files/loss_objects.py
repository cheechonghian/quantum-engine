"""
loss objects functions.

Universal Parameters
--------------------
quantum_result_data: dict
    ***To be filled***

teacher_data: dict
    Either 'teacher_model.train_data' or 'teacher_model.test_data'

loss_params: dict
    'loss_params' attribute under 'loss_function' class from quantum_engine.py
"""

# from numpy.random import default_rng
import numpy as np


def calculate_loss(quantum_result_data, teacher_data, loss_params):
    model_selection = get_loss_models()
    loss_model = model_selection[loss_params["select_loss"]]
    generate_loss_info = loss_model(quantum_result_data, teacher_data, loss_params)
    loss_info = {"loss": generate_loss_info.loss,
                 "loss_gradient": generate_loss_info.loss_gradient,
                 }
    return loss_info


def get_loss_models():
    dict_of_models = {"quad_loss": quadratic_loss,
                      "sobolev": sobolev,
                      }
    return dict_of_models


class quadratic_loss:
    def __init__(self, quantum_result_data, teacher_data, loss_params):
        self.loss = np.sum(np.square(quantum_result_data["y_data"] - teacher_data["y_data"]))
        self.loss_gradient = np.dot(quantum_result_data["grad"], 2 * (quantum_result_data["y_data"] - teacher_data["y_data"]))

class sobolev:
    def __init__(self, quantum_result_data, teacher_data, loss_params):
        return