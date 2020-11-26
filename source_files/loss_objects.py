"""
loss objects functions.

Universal Parameters
--------------------
qdata: dict
    ***To be filled***

tmodel: class
    'teacher_model' class from quantum_engine.py

l_params: dict
    'config_parameters' attribute under 'loss_function' class from quantum_engine.py
"""

# from numpy.random import default_rng
import numpy as np


def calculate_loss(qdata, tmodel, l_params):
    model_selection = get_loss_models()
    loss_model = model_selection[l_params["select_loss"]]
    generate_loss_info = loss_model(qdata, tmodel, l_params)
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
    def __init__(self, qdata, tmodel, l_params):
        self.loss = np.sum(np.square(qdata["y_adata"] - tmodel.y_tdata))
        self.loss_gradient = qdata.dot(qdata["grad"], 2 * (qdata["y_adata"] - tmodel.y_tdata))

class sobolev:
    def __init__(self, qdata, tmodel, l_params):
        return