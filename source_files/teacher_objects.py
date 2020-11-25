"""
Techer objects functions.

Universal Parameters
--------------------
config_params, noise_config_params: dict
    ***To be filled***
"""

from numpy.random import default_rng
import numpy as np


def get_models_dict():
    dict_of_models = {"lin": linear_model,
                      "quad": quadratic_model,
                      }
    return dict_of_models


def generate_x_data(config_params, noise_config_params):
    """
    Generate x coordinate given the parameters.

    Parameters
    ----------
    config_params, noise_config_params : dict
        See top of code file.

    Outputs
    -------
    x_data : numpy array
        The x-coordinate data of the teacher model
    x_data_mean : numpy array
        Equals to x_data only when no noise is applied to x_data.
    """
    # "x_choose" switch statements
    if config_params["x_choose"] == "equal_spacing":
        x_data = np.linspace(config_params["x_low_lim"], config_params["x_upp_lim"], config_params["no_of_points"])
        x_data_mean = x_data

    elif config_params["x_choose"] == "equal_space_gaussian":
        x_data_mean = np.linspace(config_params["x_low_lim"], config_params["x_upp_lim"], config_params["no_of_points"])

        # Get the variance
        if len(noise_config_params["x_g_var"]) == 1:
            x_data_var = np.ones((1, config_params["no_of_points"])) * noise_config_params["x_g_var"][0]
        elif len(noise_config_params["x_g_var"]) == config_params["no_of_points"]:
            x_data_var = noise_config_params["x_g_var"]

        # Generate x_data with gaussian distribution
        rng = default_rng()
        x_data = rng.normal(x_data_mean, x_data_var)

    elif config_params["x_choose"] == "random_uniform":
        rng = default_rng()
        x_data = rng.uniform(config_params["x_low_lim"], config_params["x_upp_lim"], config_params["no_of_points"])
        x_data_mean = x_data

    return x_data, x_data_mean


def generate_data(config_params, noise_config_params):
    """
    Generate (x,y) coordinate data.

    Parameters
    ----------
    config_params, noise_config_params : dict
        See top of code file.
    """
    x_data, x_data_mean = generate_x_data(config_params, noise_config_params)
    model_selection = get_models_dict()

    # "y_choose" switch statements
    if config_params["y_choose"] == "None":
        model = model_selection[config_params["select_model"]]
        y_data = model(x_data, config_params)

        model_gradient = model_selection[config_params["select_model"] + "_gradient"]

        y1_data = model_gradient(x_data, config_params)

    elif config_params["y_choose"] == "gaussian":
        y_data_mean = config_params["b"] * x_data_mean + config_params["a"]

        # Get the variance
        if len(noise_config_params["y_g_var"]) == 1:
            y_data_var = np.ones((1, config_params["no_of_points"])) * noise_config_params["y_g_var"][0]
        elif len(noise_config_params["y_g_var"]) == config_params["no_of_points"]:
            y_data_var = noise_config_params["y_g_var"]

        rng = default_rng()
        y_data = rng.normal(y_data_mean, y_data_var)

    data = {"x_data": x_data,
            "x_data_mean": x_data_mean,
            "y_data": y_data,
            "y_data_mean": y_data_mean,
            "y1_data": y1_data,
            "y1_data_mean": y1_data_mean,
            }
    return data


# The models below returns y coordinates as numpy arrays
"""
Parameters
----------
x : numpy array
    The x coordinates, x_data (or x_data_mean)
**params : dict
    Contains the model parameters
"""
def linear_model(x, params):
    # Linear Model: b*x+a.
    return params["b"] * x + params["a"]


#  *** NEED TO FIX THIS ***
def linear_model_gradient(x1, params, noise_params):
    # Linear Model gradient: b.
    return params["b"] * x1

def quadratic_model(x, params):
    # Quadratic Model: c*x^2+b*x+a.
    return params["c"] * np.power(x, 2) + params["b"] * x + params["a"]
