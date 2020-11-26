"""
Techer objects functions.

Universal Parameters
--------------------
t_params: dict
    ***To be filled***
"""

# from numpy.random import default_rng
import numpy as np


def generate_x_data(t_params):
    """
    Generate x coordinate given the parameters.

    Parameters
    ----------
    t_params : dict
        See top of code file.

    Outputs
    -------
    x_data : numpy array
        The x-coordinate data of the teacher model
    x_data_mean : numpy array
        Equals to x_data only when no noise is applied to x_data.
    """
    x_data_mean = np.linspace(t_params["x_low_lim"], t_params["x_upp_lim"], t_params["no_of_data_points"])

    if t_params["x_choose"] == "equal_spacing":
        x_data = x_data_mean

    return x_data, x_data_mean

    """  ## Noise Model ## (Future)
    elif t_params["x_choose"] == "equal_space_gaussian":
        x_data_mean = np.linspace(t_params["x_low_lim"], t_params["x_upp_lim"], t_params["no_of_data_points"])

        # Get the variance
        if len(t_params["x_g_var"]) == 1:
            x_data_var = np.ones((1, t_params["no_of_data_points"])) * t_params["x_g_var"][0]
        elif len(t_params["x_g_var"]) == t_params["no_of_data_points"]:
            x_data_var = t_params["x_g_var"]

        # Generate x_data with gaussian distribution
        rng = default_rng()
        x_data = rng.normal(x_data_mean, x_data_var)

    elif t_params["x_choose"] == "random_uniform":
        rng = default_rng()
        x_data = rng.uniform(t_params["x_low_lim"], t_params["x_upp_lim"], t_params["no_of_data_points"])
        x_data_mean = x_data
    """


def generate_data(t_params):
    """
    Generate (x,y) coordinate data.

    Parameters
    ----------
    t_params: dict
        See top of code file.
    """
    x_data, x_data_mean = generate_x_data(t_params)
    model_selection = get_models()
    model = model_selection[t_params["select_optimiser"]]
    gen_model = model(x_data, t_params)
    model_name = gen_model.name
    y_data_mean = gen_model.gen_y
    y1_data_mean = gen_model.gen_y1d

    if t_params["y_choose"] == "None":
        y_data = y_data_mean
        y1_data = y1_data_mean

    data = {"x_data": x_data,
            "x_data_mean": x_data_mean,
            "y_data": y_data,
            "y_data_mean": y_data_mean,
            "y1_data": y1_data,
            "y1_data_mean": y1_data_mean,
            "model_name": model_name
            }

    return data

    """  ## Noise Model ## (Future)
    elif t_params["y_choose"] == "gaussian":
        y_data_mean = t_params["b"] * x_data_mean + t_params["a"]

        # Get the variance
        if len(t_params["y_g_var"]) == 1:
            y_data_var = np.ones((1, t_params["no_of_data_points"])) * t_params["y_g_var"][0]
        elif len(t_params["y_g_var"]) == t_params["no_of_data_points"]:
            y_data_var = t_params["y_g_var"]

        rng = default_rng()
        y_data = rng.normal(y_data_mean, y_data_var)
    """


"""
Parameters
----------
x : numpy array
    The x coordinates, x_data (or x_data_mean)
t_params : dict
    Contains teacher model parameters
"""


def get_models():
    dict_of_models = {"lin": linear,
                      "quad": quadratic,
                      }
    return dict_of_models


class linear:
    def __init__(self, x, t_params):
        self.gen_y = t_params["b"] * x + t_params["a"]
        self.gen_y1d = t_params["b"] * np.ones((1, len(x)))
        self.name = r"$bx+a$, $b=${:.1f}, $a=${:.1f}".format(t_params["b"], t_params["a"])


class quadratic:
    def __init__(self, x, t_params):
        self.gen_y = t_params["c"] * np.power(x, 2) + t_params["b"] * x + t_params["a"]
        self.gen_y1d = 2 * t_params["c"] * x + t_params["b"]
        self.name = r"$cx^2+bx+a$, $c=${:.1f}, $b=${:.1f}, $a=${:.1f}".format(t_params["c"], t_params["b"], t_params["a"])