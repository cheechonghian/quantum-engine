"""
Techer objects functions.

Universal Parameters
--------------------
teacher_params: dict
    ***To be filled***
"""

# from numpy.random import default_rng
import numpy as np


def generate_x_data(teacher_params, is_train_data, is_test_data):
    """
    Generate x coordinate given the parameters.

    Parameters
    ----------
    teacher_params : dict
        See top of code file.

    Outputs
    -------
    x_data : numpy array
        The x-coordinate data of the teacher model
    x_data_mean : numpy array
        Equals to x_data only when no noise is applied to x_data.
    """
    if is_train_data is True:
        x_data_mean = np.linspace(teacher_params["x_low_lim"], teacher_params["x_upp_lim"], teacher_params["no_of_train_points"])
    elif is_test_data is True:
        x_data_mean = np.linspace(teacher_params["x_low_lim"], teacher_params["x_upp_lim"], teacher_params["no_of_test_points"])

    if teacher_params["x_choose"] == "equal_spacing":
        x_data = x_data_mean

    return x_data, x_data_mean

    """  ## Noise Model ## (Future)
    elif teacher_params["x_choose"] == "equal_space_gaussian":
        x_data_mean = np.linspace(teacher_params["x_low_lim"], teacher_params["x_upp_lim"], teacher_params["no_of_train_points"])

        # Get the variance
        if len(teacher_params["x_g_var"]) == 1:
            x_data_var = np.ones((1, teacher_params["no_of_train_points"])) * teacher_params["x_g_var"][0]
        elif len(teacher_params["x_g_var"]) == teacher_params["no_of_train_points"]:
            x_data_var = teacher_params["x_g_var"]

        # Generate x_data with gaussian distribution
        rng = default_rng()
        x_data = rng.normal(x_data_mean, x_data_var)

    elif teacher_params["x_choose"] == "random_uniform":
        rng = default_rng()
        x_data = rng.uniform(teacher_params["x_low_lim"], teacher_params["x_upp_lim"], teacher_params["no_of_train_points"])
        x_data_mean = x_data
    """


def generate_data(teacher_params, is_train_data=False, is_test_data=False):
    """
    Generate (x,y) coordinate data.

    Parameters
    ----------
    teacher_params: dict
        See top of code file.
    """
    assert (is_train_data ^ is_test_data) is True, "'is_train_data' and 'is_test_data' cannot be simultaneously True or False."

    x_data, x_data_mean = generate_x_data(teacher_params, is_train_data, is_test_data)
    model_selection = get_models()
    model = model_selection[teacher_params["select_model"]]
    gen_model = model(x_data, teacher_params)
    model_name = gen_model.name
    y_data_mean = gen_model.gen_y
    y1_data_mean = gen_model.gen_y1d

    if teacher_params["y_choose"] == "None":
        y_data = y_data_mean
        y1_data = y1_data_mean

        data = {"x_data": x_data,
                "x_data_mean": x_data_mean,
                "y_data": y_data,
                "y_data_mean": y_data_mean,
                "y1_data": y1_data,
                "y1_data_mean": y1_data_mean,
                "model_name": model_name,
                }
        if is_train_data is True:
            data["data_check"] = "train"
        elif is_test_data is True:
            data["data_check"] = "test"

        return data

    """  ## Noise Model ## (Future)
    elif teacher_params["y_choose"] == "gaussian":
        y_data_mean = teacher_params["b"] * x_data_mean + teacher_params["a"]

        # Get the variance
        if len(teacher_params["y_g_var"]) == 1:
            y_data_var = np.ones((1, teacher_params["no_of_train_points"])) * teacher_params["y_g_var"][0]
        elif len(teacher_params["y_g_var"]) == teacher_params["no_of_train_points"]:
            y_data_var = teacher_params["y_g_var"]

        rng = default_rng()
        y_data = rng.normal(y_data_mean, y_data_var)
    """


"""
Parameters
----------
x : numpy array
    The x coordinates, x_data (or x_data_mean)
teacher_params : dict
    Contains teacher model parameters
"""


def get_models():
    dict_of_models = {"lin": linear,
                      "quad": quadratic,
                      }
    return dict_of_models


class linear:
    def __init__(self, x, teacher_params):
        self.gen_y = teacher_params["b"] * x + teacher_params["a"]
        self.gen_y1d = teacher_params["b"] * np.ones((1, len(x)))
        self.name = r"$bx+a$, $b=${:.1f}, $a=${:.1f}".format(teacher_params["b"], teacher_params["a"])


class quadratic:
    def __init__(self, x, teacher_params):
        self.gen_y = teacher_params["c"] * np.power(x, 2) + teacher_params["b"] * x + teacher_params["a"]
        self.gen_y1d = 2 * teacher_params["c"] * x + teacher_params["b"]
        self.name = r"$cx^2+bx+a$, $c=${:.1f}, $b=${:.1f}, $a=${:.1f}".format(teacher_params["c"], teacher_params["b"], teacher_params["a"])