import numpy as np
import matplotlib.pyplot as plt


class TeacherModel:
    """
    Define a teacher model that will be used as a target function for the quantum model to approximate to.

    Attributes
    ----------
    """

    def __init__(self):
        pass

    def config(self, select_model="lin", x_lower_limit=0.0, number_of_points=10, x_upper_limit=1.0, a=1.0, b=1.0, c=1.0, d=1.0):
        """
        Apply configuration settings to teacher model.

        Parameters
        ----------
        select_model : str
            The teacher model type.
        x_lower_limit, x_upper_limit : float
            Lower and upper limits of the independent variable x (input) of the teacher model.
        number_of_points : int
            Number of data points.
        a, b, c, d : float
            Required model parameters.
        """

        self.teacher_params = {"select_model": select_model,
                               "x_low_lim": x_lower_limit,
                               "x_upp_lim": x_upper_limit,
                               "number_of_points": number_of_points,
                               "a": a,
                               "b": b,
                               "c": c,
                               "d": d,
                               }
        self.plot_data = generate_data(self.teacher_params)

    def create_training_data(self, number_of_points=10,):
        self.teacher_params["number_of_points"] = number_of_points
        self.training_data = generate_data(self.teacher_params)

    def plot_model(self):

        fig, ax = plt.subplots(2, 1, dpi=100, sharex=True)
        ax[0].scatter(self.plot_data["x_data"], self.plot_data["y_data"])
        ax[0].set_ylabel(r"$f(x)$")
        ax[1].scatter(self.plot_data["x_data"], self.plot_data["y1d_data"])
        ax[1].set_ylabel(r"$f^{\prime}(x)$")
        ax[1].set_xlabel(r"$x$")
        fig.subplots_adjust(hspace=0.1)
        fig.suptitle(self.plot_data["model_name"])


def generate_data(teacher_params):
    """
    Generate (x,y) coordinate data.

    Parameters
    ----------
    teacher_params: dict
        See top of code file.
    """
    x_data = np.linspace(teacher_params["x_low_lim"], teacher_params["x_upp_lim"], teacher_params["number_of_points"])
    model_selection = get_models()
    model = model_selection[teacher_params["select_model"]]
    gen_model = model(x_data, teacher_params)
    model_name = gen_model.name
    y_data = gen_model.gen_y
    y1d_data = gen_model.gen_y1d

    data = {"x_data": x_data,
            "y_data": y_data,
            "y1d_data": y1d_data,
            "model_name": model_name,
            }

    return data


def get_models():
    dict_of_models = {"lin": linear,
                      "quad": quadratic,
                      }
    return dict_of_models


class linear:
    def __init__(self, x, teacher_params):
        self.gen_y = teacher_params["b"] * x + teacher_params["a"]
        self.gen_y1d = teacher_params["b"] * np.ones((1, len(x)))
        self.name = "Teacher Linear Model" + r"$bx+a$, $b=${:.1f}, $a=${:.1f}".format(teacher_params["b"], teacher_params["a"])


class quadratic:
    def __init__(self, x, teacher_params):
        self.gen_y = teacher_params["c"] * np.power(x, 2) + teacher_params["b"] * x + teacher_params["a"]
        self.gen_y1d = 2 * teacher_params["c"] * x + teacher_params["b"]
        self.name = "Teacher Quadratic Model\n" + r"$cx^2+bx+a$, $c=${:.1f}, $b=${:.1f}, $a=${:.1f}".format(teacher_params["c"], teacher_params["b"], teacher_params["a"])
