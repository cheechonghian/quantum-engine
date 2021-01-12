"""
Created by Chee Chong Hian.

Below contains Teacher Model operations for Quantum Circuit Learning.
"""
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt

class TeacherModel:
    """
    Define a teacher model that will be used as a target function for the quantum model to approximate to.

    Attributes
    ----------
    teacher_params : dict
        Contains all parameters needed for data generation.
    training_data : dict
        Contains x, y and gradient data of the model for training.
    """

    def __init__(self, verbose=True):
        self.verbose = verbose
        if verbose is True:
            print("Select TeacherModel:\n")
            print("1. 'lin'  -> bx+c")
            print("2. 'quad' -> ax^2+bx+c")
            print("3. 'sin'  -> a*sin(bx+c)")
            print("4. 'cos'  -> a*cos(bx+c)")
            print("5. 'tan'  -> a*tan(bx+c)")
            print("6. 'exp'  -> a*exp(bx+c)")
            print("7. 'log'  -> a*log(|bx+c|)\n")
            print("To select: use TeacherModel().config(select_model='_<your_selection_here>_') \n \n")
            print("Config Setting Available:\n")
            print("* select_model [str]: (See Above)")
            print("* x_lower_limit [int]: The lower bound of x data coordinate (beware of singularities)")
            print("* x_upper_limit [int]: The upper bound of x data coordinate (beware of singularities)")
            print("* number_of_points [int]: The number of data points that you want to use for training")
            print("* a,b,c [float]: Model Parameters")

    def config(self, select_model="lin", x_lower_limit=-0.99, x_upper_limit=0.99, number_of_points=10, a=1.0, b=1.0, c=1.0, split=None):
        """
        Apply configuration settings to teacher model and plot the teacher model.

        Parameters
        ----------
        select_model : str
            The teacher model type.
        x_lower_limit, x_upper_limit : float
            Lower and upper limits of the independent variable x (input) of the teacher model.
        number_of_points : int
            The number of data points that you want to use for training
        a, b, c: float
            Required model parameters.
        split: None or float
            None: No spliting, train and test data will be equal to main data
            The composition ratio of main data: train/test
            Data are randomly picked.
            closer to 1 -> more train
            closer to 0 -> more test
            equal to 0.5 -> equal split of main data
        """

        self.teacher_params = {"select_model": select_model,
                               "x_low_lim": x_lower_limit,
                               "x_upp_lim": x_upper_limit,
                               "number_of_points": number_of_points,
                               "a": a,
                               "b": b,
                               "c": c,
                               "split": split
                               }
        self.main_data = self.generate_data()
        print("Note: Main Data is initialized")

        ### To be DEFUNCT: Warning other code uses training data ####
        #self.training_data = self.generate_data(self.teacher_params)
        #self.testing_data = self.generate_data(self.teacher_params)

        self.split_data()

        # For plotting purposes
        x_size = x_upper_limit - x_lower_limit
        x_pad = x_size * 0.05
        self.teacher_params["x_pad"] = x_pad

        self.teacher_params["y_upp_lim"] = np.amax(self.main_data["y_data"])
        self.teacher_params["y_low_lim"] = np.amin(self.main_data["y_data"])
        y_size = self.teacher_params["y_upp_lim"] - self.teacher_params["y_low_lim"]
        y_pad = y_size * 0.1
        self.teacher_params["y_pad"] = y_pad

        self.teacher_params["y1d_upp_lim"] = np.amax(self.main_data["y1d_data"])
        self.teacher_params["y1d_low_lim"] = np.amin(self.main_data["y1d_data"])
        y1d_size = self.teacher_params["y1d_upp_lim"] - self.teacher_params["y1d_low_lim"]
        y1d_pad = y1d_size * 0.1
        self.teacher_params["y1d_pad"] = y1d_pad

    def plot_model(self, type="main", join_points=False):
        fig, ax = plt.subplots(2, 1, dpi=100, sharex=True)
        if type == "main":
            data = self.main_data
            join_points = True
        elif type == "train":
            data = self.training_data
        elif type == "test":
            data = self.testing_data

        if join_points is False:
            ax[0].scatter(data["x_data"], data["y_data"])
        elif join_points is True:
            ax[0].plot(data["x_data"], data["y_data"])

        ax[0].set_ylabel(r"Teacher $f(x)$")
        ax[0].set_xlim(self.teacher_params["x_low_lim"]-self.teacher_params["x_pad"], self.teacher_params["x_upp_lim"]+self.teacher_params["x_pad"])
        ax[0].set_ylim(self.teacher_params["y_low_lim"]-self.teacher_params["y_pad"], self.teacher_params["y_upp_lim"]+self.teacher_params["y_pad"])

        if join_points is False:
            ax[1].scatter(data["x_data"], data["y1d_data"])
        elif join_points is True:
            ax[1].plot(data["x_data"], data["y1d_data"])

        ax[1].set_ylabel(r"Gradient $f^{\prime}(x)$")
        ax[1].set_xlabel(r"$x$")
        ax[1].set_ylim(self.teacher_params["y1d_low_lim"]-self.teacher_params["y1d_pad"], self.teacher_params["y1d_upp_lim"]+self.teacher_params["y1d_pad"])
        fig.subplots_adjust(hspace=0.1)
        fig.suptitle(data["model_name"], y=1.04)

    def generate_data(self):
        """
        Generate (x,y) coordinate data.

        Parameters
        ----------
        teacher_params: dict
            See top of code file.
        """
        x_data = np.linspace(self.teacher_params["x_low_lim"], self.teacher_params["x_upp_lim"], self.teacher_params["number_of_points"])
        model_selection = get_models()
        model = model_selection[self.teacher_params["select_model"]]
        gen_model = model(x_data, self.teacher_params)
        model_name = gen_model.name
        y_data = gen_model.gen_y
        y1d_data = gen_model.gen_y1d

        data = {"x_data": x_data,
                "y_data": y_data,
                "y1d_data": y1d_data,
                "model_name": model_name,
                }

        return data

    def split_data(self):
        # To generate training and testing data
        if self.teacher_params["split"] is None:
            self.training_data = self.main_data
            self.testing_data = self.main_data
            print("Note: No data splitting was done. Training Data and Testing Data are equal to Main Data")
        else:
            self.num_of_point_train = round(self.teacher_params["number_of_points"] * self.teacher_params["split"])
            self.num_of_point_test = self.teacher_params["number_of_points"] - self.num_of_point_train
            print(self.num_of_point_train)
            print(self.num_of_point_test)
            rng = rand.default_rng()
            train_data_index = rng.choice(self.teacher_params["number_of_points"], self.num_of_point_train, replace=False)
            test_data_index = np.delete(np.arange(self.teacher_params["number_of_points"]), train_data_index)

            self.training_data = {"x_data": self.main_data["x_data"][train_data_index],
                                  "y_data": self.main_data["y_data"][train_data_index],
                                  "y1d_data": self.main_data["y1d_data"][train_data_index],
                                  "model_name": self.main_data["model_name"] + "\n Training Data",
                                  }

            self.testing_data = {"x_data": self.main_data["x_data"][test_data_index],
                                 "y_data": self.main_data["y_data"][test_data_index],
                                 "y1d_data": self.main_data["y1d_data"][test_data_index],
                                 "model_name": self.main_data["model_name"] + "\n Testing Data",
                                 }

            print(f"Note: Data splitting completed. Training Data and Testing Data are split on a ratio: {self.teacher_params['split']}")


def get_models():
    dict_of_models = {"lin": linear,
                      "quad": quadratic,
                      "sin": sine,
                      "cos": cosine,
                      "tam": tangent,
                      "exp": exponential,
                      "log": logarithmic,
                      }
    return dict_of_models


class linear:
    def __init__(self, x, teacher_params):
        self.gen_y = teacher_params["b"] * x + teacher_params["c"]
        self.gen_y1d = teacher_params["b"] * np.ones(len(x))
        self.name = "Linear Model " + r"$bx+c$" + "\n" + "$b=${:.1f}, $c=${:.1f}".format(teacher_params["b"], teacher_params["c"])


class quadratic:
    def __init__(self, x, teacher_params):
        self.gen_y = teacher_params["a"] * np.power(x, 2) + teacher_params["b"] * x + teacher_params["c"]
        self.gen_y1d = 2 * teacher_params["a"] * x + teacher_params["b"]
        self.name = "Quadratic Model " + r"$ax^2+bx+c$" + "\n" + "$a=${:.1f}, $b=${:.1f}, $c=${:.1f}".format(teacher_params["a"], teacher_params["b"], teacher_params["c"])


class sine:
    def __init__(self, x, teacher_params):
        self.gen_y = teacher_params["a"] * np.sin(teacher_params["b"] * x + teacher_params["c"])
        self.gen_y1d = teacher_params["b"] * teacher_params["a"] * np.cos(teacher_params["b"] * x + teacher_params["c"])
        self.name = "Sine Model " + r"$a*sin(bx+c)$" + "\n" + "$a=${:.1f}, $b=${:.1f}, $c=${:.1f}".format(teacher_params["a"], teacher_params["b"], teacher_params["c"])


class cosine:
    def __init__(self, x, teacher_params):
        self.gen_y = teacher_params["a"] * np.cos(teacher_params["b"] * x + teacher_params["c"])
        self.gen_y1d = -1 * teacher_params["b"] * teacher_params["a"] * np.sin(teacher_params["b"] * x + teacher_params["c"])
        self.name = "Cosine Model " + r"$a*cos(bx+c)$" + "\n" + "$a=${:.1f}, $b=${:.1f}, $c=${:.1f}".format(teacher_params["a"], teacher_params["b"], teacher_params["c"])


class tangent:
    def __init__(self, x, teacher_params):
        self.gen_y = teacher_params["a"] * np.tan(teacher_params["b"] * x + teacher_params["c"])
        self.gen_y1d = teacher_params["b"] * teacher_params["a"] * 1/np.square(np.cos(teacher_params["b"] * x + teacher_params["c"]))
        self.name = "Tangent Model " + r"$a*tan(bx+c)$" + "\n" + "$a=${:.1f}, $b=${:.1f}, $c=${:.1f}".format(teacher_params["a"], teacher_params["b"], teacher_params["c"])


class exponential:
    def __init__(self, x, teacher_params):
        self.gen_y = teacher_params["a"] * np.exp(teacher_params["b"] * x + teacher_params["c"])
        self.gen_y1d = teacher_params["b"] * teacher_params["a"] * np.exp(teacher_params["b"] * x + teacher_params["c"])
        self.name = "Exponential Model " + r"$a*exp(bx+c)$" + "\n" + "$a=${:.1f}, $b=${:.1f}, $c=${:.1f}".format(teacher_params["a"], teacher_params["b"], teacher_params["c"])


class logarithmic:
    def __init__(self, x, teacher_params):
        self.gen_y = teacher_params["a"] * np.log(np.abs(teacher_params["b"] * x + teacher_params["c"]))
        self.gen_y1d = teacher_params["b"] * teacher_params["a"] / (teacher_params["b"] * x + teacher_params["c"])
        self.name = "Logarithmic Model " + r"$a*log(bx+c)$" + "\n" + "$a=${:.1f}, $b=${:.1f}, $c=${:.1f}".format(teacher_params["a"], teacher_params["b"], teacher_params["c"])
