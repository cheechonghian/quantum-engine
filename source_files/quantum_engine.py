"""
Quantum Engine script.

Universal Parameters
--------------------
verbose : bool
    Print statements to console

"""


import numpy as np
import time
import teacher_objects as tobj
import loss_objects as lobj
import quantum_optimiser as qopt
import quantum_models_objects as qmobj
import matplotlib.pyplot as plt

class teacher_model:
    """
    Define a teacher model that will be used as a target function for the quantum model to approximate to.

    Attributes
    ----------
    See parameters of __init__, config, noise_config
    """

    def __init__(self, verbose=True):
        """
        Apply configuration settings to teacher model.

        Parameters
        ----------
        verbose : bool
            To enable console print statements for more information.
        is_train_data_gen, is_test_data_gen : bool
            To check if training or testing data set has been generated.
        """
        self.verbose = verbose
        self.is_train_data_gen = False
        self.is_test_data_gen = False

    def config(self, select_model="lin", x_lower_limit=0.0, x_upper_limit=1.0, no_of_train_points=10, no_of_test_points=10, a=1.0, b=1.0, c=1.0, d=1.0,
               x_choose="equal_spacing", y_choose="None", x_g_var=np.array([0.1]), y_g_var=np.array([0.1]), y1_g_var=np.array([0.1]),):
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

        Noise Parameters
        ----------
        x_choose : str
            Choosing independent variable x by:
            1. "equal_spacing" starting and ending at the limits.
            2. "equal_space_gaussian", apply gaussian noise to 1.
                * Note that the gaussian values generated may not respect the limits.
            3. a "random_uniform" distribution.

        y_choose : str
            Apply noise to the dependent variable y (output) of the teacher model.
            1. "None", no noise applied.
            2. "gaussian" apply gaussian noise

        x_g_var, y_g_var : numpy array
            Variance of the gaussian noise.
               * If len()=1, all variances are assumed be the same.
               * Otherwise, all variances must be supplied in the numpy array.
        """

        assert isinstance(no_of_train_points, int), "'number_of_points' is not an integer."
        assert no_of_train_points >= 10, "'number_of_points' is less than 10. Too few data points."

        assert isinstance(x_g_var, np.ndarray), "'x_g_var' is not numpy array"
        assert (len(x_g_var) == 1) or (len(x_g_var) == no_of_train_points), "The number of variances in 'x_g_var' is not equal to 'number_of_points'."

        assert isinstance(y_g_var, np.ndarray), "'y_g_var' is not numpy array"
        assert (len(y_g_var) == 1) or (len(y_g_var) == no_of_train_points), "The number of variances in 'y_g_var' is not equal to 'number_of_points'."

        assert isinstance(y1_g_var, np.ndarray), "'y1_g_var' is not numpy array"
        assert (len(y1_g_var) == 1) or (len(y1_g_var) == no_of_train_points), "The number of variances in 'y1_g_var' is not equal to 'number_of_points'."

        self.teacher_params = {"select_model": select_model,
                               "x_low_lim": x_lower_limit,
                               "x_upp_lim": x_upper_limit,
                               "no_of_train_points": no_of_train_points,
                               "no_of_test_points": no_of_test_points,
                               "a": a,
                               "b": b,
                               "c": c,
                               "d": d,
                               "x_choose": x_choose,
                               "y_choose": y_choose,
                               "x_g_var": x_g_var,
                               "y_g_var": y_g_var,
                               "y1_g_var": y1_g_var,
                               }

    def generate_training_batch(self):
        self.is_train_data_gen = True
        self.train_data = tobj.generate_data(self.teacher_params, is_train_data=True)

    def generate_testing_batch(self):
        self.is_test_data_gen = True
        self.test_data = tobj.generate_data(self.teacher_params, is_test_data=True)
        pass  # testing_batch  # Numpy array

    def plot_training_batch(self):
        fig, ax = plt.subplots(2, 1, dpi=100, sharex=True)
        ax[0].scatter(self.train_data["x_data"], self.train_data["y_data"])
        ax[0].set_ylabel(r"$f(x)$")
        ax[1].scatter(self.train_data["x_data"], self.train_data["y1_data"])
        ax[1].set_ylabel(r"$f^{\prime}(x)$")
        ax[1].set_xlabel(r"$x$")
        fig.subplots_adjust(hspace=0.1)
        fig.suptitle("Teacher Model Training Batch\n" + self.train_data["model_name"])

    def plot_testing_batch(self):
        fig, ax = plt.subplots(2, 1, dpi=100, sharex=True)
        ax[0].scatter(self.test_data["x_data"], self.test_data["y_data"])
        ax[0].set_ylabel(r"$f(x)$")
        ax[1].scatter(self.test_data["x_data"], self.test_data["y1_data"])
        ax[1].set_ylabel(r"$f^{\prime}(x)$")
        ax[1].set_xlabel(r"$x$")
        fig.subplots_adjust(hspace=0.1)
        fig.suptitle("Teacher Model Testing Batch\n" + self.test_data["model_name"])


class loss_function:
    """
    Attributes
    ----------

    """

    def __init__(self, verbose=True):
        """
        Parameters
        ----------

        """
        self.verbose = verbose

    def config(self, select_loss="quad_loss"):
        """
        Parameters
        ----------

        """
        self.loss_params = {"select_loss": select_loss,
                            }

    def calculate_loss(self, quantum_result_data, teacher_data):
        """
        Parameters
        ----------
        quantum_result_data: dict
            *** To be filled ***
        teacher_data: dict
            Either 'teacher_model.train_data' or 'teacher_model.test_data'

        """
        loss_data = lobj.calculate_loss(quantum_result_data, teacher_data, self.loss_params)
        return loss_data



class quantum_encoding:
    """
    Attributes
    ----------

    """
    def __init__(self, verbose=True):
        """
        Parameters
        ----------

        """
        self.verbose = verbose

    def config(self, select_encoding):
        """
        Parameters
        ----------

        """
        self.select_encoding = select_encoding


    def encode(self, training_x_data):
        pass

class quantum_layer:
    """
    Attributes
    ----------

    """
    # This class shall contain methods for all types of quantum layers
    def __init__(self, verbose=True):
        return None


class quantum_ham_layer(quantum_layer):
    """
    Attributes
    ----------

    """
    def __init__(self, verbose=True):
        """
        Parameters
        ----------

        """
        return None

    def config(self, select_ham_type):
        """
        Parameters
        ----------

        """
        self.select_ham_type = select_ham_type
        return None


class quantum_ml_layer(quantum_layer):
    """
    Attributes
    ----------

    """
    def __init__(self, verbose=True):
        """
        Parameters
        ----------

        """
        return None

    def config(self, select_ml_type):
        self.select_ml_type = select_ml_type
        return None


class quantum_ul_layer(quantum_layer):
    """
    Attributes
    ----------

    """
    def __init__(self, verbose=True):
        """
        Parameters
        ----------

        """
        return None

    def config(self, select_upload_type):
        self.select_upload_type = select_upload_type
        return None


class quantum_measurement:
    """
    Attributes.

    ----------


    """

    def __init__(self, verbose=True):
        """
        Parameters
        ----------

        """
        return None

    def config(self, select_measurement):
        """
        Parameters
        ----------

        """
        self.select_measurement = select_measurement
        return None


class quantum_model:
    """
    Attributes
    ----------

    """
    def __init__(self, verbose=True):
        """
        Parameters
        ----------

        """
        return None

    def config(self, select_quantum_model="default", depth=2):
        """
        Parameters
        ----------

        """
        self.model_config_params = {"select_model": select_quantum_model,
                                    "depth": depth,
                                    }
        return None

    def inputs(self, my_quantum_data, my_entangling, my_parameterised, my_reupload, my_measurement):
        """
        Parameters
        ----------

        """
        self.quantum_data = quantum_data
        self.ham_layer = my_entangling
        self.ml_layer = my_parameterised
        self.ul_layer = my_reupload
        self.qmeasure = my_measurement
        return None

    def input_data(self, teacher_model_data, ):
        # Process the quantum data
        self.initial_state = quantum_data.encode(teacher_model_data)

    def run_model(self):
        measurement_result = qmobj.run_quantum_computer(self.initial_state, self.ham_layer, self.ml_layer, self.ul_layer, self.qmeasure, )
        return measurement_result


class quantum_trainer:
    """
    Attributes
    ----------

    """
    def __init__(self, verbose=True):
        """
        Parameters
        ----------

        """
        return None

    def config(self, select_optimiser="GD", max_training_set=10, learning_rate=0.5):
        """
        Parameters
        ----------

        """
        self.trainer_params = {"select_optimiser": select_optimiser,
                               "max_training_set": max_training_set,
                               "learning_rate": learning_rate,
                               }

    def inputs(self, teacher_model, quantum_model, loss_function):
        """
        Parameters
        ----------
        teacher_model: teacher_model class
        quantum_model: quantum_model class
        loss_function: loss_function class
        """
        self.teacher_model = teacher_model
        self.quantum_model = quantum_model
        self.loss_function = loss_function

    def train(self):
        my_training_report = qopt.optimiser(self.teacher_model, self.quantum_model, self.loss_function, self.trainer_params)
        return my_training_report





