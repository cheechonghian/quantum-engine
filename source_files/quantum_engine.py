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
import quantum_optimiser as qopt


class teacher_model:
    """
    Define a teacher model that will be used as a target function for the quantum model to approximate to.

    Attributes
    ----------
    See parameters of __init__, config, noise_config
    """
    def __init__(self, verbose=True):
        """
        #

        """
        if verbose is True:
            print("""
                select_model(_):

                lin -> Linear         b*x + a
                quad -> Quadratic      c*x^2 + b*x + a
                e -> Exponential    a*exp(b*x)
                l -> Logarithmic    a*log(b*x)
                sin -> Sine         a*sin(b*x + c)
                cos -> Cosine       a*cos(b*x +c)
                  """)

    def config(self, select_model="lin", x_lower_limit=0.0, x_upper_limit=1.0, number_of_points=10, a=1.0, b=1.0, c=1.0, d=1.0, ):
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
        self.select_model = select_model
        self.x_low_lim = x_lower_limit
        self.x_upp_lim = x_upper_limit
        self.no_of_points = number_of_points
        self.model_parameters = {"a": a, "b": b, "c": c, "d": d}

    def noise_config(self, x_choose="equal_spacing", y_choose="None", x_g_var=0.1, y_g_var=0.1):
        """
        Apply noise configuration settings to teacher model.

        Parameters
        ----------
        x_choose : str
            Choosing independent variable x by:
                1. "equal_spacing" starting and ending at the limits.
                2. a "random_uniform" distribution.
                3. "equal_space_gaussian", apply gaussian noise to 1.
        y_choose : str
            Apply noise to the dependent variable y (output) of the teacher model.
            1. "None", no noise applied.
            2. "gaussian" apply gaussian noise

        x_g_var, y_g_var : float
            Variance of the gaussian noise.
        """
        self.x_choose = x_choose
        self.y_choose = y_choose
        self.x_g_var = x_g_var
        self.y_g_var = y_g_var


    def generate_training_batch(self):
        model = tobj.teacher().call[self.select_model]

        return #training_batch # Numpy array

    def generate_testing_batch(self):

        return #testing_batch  # Numpy array


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
        return None

    def config(self,select_loss):
        """
        Parameters
        ----------

        """
        self.select_loss = select_loss
        return None

    def calculate_loss(self, quantum_measurement_data, training_data):
        self.loss_
        return


class quantum_data:
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

    def config(self,select_encoding):
        """
        Parameters
        ----------

        """
        self.select_encoding = select_encoding
        return None


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

    def config(self,select_ham_type):
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

    def config(self,select_ml_type):
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

    def config(self,select_upload_type):
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

    def config(self,select_measurement):
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

    def config(self, select_quantum_model):
        """
        Parameters
        ----------

        """
        self.select_quantum_model = select_quantum_model
        return None

    def inputs(self, my_entangling, my_parameterised, my_reupload, my_measurement):
        """
        Parameters
        ----------

        """
        self.ham_layer = my_entangling
        self.ml_layer = my_parameterised
        self.ul_layer = my_reupload
        self.qmeasure = my_measurement
        return None

    def input_data(self,training_data):
        # Process the quantum data
        return None

    def run_model(self):

        return # output_state # Generate an Output state

    def run_model(self):
        # Generate an Output state
        return None

    def measure_Z(self, output_state):
        # Measure the first qubit using a pauli_z gate

        return None #measurement_result # This should be a single number


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

    def config(self, optimiser_name, max_training_set=10, learning_rate=0.5):
        self.config_store = self.config_storage(optimiser_name, max_training_set, learning_rate)

        """
        Parameters
        ----------

        """

    class config_storage:
        def __init__(self, optimiser_name, max_training_sets, learning_rate):
            self.max_training_sets = max_training_sets
            self.optimiser_name = optimiser_name
            self.learning_rate = learning_rate

    def inputs(self, teacher_model, quantum_model, loss_function):
        """
        Parameters
        ----------
        teacher_model: teacher_model class
        quantum_model: quantum_model class
        loss_function: loss_function class
        optimizer: optimizer class
        """
        self.teacher_model = teacher_model
        self.quantum_model = quantum_model
        self.loss_function = loss_function

    def train(self):
        optimiser = qopt.optimiser().call[self.optimiser_name]
        my_trained_quantum_model, my_training_report = optimiser(self.teacher_model, self.quantum_model, self.loss_function, self.config_store)

        return my_trained_quantum_model, my_training_report





class training_report:
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


class trained_quantum_model:
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

    def predict(self, x):
        """
        Parameters
        ----------

        """
        y = x
        return y
