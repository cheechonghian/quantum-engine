import numpy as np
import time
import quantum_objects as qobj
import quantum_trainer as qtrain

class teacher_model:
    """
    Attributes
    ----------

    """
    def __init__(self, select_model=False, verbose=True):
        """
        Parameters
        ----------

        """
        if verbose is True:
            print("""
select_model(_):

1 -> Linear         b*x + a
2 -> Quadratic      c*x^2 + b*x + a
e -> Exponential    a*exp(b*x)
l -> Logarithmic    a*log(b*x)
sin -> Sine         a*sin(b*x + c)
cos -> Cosine       a*cos(b*x +c)
                  """)
        if select_model is not False:
            self.select_model = select_model
        return None

    def config(self, select_model):

        """
        Parameters
        ----------

        """
        self.select_model = select_model
        return None

    def generate_training_batch(self):

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

        self.select_loss = select_loss
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
    Attributes
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
        optimiser = qtrain.optimiser().call[self.optimiser_name]
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
