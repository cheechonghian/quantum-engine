import numpy as np
import time

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

    def select_model(self, select_model):
        """
        Parameters
        ----------

        """
        self.select_model = select_model
        return None

    def config(self):
        """
        Parameters
        ----------

        """
        return None


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

    def select_loss(self, select_loss):
        """
        Parameters
        ----------

        """
        self.select_loss = select_loss
        return None

    def config(self):
        """
        Parameters
        ----------

        """
        return None


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

    def select_encoding(self, select_encoding):
        """
        Parameters
        ----------

        """
        self.select_encoding = select_encoding
        return None

    def config(self):
        """
        Parameters
        ----------

        """
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

    def config(self):
        """
        Parameters
        ----------

        """
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

    def config(self):
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

    def config(self):
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

    def select_measurement(self, select_measurement):
        """
        Parameters
        ----------

        """
        self.select_measurement = select_measurement
        return None

    def config(self):
        """
        Parameters
        ----------

        """
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

    def select_quantum_model(self, select_quantum_model):
        """
        Parameters
        ----------

        """
        self.select_quantum_model = select_quantum_model
        return None

    def config(self):
        """
        Parameters
        ----------

        """
        return None


class optimiser:
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

    def select_optimiser(self, select_optimiser):
        """
        Parameters
        ----------

        """
        self.select_optimiser = select_optimiser
        return None

    def config(self):
        """
        Parameters
        ----------

        """
        return None


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

    def config(self):
        """
        Parameters
        ----------

        """
        return None

    def train(self):
        """
        Parameters
        ----------

        """
        return None


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
