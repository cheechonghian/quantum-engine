"""
is somehing.

second line
"""


class quantum_report:
    """
    To store data that is required for analysis.

    Attributes
    ----------
    loss_data : list
        Contains the loss function values for every epochs/training iterations.
    """

    def __init__(self):

        self.loss_data = []
        pass

    def store_loss_info(self, new_loss):
        """
        Append a new loss data entry into the existing list.

        Parameters
        ----------
        new_loss : float
            A new loss data entry.
        """
        self.loss_data.append(new_loss)
        pass
