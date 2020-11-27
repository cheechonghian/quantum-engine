# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 18:11:38 2020

@author: User
"""

def show_params(select_block):
    block_params {"ham": ham_params,
                  "qubit_rotate": qubit_rotate_params }
    params_select = block_params[select_block]
    print_statement = "Default parameters for " + select_block + ":" + params_select
    + "\n To confirm, run: 'quantum_block.config_dict()'. "
    + "\n To change parameters, run: "
    + "change_params = {'params': new value,}"
    + "quantum_block.config_dict(change_params)"
    return print_statement



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
