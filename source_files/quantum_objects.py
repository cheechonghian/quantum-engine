# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 16:35:08 2020

@author: CHEE
"""


class info:
    def __init__(self):
        self.function_models = ["linear",
                                ]


class function:
    """
    Attributes
    ----------

    """
    def __init__(self, a, b, c=None):
        """
        Parameters
        ----------

        """
        self.a = a
        self.b = b
        self.c = c
        return

    def linear(self, x):
        return self.b*x + self.a

