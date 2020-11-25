# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 16:35:08 2020

@author: CHEE
"""


class teacher:
    def __init__(self):
        self.call = {"lin": linear_model,
                     "quad": quadratic_model,
                     }


def linear_model(x):
    return self.b*x + self.a


def quadratic_model(x):
    return self.b*x + self.a
