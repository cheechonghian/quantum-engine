# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 20:00:51 2020

@author: CHEE
"""
import quantum_report as qrep
import numpy as np

class optimiser:
    def __init__(self):
        self.call = {"GD": standard_gradient_descent,
                     "SDG": stochastic_gradient_descent,
                     }


def standard_gradient_descent(teacher_model, quantum_model, loss_function,config_store):
    qrep.new_report()
    training_batch = teacher_model.generate_training_batch()
    for _ in range(config_store.max_training_sets):
        quantum_result = []
        quantum_gradient_plus = []
        quantum_gradient_minus = []
        for i in range(len(training_batch)):
            quantum_model.input_data(training_batch[i])
            output_state = quantum_model.run_model()
            quantum_result.append(quantum_model.measure_Z(output_state))
            output_state_plus = quantum_model.run_model_plus()
            quantum_gradient_plus.append(quantum_model.measure_Z(output_state_plus))
            output_state_minus = quantum_model.run_model_minus()
            quantum_gradient_minus.append(quantum_model.measure_Z(output_state_minus))

        loss = loss_function.calculate_loss(np.asarray(quantum_result), training_batch)
        quantum_gradient_plus = np.sum(np.stack(quantum_gradient_plus), axis=0)
        quantum_gradient_minus = np.sum(np.stack(quantum_gradient_minus), axis=0)
        quantum_gradient = 0.5 * (quantum_gradient_plus - quantum_gradient_minus) # parameter shift rule
        loss_gradient = loss_function.calculate_gradient(quantum_gradient)
        old_quantum_params = quantum_model.get_params()
        new_quantum_params = old_quantum_params - loss_gradient * config_store.learning_rate # Gradient descent
        quantum_model.updata_params(new_quantum_params)

    return None #report


def stochastic_gradient_descent():
    return






