# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 08:58:56 2020

@author: User
"""
import numpy as np
from collections import OrderedDict


def run_quantum_computer(initial_state, quantum_circuit, qcomputer_params):
    model = get_circuit_models(qcomputer_params["select_qc_model"])
    quantum_evolution = model(initial_state["amplitude"], quantum_circuit, qcomputer_params)
    final_state_amplitude = quantum_evolution.get_final_state()
    assert isinstance(final_state_amplitude, np.ndarray), "'final_state_amplitude' is not a numpy array."
    assert isinstance(final_state_amplitude.ndim != 1), "'final_state_amplitude' is not a 1D array."

    y_data = quantum_circuit["quantum_measurement"].measure(final_state_amplitude)
    assert isinstance(y_data, float), "'measurement_result' is not a number."

    final_states_amplitude_plus = quantum_evolution.get_final_states_for_gradient_plus()  # This is dictionary
    final_states_amplitude_minus = quantum_evolution.get_final_states_for_gradient_minus()  # This is dictionary
    grad_plus = quantum_circuit["quantum_measurement"].measure_gradient(final_states_amplitude_plus)
    grad_minus = quantum_circuit["quantum_measurement"].measure_gradient(final_states_amplitude_minus)
    grad = (grad_plus - grad_minus)/2
    quantum_result_data = {"y_data": y_data,
                           "grad": grad
                           }
    return quantum_result_data


def get_circuit_models():
    model_dict = {"AB_repeat": AB_repeat,
                  }
    return model_dict


class AB_repeat:
    def __Init__(self, initial_state_amplitude, quantum_circuit, qcomputer_params):
        self.initial_state_amplitude = initial_state_amplitude
        self.quantum_circuit = quantum_circuit
        self.qcomputer_params = qcomputer_params

    def get_final_state(self):
        temp_state = self.initial_state_amplitude
        for depth_iter in self.qcomputer_params["depth"]:
            temp_state = np.dot(self.quantum_circuit["quantum_block_A"].get_matrix_operator(depth_iter+1), temp_state)
            temp_state = np.dot(self.quantum_circuit["quantum_block_B"].get_matrix_operator(depth_iter+1), temp_state)
        return temp_state

    def get_final_states_for_gradient_plus(self):
        final_states_plus = OrderedDict()

        # For parameters in B
        for B_param_iter in range(self.quantum_circuit["quantum_block_B"].block_params['no_of_parameters']):
            temp_state_plus = self.initial_state_amplitude
            for depth_iter in self.qcomputer_params["depth"]:
                temp_state_plus = np.dot(self.quantum_circuit["quantum_block_A"].get_matrix_operator(depth_iter+1), temp_state_plus)
                temp_state_plus = np.dot(self.quantum_circuit["quantum_block_B"].get_matrix_operator_shift_plus(depth_iter+1, B_param_iter), temp_state_plus)

            keyname_plus = 'B' + str(B_param_iter+1) + "_plus"
            final_states_plus[keyname_plus] = temp_state_plus

        return final_states_plus

    def get_final_states_for_gradient_minus(self):
        final_states_minus = OrderedDict()

        # For parameters in B
        for B_param_iter in range(self.quantum_circuit["quantum_block_B"].block_params['no_of_parameters']):
            temp_state_minus = self.initial_state_amplitude
            for depth_iter in self.qcomputer_params["depth"]:
                temp_state_minus = np.dot(self.quantum_circuit["quantum_block_A"].get_matrix_operator(depth_iter+1), temp_state_minus)
                temp_state_minus = np.dot(self.quantum_circuit["quantum_block_B"].get_matrix_operator_shift_minus(depth_iter+1, B_param_iter), temp_state_minus)

            keyname_minus = 'B' + str(B_param_iter+1) + "_minus"
            final_states_minus[keyname_minus] = temp_state_minus

        return final_states_minus


class ABC_repeat:
    def __init__(self, initial_state_amplitude, quantum_circuit, qcomputer_params):
        pass


class A_repeat:
    def _init(self, initial_state_amplitude, quantum_circuit, qcomputer_params):
        pass
