"""
Created by Chee Chong Hian.

Below contains Quantum Trainer operations for Quantum Circuit Learning.
"""

import numpy as np
import matplotlib.pyplot as plt

class QuantumTrainer:
    """
    To the allow high level control over of the quantum training.

    Attributes
    ----------
    select_optimiser, select_loss : str
            See __init__ method.
    optimiser_params: dict
        Contains user settings for the optimiser.
    teacher_model: 'TeacherModel' class
    quantum_computer: 'QuantumComputer' class

    """

    def __init__(self, verbose=True):
        self.verbose = verbose
        if verbose is True:
            print("Select QuantumTrainer Optimiser:\n")
            print("1. 'GD'  -> Standard Gradient Descent")
            print('\n')
            print("Select QuantumTrainer Loss Function:\n")
            print("1. 'quad_loss'  -> Quadratic Loss")
            print('\n')
            print("To select: use QuantumTrainer().config(select_optimiser='_<?>_', select_loss='_<?>_') \n \n")
            print("Config Setting Available:\n")
            print("* max_training_steps [int]: The maximum number of training steps for optimisation.")
            print("* learning_rate [float]: The multiplier for gradient descent. (Only used by some supported optimisers)")
            print('\n')
            print("Required Component Inputs to QuantumTrainer:\n")
            print("1. teacher_model: 'TeacherModel' class")
            print("2. quantum_computer : 'QuantumComputer' class\n")
            print("To input: use QuantumTrainer().input(teacher_model, quantum_computer)")
            print("Note: Please run '.config()' first before running '.inputs()' \n")

    def config(self, select_optimiser="GD", select_loss="quad_loss", max_training_steps=10, learning_rate=0.001):
        """
        User settings for Quantum Trainer Class.

        Parameters
        ----------
        select_optimiser, select_loss : str
            See __init__ method.
        max_training_steps: int
            The maximum number of training steps for optimisation.
        learning_rate: float
            The multiplier for gradient descent. (Only used by some supported optimiser)
        """
        self.select_optimiser = select_optimiser
        self.select_loss = select_loss
        self.optimiser_params = {"max_training_steps": max_training_steps,
                                 "learning_rate": learning_rate,
                                 }
        self.quantum_result_store = QuantumResult()

    def inputs(self, teacher_model, quantum_computer):
        """
        User inputs of components that defines the quantum training.

        Parameters
        ----------
        teacher_model: 'TeacherModel' class
        quantum_computer: 'QuantumComputer' class
        """
        self.teacher_model = teacher_model
        self.quantum_computer = quantum_computer

    def train(self):
        # Trains the quantum circuit in the quantum computer by using pre_input loss function and optimiser.

        loss_functions_selection = get_loss_functions()
        loss_functions_model = loss_functions_selection[self.select_loss]()

        optimiser_model_selection = get_optimise_models()
        optimiser_model = optimiser_model_selection[self.select_optimiser]


        self.quantum_result_store.save_names(loss_functions_model.name)
        optimiser_model(self.teacher_model, self.quantum_computer, loss_functions_model, self.optimiser_params, self.quantum_result_store)


def get_optimise_models():
    dict_of_models = {"GD": standard_gradient_descent,
                      }
    return dict_of_models


def get_loss_functions():
    dict_of_models = {"quad_loss": quadratic_loss,
                      "sobolev_loss": sobolev_loss,
                      }
    return dict_of_models


def standard_gradient_descent(teacher_model, quantum_computer, loss_functions_model, optimiser_params, quantum_result_store):
    """
    Apply the vanilla gradient descent without any modifications.

    Parameters
    ----------
    teacher_model: 'TeacherModel' class
    quantum_computer: 'QuantumComputer' class
    loss_functions_model : A python class that calculates the loss value, gradient wrt parameters.
    optimiser_params: dict
        Contains user settings for the optimiser.
    """
    # The quantum model is calculated as a<Z>, where
    # 'a' is free parameter to optimise and <Z> is the Z_expectation of first qubit wrt to the quantum circuit.
    a = 1

    # This the training loop
    for training_iter in range(optimiser_params["max_training_steps"]):
        predict_result = {}
        predict_result_data = {}
        for data_iter in range(teacher_model.teacher_params["number_of_points"]):

            # Run the quantum circuit and get the <Z>
            predict_result_measurements = quantum_computer.run_qc(teacher_model.training_data["x_data"][data_iter], loss_functions_model.shift_x)

            # Save the results a single dictionary
            predict_result_data[data_iter+1] = predict_result_measurements

        predict_result['predict_result_data'] = predict_result_data
        predict_result["a"] = a
        quantum_result_store.save_training_result(predict_result)

        # Calculate the loss value and gradients wrt parameters.
        loss_result = loss_functions_model.calculate_loss(quantum_computer.B, teacher_model.training_data, predict_result)
        quantum_result_store.save_loss_result(loss_result)
        print(loss_result["loss"])

        # Update the parameters
        a -= optimiser_params["learning_rate"] * loss_result['a_gradient']
        new_params = gradient_descent(quantum_computer.B, loss_result, optimiser_params)
        quantum_computer.update_params(new_params)

    print(f"a = {a}")


def gradient_descent(B, loss_result, optimiser_params):
    """
    Calculate and return the new parameters of quantum circuit for the next training step.

    Parameter
    ---------
    B : 'SingleQubitRotationBlock' class
    loss_result : dict
        Contains the loss value and gradient wrt parameters of a single training step.
    optimiser_params: dict
        Contains user settings for the optimiser.
    """
    loss_gradient_parameter_dict = loss_result["loss_gradient_parameter_dict"]
    old_params = B.parameter_dict
    new_params = {}
    for depth_iter in range(B.total_depth):
        new_params[depth_iter+1] = {}
        for qubit_iter in range(B.number_of_qubits):
            new_params[depth_iter+1][qubit_iter+1] = {}
            for rotate_gate_iter in range(3):

                # The gradient descent update
                new_params[depth_iter+1][qubit_iter+1][rotate_gate_iter+1] = old_params[depth_iter+1][qubit_iter+1][rotate_gate_iter+1] - optimiser_params["learning_rate"] * loss_gradient_parameter_dict[depth_iter+1][qubit_iter+1][rotate_gate_iter+1]

    return new_params


def extract_result(predict_result_data, result_type):
    result = []
    for data_iter in range(len(predict_result_data)):
        result.append(predict_result_data[data_iter+1][result_type])
    return result


class quadratic_loss:
    def __init__(self):
        self.shift_x = False
        self.name = "Quadratic"

    def calculate_loss(self, B, training_data, predict_result):
        """
        Calculate the loss value and gradient wrt parameters of a single training step.

        Parameters
        ----------
        B : 'SingleQubitRotationBlock' class
        training_data : dict
          Contains x, y and gradient data of the model for training.
        predict_result: dict
          This contains all neccessary quantites (unsorted) for every data, to calculate the loss function.
        """
        # Extract the result the iterates over all training data
        predict_y = np.asarray(extract_result(predict_result['predict_result_data'], result_type="output_data"))
        predict_y_parameter_gradient = extract_result(predict_result['predict_result_data'], result_type="gradient_parameter_dict")

        # Quadratic loss formula
        loss = np.sum(np.square(training_data["y_data"] - predict_result["a"] * predict_y))

        # The gradient of loss wrt a
        a_gradient = np.sum(-2*(training_data["y_data"] - predict_result["a"] * predict_y) * predict_y)

        # Calculates the gradient of loss wrt parametes in B
        loss_gradient_parameter_dict = {}
        for depth_iter in range(B.total_depth):

            loss_gradient_parameter_dict[depth_iter+1] = {}
            for qubit_iter in range(B.number_of_qubits):

                loss_gradient_parameter_dict[depth_iter+1][qubit_iter+1] = {}
                for rotate_gate_iter in range(3):

                    # Get all the gradients of a single parameter calculated from all x data
                    parameter_gradient = []
                    for data_iter in range(len(predict_y)):
                        parameter_gradient.append(predict_y_parameter_gradient[data_iter][depth_iter+1][qubit_iter+1][rotate_gate_iter+1])

                    # The gradient of loss wrt parametes in B
                    loss_gradient_parameter_dict[depth_iter+1][qubit_iter+1][rotate_gate_iter+1] = np.sum((training_data["y_data"] - predict_result["a"] * predict_y) * (-2*predict_result["a"]*np.asarray(parameter_gradient)))

        loss_result = {"loss": loss,
                       "loss_gradient_parameter_dict": loss_gradient_parameter_dict,
                       "a_gradient": a_gradient}

        return loss_result


class sobolev_loss:
    def __init__(self):
        self.shift_x = True
        self.name = "Sobolev"

    def calculate_loss(self, B, training_data, predict_result):
        """
        Calculate the loss value and gradient wrt parameters of a single training step.

        Parameters
        ----------
        B : 'SingleQubitRotationBlock' class
        training_data : dict
          Contains x, y and gradient data of the model for training.
        predict_result: dict
          This contains all neccessary quantites (unsorted) for every data, to calculate the loss function.
        """
        # Extract the result the iterates over all training data
        predict_y = np.asarray(extract_result(predict_result['predict_result_data'], result_type="output_data"))
        predict_y_parameter_gradient = extract_result(predict_result['predict_result_data'], result_type="gradient_parameter_dict")
        predict_gradient_x = np.asarray(extract_result(predict_result['predict_result_data'], result_type="output_gradient_x_data"))
        predict_gradient_x_parameter_gradient = extract_result(predict_result['predict_result_data'], result_type="gradient_x_parameter_dict")

        # Sobolev loss formula
        loss = np.sum(np.square(training_data["y_data"] - predict_result["a"] * predict_y) + np.square(training_data["y_data"] - predict_result["a"] * predict_gradient_x))

        # The gradient of loss wrt a
        a_gradient = np.sum(-2*(training_data["y_data"] - predict_result["a"] * predict_y) * predict_y - 2*(training_data["y_data"] - predict_result["a"] * predict_gradient_x) * predict_gradient_x)

        # Calculates the gradient of loss wrt parametes in B
        loss_gradient_parameter_dict = {}
        for depth_iter in range(B.total_depth):

            loss_gradient_parameter_dict[depth_iter+1] = {}
            for qubit_iter in range(B.number_of_qubits):

                loss_gradient_parameter_dict[depth_iter+1][qubit_iter+1] = {}
                for rotate_gate_iter in range(3):

                    # Get all the gradients of a single parameter calculated from all x data
                    parameter_gradient = []
                    parameter_x_gradient = []
                    for data_iter in range(len(predict_y)):
                        parameter_gradient.append(predict_y_parameter_gradient[data_iter][depth_iter+1][qubit_iter+1][rotate_gate_iter+1])
                        parameter_x_gradient.append(predict_gradient_x_parameter_gradient[data_iter][depth_iter+1][qubit_iter+1][rotate_gate_iter+1])

                    parameter_gradient = np.asarray(parameter_gradient)
                    parameter_x_gradient = np.asarray(parameter_x_gradient)

                    # The gradient of loss wrt parametes in B
                    loss_gradient_parameter_dict[depth_iter+1][qubit_iter+1][rotate_gate_iter+1] = np.sum((training_data["y_data"] - predict_result["a"] * predict_y) * (-2*predict_result["a"]*parameter_gradient) + (training_data["y_data"] - predict_result["a"] * predict_gradient_x) * (-2*predict_result["a"]*parameter_x_gradient))

        loss_result = {"loss": loss,
                       "loss_gradient_parameter_dict": loss_gradient_parameter_dict,
                       "a_gradient": a_gradient}

        return loss_result


class QuantumResult:
    def __init__(self):
        self.counter = 0
        self.loss_training_history = []
        self.loss_gradient_parameter_history = {}
        self.parameter_a_training_history = []
        self.parameter_a_gradient_training_history = []

        self.quantum_model_y_prediction_training_history = {}
        self.quantum_model_y_gradient_parameter_training_history = {}

        self.quantum_model_gradient_x_prediction_training_history = {}
        self.quantum_model_gradient_x_gradient_parameter_training_history = {}

    def save_training_result(self, predict_result):
        self.counter += 1

        self.parameter_a_training_history.append(predict_result["a"])

        predict_y = np.asarray(extract_result(predict_result['predict_result_data'], result_type="output_data"))
        self.quantum_model_y_prediction_training_history[self.counter] = predict_y

        predict_y_parameter_gradient = extract_result(predict_result['predict_result_data'], result_type="gradient_parameter_dict")
        self.quantum_model_y_gradient_parameter_training_history[self.counter] = predict_y_parameter_gradient

        if "output_gradient_x_data" in predict_result['predict_result_data'][1]:
            predict_gradient_x = np.asarray(extract_result(predict_result['predict_result_data'], result_type="output_gradient_x_data"))
            self.quantum_model_gradient_x_prediction_training_history[self.counter] = predict_gradient_x

        if "gradient_x_parameter_dict" in predict_result['predict_result_data'][1]:
            predict_gradient_x_parameter_gradient = extract_result(predict_result['predict_result_data'], result_type="gradient_x_parameter_dict")
            self.quantum_model_gradient_x_gradient_parameter_training_history[self.counter] = predict_gradient_x_parameter_gradient

    def save_loss_result(self, loss_result):
        self.loss_training_history.append(loss_result["loss"])
        self.loss_gradient_parameter_history[self.counter] = loss_result["loss_gradient_parameter_dict"]
        self.parameter_a_gradient_training_history.append(loss_result["a_gradient"])

    def save_names(self, loss_name):
        self.loss_name = loss_name
        pass

    def plot_loss(self):
        fig, ax = plt.subplots(dpi=100)
        training_iter = np.arange(1, self.counter+1, 1)
        ax.plot(training_iter, np.asarray(self.loss_training_history))
        ax.set_ylabel(r"Loss $L$")
        ax.set_xlabel("Training Epochs")
        fig.suptitle("Loss Value ("+self.loss_name+")")
