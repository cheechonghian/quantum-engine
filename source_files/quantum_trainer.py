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

        # Get the selected loss function
        loss_functions_selection = get_loss_functions()
        loss_functions_model = loss_functions_selection[self.select_loss]()

        # Get the optimiser model
        optimiser_model_selection = get_optimise_models()
        optimiser_model = optimiser_model_selection[self.select_optimiser]

        # Setup QuantumResults
        self.quantum_result_store.save_names(loss_functions_model.name)
        self.quantum_result_store.save_training_date(self.teacher_model.training_data)

        # Run the optimiser model with the selected loss function
        optimiser_model(self.teacher_model, self.quantum_computer, loss_functions_model, self.quantum_result_store, self.optimiser_params)


def get_optimise_models():
    # This function all one to call a function/class by entering the name of it
    dict_of_models = {"GD": standard_gradient_descent,
                      }
    return dict_of_models


def get_loss_functions():
    # This function all one to call a function/class by entering the name of it
    dict_of_models = {"quad_loss": quadratic_loss,
                      "sobolev_loss": sobolev_loss,
                      }
    return dict_of_models


def standard_gradient_descent(teacher_model, quantum_computer, loss_functions_model, quantum_result_store, optimiser_params,):
    """
    Apply the vanilla gradient descent without any modifications.

    Parameters
    ----------
    teacher_model: 'TeacherModel' class
    quantum_computer: 'QuantumComputer' class
    loss_functions_model : Loss function class
        Calculates the loss value, gradient wrt parameters.
    quantum_result_store: "QuantumResult" class
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

        # At every training step, input the data sequentially.
        for data_iter in range(teacher_model.teacher_params["number_of_points"]):

            # Run the quantum circuit and get the <Z> measurement and its gradients
            predict_result_measurements = quantum_computer.run_qc(teacher_model.training_data["x_data"][data_iter], loss_functions_model.shift_x)

            # Save the results a single dictionary
            predict_result_data[data_iter+1] = predict_result_measurements

        # Package all results in another dictionary
        predict_result['predict_result_data'] = predict_result_data
        predict_result["a"] = a
        quantum_result_store.save_training_result(predict_result)

        # Calculate the loss value and gradients wrt parameters.
        loss_result = loss_functions_model.calculate_loss(quantum_computer.B, teacher_model.training_data, predict_result)
        quantum_result_store.save_loss_result(loss_result)
        print(loss_result["loss"])

        # Update the parameters
        new_params = gradient_descent(quantum_computer.B, predict_result, loss_result, optimiser_params)
        quantum_computer.update_params(new_params["parameter_theta"])
        a = new_params["a"]


def gradient_descent(B, predict_result, loss_result, optimiser_params):
    """
    Calculate and return the new parameters of quantum circuit for the next training step.

    Parameter
    ---------
    B : 'SingleQubitRotationBlock' class
    predict_result: dict
        Contains the old parameter a. It also contain other quantites (unsorted) for every data, to calculate the loss function. (which is not used in this function)
    loss_result : dict
        Contains the loss value and gradient wrt parameters of a single training step.
    optimiser_params: dict
        Contains user settings for the optimiser.
    """
    # All new updated parameters will be stored here
    new_params = {}

    # Update parameter a
    new_params["a"] = predict_result["a"] - optimiser_params["learning_rate"] * loss_result['a_gradient']

    # Update the parameter in B
    loss_gradient_parameter_dict = loss_result["loss_gradient_parameter_dict"]
    old_params = B.parameter_dict
    new_params["parameter_theta"] = {}
    for depth_iter in range(B.total_depth):

        new_params["parameter_theta"][depth_iter+1] = {}
        for qubit_iter in range(B.number_of_qubits):

            new_params["parameter_theta"][depth_iter+1][qubit_iter+1] = {}
            for rotate_gate_iter in range(3):

                # The gradient descent update
                new_params["parameter_theta"][depth_iter+1][qubit_iter+1][rotate_gate_iter+1] = old_params[depth_iter+1][qubit_iter+1][rotate_gate_iter+1] - optimiser_params["learning_rate"] * loss_gradient_parameter_dict[depth_iter+1][qubit_iter+1][rotate_gate_iter+1]

    return new_params


def extract_result(predict_result_data, result_type):
    """
    Extract the required results by x training coordinates and put it in a list.

    Parameters
    ----------
    predict_result_data: dict
        The all result is organised by x training coordinates.
        For example:
            x data        Results
            1             A1,B1,C1
            2             A2,B2,C2
            3             A3,B3,C3
        This function will
                        Result   x data
        (result_type)-> A        A1,A2,A3 <-(Output)
                        B        B1,B2,B3
                        C        C1,C2,C3
    result_type: str
        A key value in predict_result_data, which you want to extract
    """
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
                    loss_gradient_parameter_dict[depth_iter+1][qubit_iter+1][rotate_gate_iter+1] = np.sum((training_data["y_data"] - predict_result["a"] * predict_y) * (-2*predict_result["a"]*parameter_gradient) + (training_data["y1d_data"] - predict_result["a"] * predict_gradient_x) * (-2*predict_result["a"]*parameter_x_gradient))

        loss_result = {"loss": loss,
                       "loss_gradient_parameter_dict": loss_gradient_parameter_dict,
                       "a_gradient": a_gradient}

        return loss_result


class QuantumResult:
    def __init__(self):
        # Prepare the items to be save for analysis
        self.counter = 0
        self.loss_training_history = []
        self.loss_gradient_parameter_history = {}
        self.parameter_a_training_history = []
        self.parameter_a_gradient_training_history = []

        self.quantum_model_y_prediction_training_history = {}
        self.quantum_model_y_gradient_parameter_training_history = {}

        self.quantum_model_gradient_x_prediction_training_history = {}
        self.quantum_model_gradient_x_gradient_parameter_training_history = {}

    def save_training_date(self, training_data):
        # Save the teacher training data
        self.training_data = training_data

    def save_training_result(self, predict_result):
        # Extract and organise the results before saving them
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
        # Save the calculation of the loss result.
        self.loss_training_history.append(loss_result["loss"])
        self.loss_gradient_parameter_history[self.counter] = loss_result["loss_gradient_parameter_dict"]
        self.parameter_a_gradient_training_history.append(loss_result["a_gradient"])

    def save_names(self, loss_name):
        # Save the name of the loss function
        self.loss_name = loss_name
        pass

    def plot_loss(self):
        # Plot loss value vs training iter.
        fig, ax = plt.subplots(dpi=100)
        training_iter = np.arange(1, self.counter+1, 1)
        ax.plot(training_iter, np.asarray(self.loss_training_history))
        ax.set_ylabel(r"Loss $L$")
        ax.set_xlabel("Training Epochs")
        fig.suptitle("Loss Value ("+self.loss_name+")")

    def plot_a(self):
        # Plot parameter a vs training iter
        fig, ax = plt.subplots(dpi=100)
        training_iter = np.arange(1, self.counter+1, 1)
        ax.plot(training_iter, np.asarray(self.parameter_a_training_history))
        ax.set_ylabel(r"Multiplier $a$")
        ax.set_xlabel("Training Epochs")
        fig.suptitle(r"Multiplier $a$ Value")

    def plot_final_result(self):
        # Plot the trained quantum model
        fig, ax = plt.subplots(dpi=100)
        prediction = self.parameter_a_training_history[-1] * self.quantum_model_y_prediction_training_history[self.counter]
        initial = self.parameter_a_training_history[0] * self.quantum_model_y_prediction_training_history[1]
        ax.plot(self.training_data['x_data'], prediction, label="Quantum (Trained)")
        ax.plot(self.training_data['x_data'], initial, label="Quantum (Untrained)")
        ax.plot(self.training_data["x_data"], self.training_data["y_data"], label="Teacher")
        ax.set_ylabel(r"Function Model $f(x)$")
        ax.set_xlabel(r"$x$")
        ax.legend()
        title = r"Quantum Circuit Learning" + "\n" + self.training_data["model_name"]
        fig.suptitle(title, y=1.04)

    def plot_final_result_gradient(self):
        # Plot the gradient of the quantum model.
        fig, ax = plt.subplots(dpi=100)
        if bool(self.quantum_model_gradient_x_prediction_training_history) is True:
            prediction = self.parameter_a_training_history[-1] * self.quantum_model_gradient_x_prediction_training_history[self.counter]
            initial = self.parameter_a_training_history[0] * self.quantum_model_gradient_x_prediction_training_history[1]
            ax.plot(self.training_data['x_data'], prediction, label="Quantum (Trained)")
            ax.plot(self.training_data['x_data'], initial, label="Quantum (Untrained)")
        else:
            statement = "Note: Quantum Gradient is not available " + self.loss_name + "loss functions."
            print(statement)

        ax.plot(self.training_data["x_data"], self.training_data["y1d_data"], label="Teacher")
        ax.set_ylabel(r"Gradient Model $f^{\prime}(x)$")
        ax.set_xlabel(r"$x$")
        ax.legend()
        title = r"Quantum Circuit Learning" + "\n" + self.training_data["model_name"]
        fig.suptitle(title, y=1.04)

