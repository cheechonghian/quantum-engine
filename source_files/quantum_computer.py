"""
Created by Chee Chong Hian.

Below contains Quantum Computer operations for Quantum Circuit Learning.
"""
import numpy as np
import networkx as nx
import math as m
from numpy.random import default_rng
import scipy as spy


# This code is modified from QAOA internship code.
class PureQuantumState:
    """
    Define a pure quantum state.

    There are currently two main ways of defining a pure quantum state from scratch:
        1. Input a vector representation of a quantum state in computational basis.
        2. Use class constructor methods:
            A. all_plus_state
            B. all_minus_state

    Attributes
    ----------
    state_vector: numpy vector
        A quantum state in the computational basis.
    prob_vector: numpy vector
        An elementwise conjugate squared quantum state in the computational basis.
    hash_name_tuple: tuple of the numpy vector
        An immuatable vector that represets "name" of the quantum state.
    """

    def __init__(self, state_vector):
        """
        Define initialization of a pure quantum state from a given vector.

        Parameters
        ----------
        name: str
            The name of the pure quantum state object.
        state_vector: numpy vector
            A quantum state in the computational basis.
        """
        self.state_vector = normalize(np.asarray(state_vector))
        self.prob_vector = np.multiply(np.conj(self.state_vector), self.state_vector)
        self.hash_name_tuple = tuple(self.state_vector)

    def fidelity(self, another_pure_quantum_state):
        """
        Return the fidelity, |<x|x'>|^2, of two quantum states.

        Parameters
        ----------
        another_pure_quantum_state: PureQuantumState object
            Basically, it is just another PureQuantumState object.
        """
        return np.vdot(self.state_vector, another_pure_quantum_state.state_vector) * np.vdot(another_pure_quantum_state.state_vector, self.state_vector)

    def expectation_value_diagonal(self, diagonal_matrix):
        """
        Return the expectation value, <x|C|x>, of an diagonal operator .

        Parameters
        ----------
        MaxCutHamiltonian_object: MaxCutHamiltonian object
            Basically, it is just a MaxCutHamiltonian object that has a diagonal attribute.
        """
        return np.vdot(self.state_vector, np.multiply(diagonal_matrix, self.state_vector))

    def expectation_value(self, Operator_matrix):
        """
        Return the expectation value, <x|O|x>.

        Parameters
        ----------
        Operator_matrix: numpy matrix
            Basically, it is just a numpy matrix
        """
        return np.real(np.vdot(self.state_vector, np.matmul(Operator_matrix, self.state_vector)))

    # Class methods (generators)
    @classmethod
    def all_plus_state(clf, number_of_qubits):
        """
        Construct a PureQuantumState object that represents the all plus state |+>.

        Parameters
        ----------
        number_of_qubits: int
            The number of qubits required in the state. It is equal to the number of vertices of the graph of interest.
        """
        state_vector = [1 for _ in range(2 ** number_of_qubits)]

        return clf(state_vector)

    @classmethod
    def all_minus_state(clf, number_of_qubits):
        """
        Construct a PureQuantumState object that represents the all miunus state |->.

        Parameters
        ----------
        number_of_qubits: int
            The number of qubits required in the state. It is equal to the number of vertices of the graph of interest.
        """
        minus_state = np.array([1, -1], dtype=complex)  # Define a single qubit minus state.

        if number_of_qubits == 1:
            return clf('Initial State: All Minus', minus_state)

        else:
            state_vector = np.array([1, -1], dtype=complex)

            for _ in range(number_of_qubits - 1):
                state_vector = np.kron(state_vector, minus_state)

            return clf(state_vector)

    @classmethod
    def all_zero_qubit(clf, number_of_qubits):
        """
        Construct a PureQuantumState object that represents the all zero state |0>.

        Parameters
        ----------
        number_of_qubits: int
            The number of qubits required in the state. It is equal to the number of vertices of the graph of interest.
        """
        zero_qubit = np.array([1, 0], dtype=complex)
        output_state = zero_qubit
        if number_of_qubits > 1:
            for _ in range(number_of_qubits - 1):
                output_state = np.kron(output_state, zero_qubit)
        return clf(output_state)


def normalize(numpy_vector):
    """
    Normalize a numpy vector.

    Parameters
    ----------
    numpy_vector: numpy 1D array
        The vector that is going to be normalised.
    """
    norm = np.linalg.norm(numpy_vector)
    if norm <= 1.e-15:
        print('The normalization is almost or very close to zero, please check your vector.')
        return numpy_vector
    return numpy_vector / norm


class HamlitonianMixerBlock:
    """
    Define an unitary evolution that is associated with a fully connnected graph with random weights and bias. Random: Uniform distribution between [-1,1].

    Attributes
    ----------
    number_of_qubits: int
        The number of qubits that the quantum computer has.
    total_time: float
        The total time evolution of the Ising Hamiltonian.
    total_depth: int
        The number of times the evolution is repeated, after the rotation block.
    matrix_operator_depth: dict
        Store the ising hamlitonian evolution matrix operator in computational basis.
    graph_store: dict
        Stores the complete graph that corresponds to the Ising Hamiltonian.
    """

    def __init__(self):
        print("Note: 'HamlitonianMixerBlock' class does not have any user config settings.")
        pass

    def config(self, number_of_qubits, total_time, total_depth):
        """
        Parameters
        ----------
        number_of_qubits: int
            The number of qubits that the quantum computer has.
        total_time: float
            The total time evolution of the Ising Hamiltonian.
        total_depth: int
            The number of times the evolution is repeated, after the rotation block.
        """
        self.number_of_qubits = number_of_qubits
        self.total_time = total_time
        self.total_depth = total_depth

        graph_store = {}
        matrix_operator_depth = {}
        for depth_iter in range(total_depth):
            # Generate a complete graph
            my_graph = nx.complete_graph(self.number_of_qubits)

            # Set random weights and bias
            set_weight(my_graph, weight_list='Random_Weights')
            set_bias(my_graph, bias_list='Random_Bias')
            graph_store[depth_iter + 1] = my_graph

            # Generate the ising hamiltonian
            my_ising_ham = ising_matrix(my_graph)

            # Generate and Store the ising evolution
            matrix_operator_depth[depth_iter + 1] = spy.linalg.expm(-1j * my_ising_ham * total_time)

        self.matrix_operator_depth = matrix_operator_depth
        self.graph_store = graph_store

    def show_example(self, number_of_qubits=8, total_time=1, total_depth=2):
        # For debugging purposes, to test out the 'config' class method
        self.config(number_of_qubits, total_time, total_depth)


def set_weight(my_graph, weight_list):
    """
    Set the weights attributes of all edges in the given graph.

    Parameters
    ----------
    my_graph: class
        A graph of interest defined as a instance of a class <networkx.classes.graph.Graph>
    weight_list: string
        A setting to tell the function to use the appropriate subroutines set the correct graphs.
        'No_Weights': Set weight =1 (non-weighted) graphs.
        'Random_Weights': Set random weighted graphs.
    """
    for edge in my_graph.edges():
        if weight_list == 'No_Weights':
            my_graph[edge[0]][edge[1]]['weight'] = 1

        elif weight_list == 'Random_Weights':
            my_graph[edge[0]][edge[1]]['weight'] = default_rng().uniform(-1, 1)

    return None


def set_bias(my_graph, bias_list):
    """
    Set the weights attributes of all edges in the given graph.

    Parameters
    ----------
    my_graph: class
        A graph of interest defined as a instance of a class <networkx.classes.graph.Graph>
    bias_list: string
        A setting to tell the function to use the appropriate subroutines set the correct graphs.
        'No_Bias': Set bias =0 (non-bias) graphs.
        'Random_Bias': Set random bias node graphs.
    """
    for node in my_graph.nodes():
        if bias_list == 'No_Bias':
            my_graph.nodes[node]['bias'] = 0

        elif bias_list == 'Random_Bias':
            my_graph.nodes[node]['bias'] = default_rng().uniform(-1, 1)

    return None


def generate_tensor_ZZ_guide(no_of_qubits, edge_list):
    """
    Generate a list that denotes the quantum gate placement of pauli z gates on a quantum circuit given the edge list.

    Quantum gates:
        Pauli Z  0
        Identity 1
    Example:
        [[0, 0, 1],[1, 0, 0], [0, 1, 0]] --> (Z_1 . Z_2 . I) + (I. Z_2 . Z_3) + (Z_1 . I . Z_3)
        . --> Tensor product

    Paramters
    ---------
    no_of_qubits: int
        The number of qubits on the quantum circuit/ number of vertices on the graph.
    edge_list: list of tuples
        The list containing the list of tuples of labelled vertices (starting from 0).
    """
    # Create empty list to store values
    output_list = []
    temp = []

    for i in edge_list:  # Iterate through each and every edge
        temp = []

        for j in range(no_of_qubits):

            if j in i:
                temp.append(0)  # This will correspond to a pauli-Z

            else:
                temp.append(1)  # This will correspond to a identity

        output_list.append(temp)
    return output_list


def ising_matrix(my_graph):
    """
    Output a numpy ising matrix.

    Parameter
    ---------
    my_graph: networkX graph object
        A graph that contains information of the vertices and edges, it must contain weight information.
    """
    # Prepare the graph information to be used for ZZ.
    number_of_qubits = my_graph.number_of_nodes()
    edge_list = list(my_graph.edges())
    weight_list = [my_graph.edges[edge_iter]['weight'] for edge_iter in edge_list]

    # Prepare matrices to be used for ZZ.
    output_zz_matrix = np.zeros((2 ** number_of_qubits, 2 ** number_of_qubits), dtype=complex)
    output_zz_vector = np.zeros(2 ** number_of_qubits, dtype=complex)
    z_pauli_vector = np.array([1, -1], dtype=complex)
    identity_vector = np.array([1, 1], dtype=complex)

    # Create a placement matrix for ZZ.
    tensor_ZZ_guide = generate_tensor_ZZ_guide(number_of_qubits, edge_list)

    # Create a zz vector for each edge.
    position_count = 0
    for guide in tensor_ZZ_guide:
        temp_pause_first = 1

        for check_gate in guide:
            if check_gate == 0:
                quantum_gate = z_pauli_vector

            elif check_gate == 1:
                quantum_gate = identity_vector

            if temp_pause_first == 1:
                temp_mat = quantum_gate
                temp_pause_first = 0

            else:
                temp_mat = np.kron(temp_mat, quantum_gate)

        # Sum every edge associated ZZ vector
        output_zz_vector += weight_list[position_count] * temp_mat
        position_count += 1

    # Create the ZZ matrix
    for i in range(2 ** number_of_qubits):
        output_zz_matrix[i][i] += output_zz_vector[i]

    # Prepare matrices to be used for X.
    output_x_matrix = np.zeros((2 ** number_of_qubits, 2 ** number_of_qubits), dtype=complex)
    x_pauli = np.array([[0, 1], [1, 0]], dtype=complex)
    identity = np.array([[1, 0], [0, 1]], dtype=complex)

    # Create a placement matrix for X.
    """
    Quantum gates:
        Pauli X  1
        Identity 0
    Example:
        [[1, 0, 0],[0, 1, 0], [0, 0, 1]] --> (X_1 . I . I) + (I. X_2 . I) + (I . I . X_3)
        . --> Tensor product
    """
    identity_general = np.identity(number_of_qubits)

    # Create a X matrix for each node and sum them together
    node_iter = 0
    for row in identity_general:
        temp_pause_first = 1
        for check_gate in row:
            if check_gate == 1:
                quantum_gate = x_pauli * my_graph.nodes[node_iter]['bias']
                node_iter += 1

            elif check_gate == 0:
                quantum_gate = identity

            if temp_pause_first == 1:
                temp_mat = quantum_gate
                temp_pause_first = 0

            else:
                temp_mat = np.kron(temp_mat, quantum_gate)

        output_x_matrix += temp_mat

    return output_x_matrix + output_zz_matrix


class QuantumEncoding:
    """
    To allow generation of matrix operators that encodes the classical data.

    Attributes
    ----------
    select_encoding: str
        See __init__ method
    matrix_operator: numpy array
        The encoding matrix operator that pertains to one single x coordinate data.
    """

    def __init__(self):
        print("Select QuantumEncoding:\n")
        print("1. 'ryasin'        -> RY(arcsin(x))")
        print("2. 'rzacos_ryasin' -> RZ(arccos(x^2))RY(arcsin(x))\n")
        print("To select: use QuantumEncoding().config(select_encoding='_<your_selection_here>_') \n \n")

    def config(self, select_encoding):
        """
        For users to set their preferred encoding.

        Parameters
        ----------
        select_encoding: str
            See __init__ method
        """
        self.select_encoding = select_encoding
        encode_selection = get_encoding()
        self.encode_model = encode_selection[self.select_encoding]()
        self.number_of_encode_rotate_gates = self.encode_model.number_rotate_gates

    def encode_data(self, number_of_qubits, teacher_x_single_data):
        """
        Generates the operator matrix that will encode the data into the quantum state.

        Parameters
        ----------
        number_of_qubits : int
            The total number of qubits in the quantum circuit.
        teacher_x_single_data : float
            Just one x data value from the TeacherModel.
        """
        encode_operator_single_qubit = self.encode_model.get_single_qubit_matrix_operator(teacher_x_single_data)

        if number_of_qubits == 1:
            self.matrix_operator = encode_operator_single_qubit
        else:
            temp_mat = encode_operator_single_qubit
            for _ in range(number_of_qubits - 1):
                temp_mat = np.kron(temp_mat, encode_operator_single_qubit)

            self.matrix_operator = temp_mat

    def encode_data_shift_x(self, number_of_qubits, teacher_x_single_data):
        # generate
        matrix_operator_shift_dict = {}
        for qubit_shift_iter in range(number_of_qubits):

            matrix_operator_shift_dict[qubit_shift_iter+1] = {}
            for encode_rotate_gate_shift_iter in range(self.number_of_encode_rotate_gates):

                matrix_operator_shift_dict[qubit_shift_iter+1][encode_rotate_gate_shift_iter+1] = {}
                for x_shift in ["plus", "minus"]:
                    encode_operator_single_qubit_shift = self.encode_model.get_single_qubit_matrix_operator_shift(teacher_x_single_data, encode_rotate_gate_shift_iter, x_shift)

                    # Build the correct encoding matrix operator shifted for every qubit and rotation qubit
                    if number_of_qubits == 1:
                        matrix_operator_shift_dict[qubit_shift_iter+1][encode_rotate_gate_shift_iter+1][x_shift] = encode_operator_single_qubit_shift
                    else:
                        place_list = np.zeros(number_of_qubits)
                        place_list[qubit_shift_iter] += 1
                        temp_pause_first = 1
                        for qubit_check_iter in range(number_of_qubits - 1):
                            if temp_pause_first == 1:
                                if place_list[qubit_check_iter] == 1:
                                    temp_mat = encode_operator_single_qubit_shift
                                else:
                                    temp_mat = self.encode_model.get_single_qubit_matrix_operator(teacher_x_single_data)
                                temp_pause_first = 0

                            else:
                                if place_list[qubit_check_iter] == 1:
                                    second_mat = encode_operator_single_qubit_shift
                                else:
                                    second_mat = self.encode_model.get_single_qubit_matrix_operator(teacher_x_single_data)

                            temp_mat = np.kron(temp_mat, second_mat)

                        matrix_operator_shift_dict[qubit_shift_iter+1][encode_rotate_gate_shift_iter+1][x_shift] = temp_mat

        self.matrix_operator_shift_dict = matrix_operator_shift_dict

    def get_gradients_for_shift_x(self, number_of_qubits, output_result_shift_x):
        # To generate
        """
        shift_x_result = {"output_gradient_x_data": output_gradient_x_data,
                          "gradient_x_parameter_dict" : gradient_x_parameter_dict,
                          }

        return shift_x_result
        """
        pass


def get_encoding():
    dict_of_models = {"ryasin": ryasin,
                      "rzacos_ryasin": rzacos_ryasin,
                      }
    return dict_of_models


# def get_encoding_number_rotate_gates():
#     dict_of_models = {"ryasin": 1,
#                       "rzacos_ryasin": 2,
#                       }
#     return dict_of_models


# def get_encoding_number_rotate_gates():
#     dict_of_models = {"ryasin": 1,
#                       "rzacos_ryasin": 2,
#                       }
#     return dict_of_models


# def ryasin(teacher_x_single_data):
#     """
#     Generate the operator matrix of Pauli_Y(arcsin(x)).

#     Parameters
#     ----------
#     teacher_x_single_data : float
#         Just one x data value from the TeacherModel.
#     """
#     arcsin_x = m.asin(teacher_x_single_data)
#     RY_arcsin_x = np.cos(arcsin_x / 2) * np.eye(2) - 1j * np.sin(arcsin_x / 2) * np.array([[0, -1j], [1j, 0]])

#     return RY_arcsin_x


# def rzacos_ryasin(teacher_x_single_data):
#     """
#     Generate the operator matrix of Pauli_Z(arccos(x^2)) * Pauli_Y(arcsin(x)).

#     Parameters
#     ----------
#     teacher_x_single_data : float
#         Just one x data value from the TeacherModel.
#     """
#     arcsin_x = m.asin(teacher_x_single_data)
#     arccos_x2 = m.acos(teacher_x_single_data**2)
#     RY_arcsin_x = np.cos(arcsin_x / 2) * np.eye(2) - 1j * np.sin(arcsin_x / 2) * np.array([[0, -1j], [1j, 0]])
#     RZ_arccos_x2 = np.cos(arccos_x2 / 2) * np.eye(2) - 1j * np.sin(arccos_x2 / 2) * np.array([[1, 0], [0, -1]])

#     return np.matmul(RZ_arccos_x2, RY_arcsin_x)


class ryasin:
    def __init__(self):
        self.number_rotate_gates = 1

    def get_single_qubit_matrix_operator(self, teacher_x_single_data):
        """
        Generate the operator matrix of Pauli_Y(arcsin(x)).

        Parameters
        ----------
        teacher_x_single_data : float
            Just one x data value from the TeacherModel.
        """
        arcsin_x = m.asin(teacher_x_single_data)
        RY_arcsin_x = np.cos(arcsin_x / 2) * np.eye(2) - 1j * np.sin(arcsin_x / 2) * np.array([[0, -1j], [1j, 0]])

        return RY_arcsin_x

    def calculate_x_gradient(self, teacher_x_single_data, gradient_PCR):
        """
        Calculate the full x gradient.

        gradient_PCR : list

        """
        x_gradient = gradient_PCR[0] / m.sqrt(1-teacher_x_single_data**2)

        return x_gradient

    def get_single_qubit_matrix_operator_shift(self, teacher_x_single_data, encode_rotate_gate_shift_iter, x_shift):
        """
        Generate the operator matrix shifted of Pauli_Y(arcsin(x) +- pi/2 ).

        Parameters
        ----------
        teacher_x_single_data : float
            Just one x data value from the TeacherModel.
        """
        assert encode_rotate_gate_shift_iter == 0, "'encode_rotate_gate_shift_iter' is not zero"
        if x_shift == 'plus':
            angle_shift = m.pi/2
        if x_shift == 'minus':
            angle_shift = -m.pi/2

        arcsin_x = m.asin(teacher_x_single_data)
        arcsin_x += angle_shift
        RY_arcsin_x = np.cos(arcsin_x / 2) * np.eye(2) - 1j * np.sin(arcsin_x / 2) * np.array([[0, -1j], [1j, 0]])

        return RY_arcsin_x


class rzacos_ryasin:
    def __init__(self):
        self.number_rotate_gates = 2

    def get_single_qubit_matrix_operator(self, teacher_x_single_data):
        """
        Generate the operator matrix of Pauli_Z(arccos(x^2)) * Pauli_Y(arcsin(x)).

        Parameters
        ----------
        teacher_x_single_data : float
            Just one x data value from the TeacherModel.
        """
        arcsin_x = m.asin(teacher_x_single_data)
        arccos_x2 = m.acos(teacher_x_single_data**2)
        RY_arcsin_x = np.cos(arcsin_x / 2) * np.eye(2) - 1j * np.sin(arcsin_x / 2) * np.array([[0, -1j], [1j, 0]])
        RZ_arccos_x2 = np.cos(arccos_x2 / 2) * np.eye(2) - 1j * np.sin(arccos_x2 / 2) * np.array([[1, 0], [0, -1]])

        return np.matmul(RZ_arccos_x2, RY_arcsin_x)

    def calculate_x_gradient(self, teacher_x_single_data, gradient_PCR):
        """
        Calculate the full x gradient.

        gradient_PCR : list

        """
        x_gradient = x_gradient = gradient_PCR[0] / m.sqrt(1-teacher_x_single_data**2) - 2*teacher_x_single_data*gradient_PCR[1] / m.sqrt(1-teacher_x_single_data**4)

        return x_gradient

    def get_single_qubit_matrix_operator_shift(self, teacher_x_single_data, encode_rotate_gate_shift_iter, x_shift):
        """
        Generate the operator matrix shifted of Pauli_Z(arccos(x^2) +- pi/2 ) * Pauli_Y(arcsin(x) +- pi/2 )

        Parameters
        ----------
        teacher_x_single_data : float
            Just one x data value from the TeacherModel.
        """
        if x_shift == 'plus':
            angle_shift = m.pi/2
        if x_shift == 'minus':
            angle_shift = -m.pi/2

        arcsin_x = m.asin(teacher_x_single_data)
        arccos_x2 = m.acos(teacher_x_single_data**2)

        if encode_rotate_gate_shift_iter == 0:
            arcsin_x += angle_shift
        if encode_rotate_gate_shift_iter == 1:
            arccos_x2 += angle_shift

        RY_arcsin_x = np.cos(arcsin_x / 2) * np.eye(2) - 1j * np.sin(arcsin_x / 2) * np.array([[0, -1j], [1j, 0]])
        RZ_arccos_x2 = np.cos(arccos_x2 / 2) * np.eye(2) - 1j * np.sin(arccos_x2 / 2) * np.array([[1, 0], [0, -1]])

        return np.matmul(RZ_arccos_x2, RY_arcsin_x)

class SingleQubitRotationBlock:
    """
    Allow generation and related calculations of the Single Qubit XZX Rotation Matrix Operators.

    Attributes
    ----------
    number_of_qubits: int
        The total number of qubits in the quantum circuit.
    total_depth: int
        The number of times the evolution is repeated, before the mixer/entangling block.
    parameter_dict: dict
         Contains all angular parmaters organised (NOT sorted) by depth, qubit, and rotate_gate_iter.
    matrix_operator_depth: dict
        Contains all matrix operators organised (NOT sorted) by depth
    """

    def __init__(self):
        print("Note: 'SingleQubitRotationBlock' class does not have any user config settings.")
        pass

    def config(self, number_of_qubits, total_depth):
        self.number_of_qubits = number_of_qubits
        self.total_depth = total_depth

        # For the xzx model, the total number of parameters is= 3 * number_of_qubits * total_depth.
        # We shall randomly initialise the angular rotation parameters when first created.
        parameter_dict = {}  # Prepare two empty dictionary to store the angular rotation parameters
        matrix_operator_depth = {}  # and to store matrix operator at every depth
        for depth_iter in range(total_depth):
            parameter_dict[depth_iter+1] = {}
            first_qubit = 1

            for qubit_iter in range(number_of_qubits):
                parameter_dict[depth_iter+1][qubit_iter+1] = {}
                first_gate = 1

                for rotate_gate_iter in range(3):
                    # Note that rotation gates RxRzRx are represented as ("rotate_gate_iter"+1): 1,2,3
                    # Randomly choose angular parameters from a uniform distribution [-pi/2,pi/2] when first initialized.
                    parameter_dict[depth_iter+1][qubit_iter+1][rotate_gate_iter+1] = default_rng().uniform(0, 2*m.pi)

                    # Carefully construct the matrix operator gate by gate for a qubit
                    if ((rotate_gate_iter+1) == 1) or ((rotate_gate_iter+1) == 3):
                        rotate_qubit = rotate_x(parameter_dict[depth_iter+1][qubit_iter+1][rotate_gate_iter+1])
                    else:
                        rotate_qubit = rotate_z(parameter_dict[depth_iter+1][qubit_iter+1][rotate_gate_iter+1])

                    if first_gate == 1:
                        first_gate = 0
                        temp_rotate_qubit = rotate_qubit
                    else:
                        temp_rotate_qubit = np.matmul(rotate_qubit, temp_rotate_qubit)

                # Carefully apply kronecker(tensor) products of single qubit rotations
                if first_qubit == 1:
                    first_qubit = 0
                    temp_rotate = temp_rotate_qubit
                else:
                    temp_rotate = np.kron(temp_rotate, temp_rotate_qubit)

            matrix_operator_depth[depth_iter+1] = temp_rotate

        self.parameter_dict = parameter_dict
        self.matrix_operator_depth = matrix_operator_depth

    def parameter_shift(self, shift, depth_shift, qubit_shift, rotate_gate_shift):
        """
        Return a whole matrix operator shifted at a particulatar depth. Does not affect the current parameters or matrix operators.

        Parameters
        ----------
        shift: str
            "plus": add pi/2 angular shift to the targeted rotation gate.
            "minus": add -pi/2 angular shift to the targeted rotation gate.
        depth_shift: int
            The depth that the target parameter is in. *** depth_shift starts from one. ***
        qubit_shift: int
            The qubit that the target parameter is in. *** qubit_shift starts from one. ***
        rotate_gate_shift: int
            The rotation gate that the target parameter is in. *** Note that rotation gates RxRzRx are represented as ("rotate_gate_iter"+1): 1,2,3 ***
        """
        # Create the angular parameter shift.
        if shift == 'plus':
            angle_shift = m.pi/2
        if shift == 'minus':
            angle_shift = -m.pi/2

        first_qubit = 1
        for qubit_iter in range(self.number_of_qubits):
            first_gate = 1

            for rotate_gate_iter in range(3):
                # Note that rotation gates RxRzRx are represented as ("rotate_gate_iter"+1): 1,2,3
                angle = self.parameter_dict[depth_shift][qubit_iter+1][rotate_gate_iter+1]

                # Apply the parameter shift to the selected gate.
                if (qubit_shift == (qubit_iter+1)) and (rotate_gate_shift == (rotate_gate_iter+1)):
                    angle += angle_shift

                # Carefully construct the matrix operator gate by gate for a qubit
                if ((rotate_gate_iter+1) == 1) or ((rotate_gate_iter+1) == 3):
                    rotate_qubit = rotate_x(angle)
                else:
                    rotate_qubit = rotate_z(angle)

                if first_gate == 1:
                    first_gate = 0
                    temp_rotate_qubit = rotate_qubit
                else:
                    temp_rotate_qubit = np.matmul(rotate_qubit, temp_rotate_qubit)

            # Carefully apply kronecker(tensor) products of single qubit rotations
            if first_qubit == 1:
                first_qubit = 0
                temp_rotate = temp_rotate_qubit
            else:
                temp_rotate = np.kron(temp_rotate, temp_rotate_qubit)

        return temp_rotate

    def update_params(self, new_params):
        """
        Update all the angular parameters.

        Parameters
        ----------
        new_params: dict
            Contains new angular parameters.
        """
        matrix_operator_depth = {}  # To store new matrix operator at every depth
        for depth_iter in range(self.total_depth):
            first_qubit = 1

            for qubit_iter in range(self.number_of_qubits):
                first_gate = 1

                for rotate_gate_iter in range(3):
                    # Note that rotation gates RxRzRx are represented as ("rotate_gate_iter"+1): 1,2,3
                    # Carefully construct the matrix operator gate by gate for a qubit
                    if ((rotate_gate_iter+1) == 1) or ((rotate_gate_iter+1) == 3):
                        rotate_qubit = rotate_x(new_params[depth_iter+1][qubit_iter+1][rotate_gate_iter+1])
                    else:
                        rotate_qubit = rotate_z(new_params[depth_iter+1][qubit_iter+1][rotate_gate_iter+1])

                    if first_gate == 1:
                        first_gate = 0
                        temp_rotate_qubit = rotate_qubit
                    else:
                        temp_rotate_qubit = np.matmul(rotate_qubit, temp_rotate_qubit)

                # Carefully apply kronecker(tensor) products of single qubit rotations
                if first_qubit == 1:
                    first_qubit = 0
                    temp_rotate = temp_rotate_qubit
                else:
                    temp_rotate = np.kron(temp_rotate, temp_rotate_qubit)

            matrix_operator_depth[depth_iter+1] = temp_rotate

        self.parameter_dict = new_params
        self.matrix_operator_depth = matrix_operator_depth


def rotate_x(angle):
    return np.array([[m.cos(angle/2), -1j*m.sin(angle/2)], [-1j*m.sin(angle/2), m.cos(angle/2)]], dtype='complex_')


def rotate_y(angle):
    return np.array([[m.cos(angle/2), -1*m.sin(angle/2)], [m.sin(angle/2), m.cos(angle/2)]], dtype='complex_')


def rotate_z(angle):
    return np.array([[m.cos(angle/2) - 1j*m.sin(angle/2), 0], [0, m.cos(angle/2) + 1j*m.sin(angle/2)]], dtype='complex_')


class QuantumMeasurement:
    """
    To allow the generation of measurement operators.

    Attributes
    ----------
    select_measurement: str
            See __init__ method
    matrix_operator: numpy array
        The obeservable matrix.
    """

    def __init__(self):
        print("Select QuantumMeasurement:\n")
        print("1. 'first_qubit_Z'  -> To measure <Z> expectation of the first qubit.\n")
        print("To select: use QuantumMeasurement().config(select_measurement='_<your_selection_here>_') \n \n")

    def config(self, select_measurement):
        """
        For users to set their preferred measurement.

        Parameters
        ----------
        select_measurement: str
            See __init__ method
        """
        self.select_measurement = select_measurement

    def create_observable(self, number_of_qubits):
        """
        Generate the observable matrix.

        Parameters
        ----------
        number_of_qubits: int
            The total number of qubits in the quantum circuit.
        """
        measure_selection = get_measurement()
        measure_model = measure_selection[self.select_measurement]
        self.matrix_operator = measure_model(number_of_qubits)


def get_measurement():
    dict_of_models = {"first_qubit_Z": first_qubit_Z,
                      }
    return dict_of_models


def first_qubit_Z(number_of_qubits):
    """
    Generate the observable matrix that corresponds to Z measurement in the first qubit.

    Parameters
    ----------
    number_of_qubits: int
        The total number of qubits in the quantum circuit.
    """
    # Initalise the numpy array
    observable = np.zeros((2 ** number_of_qubits, 2 ** number_of_qubits))

    # Accelerate the computation by exploiting the diagonals.
    z_pauli_vector = np.array([1, -1])
    identity_vector = np.array([1, 1])
    if number_of_qubits == 1:
        for i in range(2 ** number_of_qubits):
            observable[i][i] += z_pauli_vector[i]

    else:
        temp = np.kron(z_pauli_vector, identity_vector)
        for _ in range(number_of_qubits - 2):
            temp = np.kron(temp, identity_vector)

        for i in range(2 ** number_of_qubits):
            observable[i][i] += temp[i]

    return observable


class QuantumComputer:
    """
    To the allow high level control over of the quantum computation.

    Attributes
    ----------
    number_of_qubits: int
        The number of qubit that the quantum computer uses for simulation.
    depth: int
        The number of repeating layers of 'HamlitonianMixerBlock' and 'SingleQubitRotationBlock'.
    Encode : 'QuantumEncoding' class
    A : 'HamlitonianMixerBlock' class
    B : 'SingleQubitRotationBlock' class
    Observable: 'QuantumMeasurement' class
    """

    def __init__(self):
        print("Note: 'QuantumComputer' uses the Quantum Circuit Learning Model.\n")
        print("Config Settings Available:\n")
        print("* number_of_qubits [int]: The number of qubit that the quantum computer uses for simulation.")
        print("* depth [int]: The number of repeating layers of 'HamlitonianMixerBlock' and 'SingleQubitRotationBlock'.\n")
        print("To select: use QuantumComputer().config() \n \n")
        print("Required Component Inputs to QuantumComputer:\n")
        print("1. Encode: 'QuantumEncoding' class")
        print("2. A : 'HamlitonianMixerBlock' class")
        print("3. B : 'SingleQubitRotationBlock' class")
        print("4. Observable: 'QuantumMeasurement' class\n")
        print("To input: use QuantumComputer().input(Encode, A, B, Observable)")
        print("Note: Please run '.config()' first before running '.inputs()' \n")

    def config(self, number_of_qubits=3, depth=2, ):
        """
        User setting for quantum computer simulation.

        Parameters
        ----------
        number_of_qubits: int
            The number of qubit that the quantum computer uses for simulation.
        depth: int
            The number of repeating layers of 'HamlitonianMixerBlock' and 'SingleQubitRotationBlock'.
        """
        self.number_of_qubits = number_of_qubits
        self.depth = depth

    def inputs(self, Encode, A, B, Observable,):
        """
        User inputs of quantum components that defines the quantum computation.

        Parameters
        ----------
        Encode : 'QuantumEncoding' class
        A : 'HamlitonianMixerBlock' class
        B : 'SingleQubitRotationBlock' class
        Observable: 'QuantumMeasurement' class
        """
        # Setup the quantum components
        A.config(self.number_of_qubits, total_time=10, total_depth=self.depth)
        B.config(self.number_of_qubits, total_depth=self.depth)
        Observable.create_observable(self.number_of_qubits)

        self.Encode = Encode
        self.A = A
        self.B = B
        self.Observable = Observable
        self.initial_state = PureQuantumState.all_zero_qubit(self.number_of_qubits)

    def run_qc(self, teacher_x_single_data, shift_x):

        self.Encode.encode_data(self.number_of_qubits, teacher_x_single_data)
        encoded_state_vector = np.matmul(self.Encode.matrix_operator, self.initial_state.state_vector)
        output_result = self.run_qc_main(self, teacher_x_single_data, encoded_state_vector)

        # Get the gradients wrt inputs and wrt parameters.
        if shift_x is True:
            self.Encode.encode_data_shift_x(self.number_of_qubits, teacher_x_single_data)

            output_result_shift_x = {}
            for qubit_shift_iter in range(self.number_of_qubits):

                output_result_shift_x[qubit_shift_iter+1] = {}
                for encode_rotate_gate_shift_iter in range(len(self.Encode.number_of_encode_rotate_gates)):

                    output_result_shift_x[qubit_shift_iter+1][encode_rotate_gate_shift_iter+1] = {}
                    for x_shift in ["plus", "minus"]:
                        encoded_state_vector_shift_x = np.matmul(self.Encode.matrix_operator_shift_dict[qubit_shift_iter+1][encode_rotate_gate_shift_iter+1][x_shift], self.initial_state.state_vector)
                        output_result_shift_x[qubit_shift_iter+1][encode_rotate_gate_shift_iter+1][x_shift] = self.run_qc_main(self, teacher_x_single_data, encoded_state_vector_shift_x)

            shift_x_result = self.Encode.get_gradients_for_shift_x(self.number_of_qubits, output_result_shift_x)
            output_result["output_gradient_x_data"] = shift_x_result["output_gradient_x_data"]
            output_result["gradient_x_parameter_dict"] = shift_x_result["gradient_x_parameter_dict"]

        return output_result

    def run_qc_main(self, teacher_x_single_data, encoded_state_vector):
        """
        Run the quantum computer given the single x data coordinate, the encoded state and measure the result.

        Paramters
        ---------
        teacher_x_single_data : float
            Just one x data value from the TeacherModel.

        encoded_state_vector: numpy vector
            The quantum wavevector that represents the quantum state encoded with classical data

        Output
        ------
        output_result : dict
            Contains the measurement result and the gradient wrt to the parameters in B.
        """
        # AB_Repeat
        temp_state_vector = encoded_state_vector
        for depth_iter in range(self.depth):
            temp_state_vector = np.matmul(self.A.matrix_operator_depth[depth_iter + 1], temp_state_vector)
            temp_state_vector = np.matmul(self.B.matrix_operator_depth[depth_iter + 1], temp_state_vector)

        # Measurement
        final_state = PureQuantumState(temp_state_vector)
        output_data = final_state.expectation_value(self.Observable.matrix_operator)  # The predicted value by the quantum model.

        # Get the gradients of model wrt each parameter
        gradient_parameter_dict = {}  # Prepare a empty dictionary to store the gradient for e
        for depth_shift_iter in range(self.depth):

            gradient_parameter_dict[depth_shift_iter+1] = {}
            for qubit_shift_iter in range(self.number_of_qubits):

                gradient_parameter_dict[depth_shift_iter+1][qubit_shift_iter+1] = {}
                for rotate_gate_shift_iter in range(3):

                    output_data_shift = []
                    for shift in ["plus", "minus"]:

                        # AB_Repeat
                        temp_state_vector = encoded_state_vector
                        for depth_iter in range(self.depth):
                            temp_state_vector = np.matmul(self.A.matrix_operator_depth[depth_iter + 1], temp_state_vector)

                            # Apply the correct B operator.
                            if depth_shift_iter == depth_iter:
                                B_matrix_operator = self.B.parameter_shift(shift, depth_shift_iter+1, qubit_shift_iter+1, rotate_gate_shift_iter+1)
                            else:
                                B_matrix_operator = self.B.matrix_operator_depth[depth_iter + 1]

                            temp_state_vector = np.matmul(B_matrix_operator, temp_state_vector)

                        # Measurement
                        final_state = PureQuantumState(temp_state_vector)
                        output_data_shift.append(final_state.expectation_value(self.Observable.matrix_operator))

                    # Apply the parameter shift rule and store the result.
                    gradient_parameter_dict[depth_shift_iter+1][qubit_shift_iter+1][rotate_gate_shift_iter+1] = 0.5 * (output_data_shift[0] - output_data_shift[1])

        output_result = {"gradient_parameter_dict": gradient_parameter_dict,
                         "output_data": output_data, }

        return output_result

    def update_params(self, new_params):
        """
        Update the parameters of B.

        Parameters
        ----------
        new_params: dict
            Contains new angular parameters.
        """
        self.B.update_params(new_params)
