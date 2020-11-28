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


class Graph:
    """
    Define a graph class.

    This class uses the networkX graph package to handle graph operations and funtions.
    However, one must be careful to DEFINE a  graph.
    TO DEFINE a Graph:
        1. Using networkX graph generators
            a) Only use non-weighted and non-directed graph generators.
            b) Put the generated graph under argument name: 'my_networkX_graph'.
            c) Set argument name 'weight_list' to either 'No_Weights' (all weights = 1) or 'Random_Weights' (sampled from uniform  distribution -1 to 1))
            NOTE: Custom Weights are not possible.
        2. Custom graph creation (Preferred)
            a) Define the 'number_of_vertices'.
            b) Define the 'edge_list' as a list of tuples of two vertices. WARNING: VERTEX LABELS BE INTEGERS AND START FROM 0.
            c) Define the 'weight_list' as a list of floats. (Must correspond to the 'edge_list')

    Attributes
    ----------
    networkX_graph: a networkX_graph object class
        The  problem graph of interest. Extra information may be added during initialization.
    accelerate: Boolean
        To be set true if one wishes to use Greedy Newton. It will create the necessary data structure that will be used to store Pauli composition expectation values.
    """

    def __init__(self, bias_list=None, weight_list=None, my_networkX_graph=None):
        """
        Define initialization of  graph class that allows easier graph processing.

        Parameters
        ----------
        name: str
            The name the  Graph class
        number_of_vertices: int
            The number of vertices in the graph
        edge_list: list of tuples of two vertices
            A list containing edge information, edges are defined using tuple of two vertices labelled (starting from index 0)
        bias_list: list of floats <<-- or -->> string
            (Custom graph creation)
                A list containing numbers(floats) that represent bias.
            <<-- or -->>
            (Using networkX graph generators)
                A string that will dictate the type of weight that will be set later. Accepted strings 'No_Bias' (all bias = 0) or 'Random_Bias' (sampled from uniform  distribution -1 to 1)
        weight_list: list of floats <<-- or -->> string
            (Custom graph creation)
                A list containing numbers(floats) that represent weights.
            <<-- or -->>
            (Using networkX graph generators)
                A string that will dictate the type of weight that will be set later. Accepted strings 'No_Weights' (all weights = 1) or 'Random_Weights' (sampled from uniform  distribution -1 to 1))
        my_networkX_graph: A networkX graph class object (non-weighted and non-directed)
            A graph generated using networkX non-weighted and non-directed graph generators.
        """
        # Define the grap
        set_weight(my_networkX_graph, weight_list)
        set_bias(my_networkX_graph, bias_list)
        self.networkX_graph = my_networkX_graph

        return


# Helper Functions
def define_networkX_graph(number_of_vertices, edge_list, weight_list=None):
    """
    Create a networkX graph based on the number of vertices, edges and weights(if any).

    Parameters
    ----------
    number_of_vertices: int
        The number of vertices in the graph
    edge_list: list of tuples of two vertices
            A list containing edge information, edges are defined using tuple of two vertices labelled (starting from index 0)
    weight_list: list of floats
        A list containing numbers(floats) that represent weights.
    """
    print('Loading Graph...')
    myNewGraph = nx.Graph()
    print('Adding vertices...')
    myNewGraph.add_nodes_from([i for i in range(number_of_vertices)])
    print(f'Number of vertices is {number_of_vertices}')
    print('Adding edges...')

    for edge in edge_list:
        if edge[0] >= number_of_vertices or edge[1] >= number_of_vertices:
            print('The edge information contains extra undefined vertice(s). Please make sure the number of vertices is correct.')
            return None
        else:
            myNewGraph.add_edge(edge[0], edge[1])

    print('Edges loaded, edge order may differ.')
    print('Loading weights...')

    if weight_list is None:
        set_weight(myNewGraph, 'No_Weights', seed=None)
        print('No weight information available. Defaulting to all weights equal to one...')

    elif weight_list == 'Random':
        print('Samling weights from a uniform distribution over [0,1). Other random distributions to be added soon...')
        set_weight(myNewGraph, 'Random', seed=None)

    elif len(weight_list) != len(edge_list):
        set_weight(myNewGraph, 'No_Weights', seed=None)
        print('Number of weights does not match the number of edges. Please make sure the number of weights is correct.')
        return None

    else:
        # Set custom weight information. Please use list of floats or int.
        for edge in edge_list:
            myNewGraph[edge[0]][edge[1]]['weight'] = float(weight_list[edge_list.index(edge)])
        print('Weights loaded, weight order may differ.')

    print('Graph is loaded and ready. Please check your edges for any unintended additional/missing edges carefully.')
    return myNewGraph


def set_weight(myGraph, weight_list):
    """
    Set the weights attributes of all edges in the given graph.

    Parameters
    ----------
    myGraph: class
        A graph of interest defined as a instance of a class <networkx.classes.graph.Graph>
    weight_list: string
        A setting to tell the function to use the appropriate subroutines set the correct graphs.
        'No_Weights': Set weight =1 (non-weighted) graphs.
        'Random_Weights': Set random weighted graphs.
    """
    for edge in myGraph.edges():
        if weight_list == 'No_Weights':
            myGraph[edge[0]][edge[1]]['weight'] = 1

        elif weight_list == 'Random_Weights':
            myGraph[edge[0]][edge[1]]['weight'] = default_rng().uniform(-1, 1)

    return None


def set_bias(myGraph, bias_list):
    """
    Set the weights attributes of all edges in the given graph.

    Parameters
    ----------
    myGraph: class
        A graph of interest defined as a instance of a class <networkx.classes.graph.Graph>
    bias_list: string
        A setting to tell the function to use the appropriate subroutines set the correct graphs.
        'No_Bias': Set bias =0 (non-bias) graphs.
        'Random_Bias': Set random bias node graphs.
    """
    for node in myGraph.nodes():
        if bias_list == 'No_Bias':
            myGraph.nodes[node]['bias'] = 0

        elif bias_list == 'Random_Bias':
            myGraph.nodes[node]['bias'] = default_rng().uniform(-1, 1)

    return None


class HamlitonianMixerBlock:
    """
    Define an unitary evolution that is associated with a fully connnected graph with random weights and bias. Random: Uniform distribution between [-1,1].

    Attributes
    ----------
    graph: networkx graph object
        A weighted graph of interest. If the graph is unweighted must be converted to weighted graphs treating all weights equal to one.
    matrix: numpy matrix
        The unitary driven by an Ising Hamiltonian Mixer operator in computational basis.
    """

    def __init__(self):
        """
        Define initialization of the maxcut operator.

        Parameters
        ----------
        name: str
            The name of the Mixer operator object.
        graph: networkx graph object
            A weighted graph of interest. If the graph is unweighted must be converted to weighted graphs treating all weights equal to one.
        use_negative: bool
            Set whether if the C should be a negative maxcut hamiltonian or just a normal one. To align with physical quantum theory, use negative == True.
        """
        pass

    def config(self, number_of_qubits, total_time, total_depth):
        """
        Parameters
        ----------
        number_of_qubits : int
        total_time : float
        total_depth: int
        """
        self.number_of_qubits = number_of_qubits
        self.total_time = total_time
        self.total_depth = total_depth
        matrix_operator_depth = {}
        for depth_iter in range(total_depth):
            # Generate a complete graph
            my_graph = nx.complete_graph(self.number_of_qubits)

            # Generate Graph class to support future features
            myGraph = Graph(bias_list='Random_Bias', weight_list='Random_Weights', my_networkX_graph=my_graph)

            # Generate the ising hamiltonian
            my_ising_ham = ising_matrix(myGraph.networkX_graph)

            # Store the ising evolution
            matrix_operator_depth[depth_iter + 1] = spy.linalg.expm(-1j * my_ising_ham * total_time)

        self.matrix_operator_depth = matrix_operator_depth


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


def ising_matrix(myGraph):
    """
    Output a numpy ising matrix.

    Parameter
    ---------
    myGraph: networkX graph object
        A graph that contains information of the vertices and edges, it must contain weight information.
    """
    # Prepare the graph information to be used for ZZ.
    number_of_qubits = myGraph.number_of_nodes()
    edge_list = list(myGraph.edges())
    weight_list = [myGraph.edges[edge_iter]['weight'] for edge_iter in edge_list]

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
                quantum_gate = x_pauli * myGraph.nodes[node_iter]['bias']
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
    Attributes
    ----------

    """
    def __init__(self):
        """
        Parameters
        ----------

        """

    def config(self, select_encoding):
        """
        Parameters
        ----------

        """
        self.select_encoding = select_encoding

    def encode_data(self, no_of_qubits, teacher_x_data):
        """
        Generates the operator matrix that will encode the data into the quantum state.

        Parameters
        ----------
        number_of_qubits : int
            The total number of qubits in the quantum circuit.
        teacher_x_single_data : float
            Just one x data value from the TeacherModel.
        """
        if self.select_encoding == "RY_arcsin":
            self.matrix_operator = encoding_RY_arcsin(no_of_qubits, teacher_x_data)


def encoding_RY_arcsin(number_of_qubits, teacher_x_single_data):
    """
    Generate the operator matrix that products of Pauli_Y(arcsin(x)).

    Parameters
    ----------
    number_of_qubits : int
        The total number of qubits in the quantum circuit.
    teacher_x_single_data : float
        Just one x data value from the TeacherModel.
    """
    arcsin_x = m.asin(teacher_x_single_data)
    RY_arcsin_x = np.cos(arcsin_x / 2) * np.eye(2) - 1j * np.sin(arcsin_x / 2) * np.array([[0, 1j], [1j, 0]])
    if number_of_qubits == 1:
        return RY_arcsin_x
    else:
        temp_mat = RY_arcsin_x
        for _ in range(number_of_qubits - 1):
            temp_mat = np.kron(temp_mat, RY_arcsin_x)

        return temp_mat


class SingleQubitRotationBlock:
    def __init__(self):
        pass

    def config(self, number_of_qubits, total_depth):
        self.number_of_qubits = number_of_qubits
        self.total_depth = total_depth

        # For the xzx model, the total number of parameters is= 3 * number_of_qubits * total_depth.
        # We shall randomly initialise the angular rotation parameters when first created.
        parameter_dict = {}  # Prepare two empty dictionary to store the angular rotation parameters
        matrix_operator_depth = {}  # and to store matrix operator
        for depth_iter in range(total_depth):
            parameter_dict[depth_iter+1] = {}
            first_qubit = 1

            for qubit_iter in range(number_of_qubits):
                parameter_dict[depth_iter+1][qubit_iter+1] = {}
                first_gate = 1

                for rotate_gate_iter in range(3):
                    # Note that rotation gates RxRzRx are represented as ("rotate_gate_iter"+1): 1,2,3
                    # Randomly choose angular parameters from a uniform distribution [-pi/2,pi/2]
                    parameter_dict[depth_iter+1][qubit_iter+1][rotate_gate_iter+1] = default_rng().uniform(-m.pi/2, m.pi/2)

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
        Returns a whole matrix operator shifted at a particulatar depth.

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
        matrix_operator_depth = {}  # and to store matrix operator
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
    return np.array([[m.cos(angle/2), -1j*m.sin(angle/2)], [-1j*m.sin(angle/2), m.cos(angle/2)]],dtype = 'complex_')

def rotate_y(angle):
    return np.array([[m.cos(angle/2), -1*m.sin(angle/2)], [m.sin(angle/2), m.cos(angle/2)]],dtype = 'complex_')

def rotate_z(angle):
    return np.array([[m.cos(angle/2) - 1j*m.sin(angle/2), 0], [0, m.cos(angle/2) + 1j*m.sin(angle/2)]],dtype = 'complex_')

class QuantumMeasurement:
    """
    Attributes.

    ----------


    """

    def __init__(self):
        """
        Parameters
        ----------

        """
        pass

    def config(self, select_measurement):
        """
        Parameters
        ----------

        """
        self.select_measurement = select_measurement

    def create_observable(self, number_of_qubits):
        if self.select_measurement == "first_qubit_Z":
            self.matrix_operator = first_qubit_Z(number_of_qubits)


def first_qubit_Z(number_of_qubits):
    observable = np.zeros((2 ** number_of_qubits, 2 ** number_of_qubits), dtype=complex)
    z_pauli_vector = np.array([1, -1], dtype=complex)
    identity_vector = np.array([1, 1], dtype=complex)
    if number_of_qubits == 1:
        for i in range(2 ** number_of_qubits):
            observable[i][i] += z_pauli_vector[i]

    elif number_of_qubits == 2:
        for i in range(2 ** number_of_qubits):
            observable[i][i] += np.kron(z_pauli_vector, identity_vector)[i]
    else:
        temp = np.kron(z_pauli_vector, identity_vector)
        for _ in range(number_of_qubits - 2):
            temp = np.kron(temp, identity_vector)

        for i in range(2 ** number_of_qubits):
            observable[i][i] += temp[i]

    return observable


class QuantumComputer:
    """
    Attributes
    ----------

    """
    def __init__(self):
        """
        Parameters
        ----------

        """
        return None

    def config(self, select_qc_model="AB_repeat", number_of_qubits=2, depth=2, ):
        """
        Parameters
        ----------

        """
        self.select_qc_model = select_qc_model
        self.number_of_qubits = number_of_qubits
        self.depth = depth

    def inputs(self, Encode, A, B, Observable,):
        """
        Parameters
        ----------

        """
        A.config(self.number_of_qubits, total_time=10, total_depth=self.depth)
        B.config(self.number_of_qubits, total_depth=self.depth)
        Observable.create_observable(self.number_of_qubits)
        self.Encode = Encode
        self.A = A  # Assumes this to be a fixed gate
        self.B = B  # Assume this to be parameterised gate
        self.Observable = Observable

        return None

    def run_qc(self, input_data):
        self.Encode.encode_data(self.number_of_qubits, input_data)
        initial_state = PureQuantumState.all_zero_qubit(self.number_of_qubits)

        # Encoding data
        encoded_state_vector = np.matmul(self.Encode.matrix_operator, initial_state.state_vector)

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
        self.B.update_params(new_params)



