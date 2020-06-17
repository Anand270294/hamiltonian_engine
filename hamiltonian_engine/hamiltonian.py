from sympy import *
from sympy.abc import I
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
import networkx as nx
import numpy as np

import os
os.path.sys.path.append('../hamiltonian_engine/')
from utils import circuit_builder as cir_build

class hamiltonian:
    """
    A Parent class that aids with generating Quantum Alternating Operator Ansatz Circuit 

    This abstract class is meant to hold symbols that is meant to generate the cost/phase hamiltonians, allow for overloaded operations
    between various hamiltonians and other data types. This class is not meant to be instantiated directly buy via the other 2
    hamiltonian classes, certain functions is this class may be specific to only a certain child hamiltonian class.

    ...

    Attributes
    ----------
    expression_str : str
        expression of classical objective function that is not symbolic
    var : list 
        list of variables in the classical objective function

    X : dict 
        X gates based on the number of variables
    Y : dict
        Y gates based on the number of variables
    Z : dict 
        Z gates based on the number of variables

    constants : list 
        list of constants; symbols that are not in the variable list
    
    symbolic_map : dict
        dict that is used to map variables in to a Pauli Expression
    
    qubit_map : dict
        dict that holds how qubits are mapped to vertices/variables

    quanCir_list : list
        Monomial list of all Z interactions between qubits
    
    quantum_circuit : list
        list of generated quantum circuit based on the hamiltonian and length = P-steps
    
    obj_exp : str
        symbolic expression of the classical objective function
    
    Hamil_exp : str
        symbolic expression of phase hamiltonian
    
    full_hamiltonian : str
        symbolic expression of phase hamiltonian that has all zz/z expression


    Methods
    -------

    get_objFun()
        returns the classical objective function 
    
    get_pHamil()
        returns the symbolic phase hamiltonian

    get_qclist()
        returns the quanCir_list
    
    get_exprstr()
        returns the expression_str
    
    get_variables()
        returns the list of variables

    get_qubitMap()
        returns qubit_map

    get_quantumCircuitAslist()
        returns quantum_circuit
    
    get_quantumCircuit()
        returns a combined circuit of all the circuits in the quantum_circuit list
    """

    # Symbolic Attributes
    X = {}
    Y = {}
    Z = {}
    constants = []
    symbolic_map = {}

    # Meta data for circuit generation
    quanCir_list = []
    qubit_map = None

    quantum_circuit = []
    an = QuantumRegister(1, 'ancilla')

    # Equations of phase hamiltonians
    obj_exp = ""
    Hamil_exp = ""
    full_hamiltonian = None


    def __init__(self, expr_str, var):
        """
        Parameters
        ----------
        expr_str : str
            Classical Objective function as string
        var : list
            list of variables present in expr_str
        """

        if expr_str != None:

            self.__checkVariables(expr_str, var)

            # Initialize expressions and variables
            self.expression_str = expr_str
            self.variables = var

            self.obj_exp = sympify(self.expression_str)

            # Number of pauli objects are limited to the number of variables/qubits being used
            i = 0
            for sym in self.obj_exp.free_symbols:
                if self.variables.index(str(sym)) >= 0:
                    i = self.variables.index(str(sym))
                    ind = self.variables[i].find('_')
                    subscrpt = self.variables[i][ind + 1]      
                    self.Z[sym] = symbols('Z_{}'.format(subscrpt))
                    self.X[sym] = symbols('X_{}'.format(subscrpt))
                    self.Y[sym] = symbols('Y_{}'.format(subscrpt))
                    self.symbolic_map[Not(sym)] = 0.5 * (I + self.Z[sym])
                    self.symbolic_map[sym] = 0.5*(I - self.Z[sym])
                    #i += 1
                else:
                    self.symbolic_map[sym] = sym * I
                    self.constants.append(sym)
        else:
            self.variables = var

    def __add__(self, other):
        if isinstance(other, phase_hamiltonian) & isinstance(self, phase_hamiltonian):
            temp_func = str(self.obj_exp + other.obj_exp - (self.obj_exp * other.obj_exp))
            temp_var  = self.variables + list(set(other.variables) - set(self.variables))  

            return phase_hamiltonian(temp_func, temp_var) 

        elif (isinstance(other, float) | isinstance(other, int)) & isinstance(self, phase_hamiltonian):
            temp_func = str(float(other) + self.obj_exp)

            return phase_hamiltonian(temp_func, self.variables)
        

    def __mul__(self, other):
        if isinstance(other, phase_hamiltonian) & isinstance(self, phase_hamiltonian):
            temp_func = str(self.obj_exp * other.obj_exp)
            temp_var  = self.variables + list(set(other.variables) - set(self.variables))

            return phase_hamiltonian(temp_func, temp_var)

        elif (isinstance(other, float) | isinstance(other, int)) & isinstance(self, phase_hamiltonian):
            temp_func = str(float(other) * self.obj_exp)

            return phase_hamiltonian(temp_func, self.variables)
    
    def __invert__(self):
        return phase_hamiltonian(str(1 - (self.obj_exp)), self.variables)

    def __truediv__(self, other):
        if isinstance(self, hamiltonian) & isinstance(other, hamiltonian):
            assert len(self.quantum_circuit) == len(other.quantum_circuit)
            cirq = None
            for i in range(len(self.quantum_circuit)):
                if i == 0 :
                    cirq = self.quantum_circuit[i] 
                    temp = other.quantum_circuit[i]
                    if temp.has_register(self.an):
                        cirq.add_register(self.an)
                    cirq = cirq + temp
                else:
                    temp1 = self.quantum_circuit[i] 
                    temp2 = other.quantum_circuit[i]
                    if temp.has_register(self.an):
                        temp1.add_register(self.an)
                    cirq = cirq + temp1 + temp2
        return cirq

    def __checkVariables(self, expression: str, variables: list):
        for v in variables:
            if(v in expression):
                pass
            else:
                raise ValueError(
                    'Variables Mismatch! Unable to find {} in the Objective Function: {}'.format(v, expression))

    def get_objFun(self):
        return self.obj_exp

    def get_pHamil(self):
        if self.full_hamiltonian == None:
            return self.Hamil_exp
        else:
            return self.full_hamiltonian

    def get_qclist(self):
        return self.quanCir_list

    def get_exprstr(self):
        return self.expression_str

    def get_variables(self):
        return self.variables

    def get_qubitMap(self):
        return self.qubit_map

    def get_quantumCircuitAslist(self):
        return self.quantum_circuit
    
    def get_quantumCircuit(self):
        for i in range(len(self.quantum_circuit)):
            if i  == 0:
                c = self.quantum_circuit[i]
            else:
                temp = self.quantum_circuit[i]
                c  = c + temp

        return c


class phase_hamiltonian(hamiltonian):
    """
    Child class of hamiltonian, phase hamiltonian generates the PauliSum symbolically and generates the quantum circuit.Users
    use this class to instantiate the class.

    Methods
    -------
    Hamify(pwr_args=True, boolean=False)
        generates the Pauli Sum expression using SymPy symbols
    
    perDitMap(gamma, p, k_dits, graph: nx.Graph, sub_expr={'i': 'i'}, barrier=False, initial_Hadamard=False)
        maps the each variable/vertex to dits(multiple qubits) used for problems that require order and sequence
    
    perQubitMap(gamma, p, barrier=False, initial_Hadamard=False)
        maps each variable to a single qubit; number of variables = number of qubits

    perEdgeMap(gamma:list, p:int, graph:nx.Graph,  barrier=False, initial_Hadamard=False)
        maps qubits based on the graph provided by the user
    """

    def __init__(self, expr_str:str, var):
        super().__init__(expr_str, var)

    def Hamify(self, pwr_args=True, boolean=False):
        """ Converts the objective expression that was made symbolic by the parent class into a Z/ZZ Pauli expression

        This methods follows a few rules that would correctly map the variables to 1/2 (I - Zi) or 1/2 (I + Zi). If User want 
        generate the circuit from this class, then they must leave pwr_args=True (default) value. This function also generates 
        quantCir_list to be used for circuit generation.

        Parameters
        ----------
        pwr_args : bool, optional
            Set to False if users want to keep the Pauli Expression that has powers > 1; however in a quantum circuit it does not matter
        boolean : bool, optional
            If the objective function is a boolean expression set this to True else an error will be raised by SymPy
        """

        if boolean == True:
            self.Hamil_exp = simplify_logic(self.obj_exp)
            self.Hamil_exp = self.Hamil_exp.replace(Or, Add)
            self.Hamil_exp = self.Hamil_exp.replace(And, Mul)

        else:
            self.Hamil_exp = self.obj_exp

        # maps x -> 1/2(I - Zi) or ~x -> 1/2(I + Zi)
        self.Hamil_exp = self.Hamil_exp.xreplace(self.symbolic_map)

        self.Hamil_exp = expand(self.Hamil_exp)

        self.Hamil_exp = self.Hamil_exp.replace(
            lambda expr: expr.is_Pow and (expr.count(I) > 0), lambda expr: expr.base**1)

        #  Remove all Identity matrices that multipled with Pauli Z operators
        for sym in self.obj_exp.free_symbols:
            self.Hamil_exp = self.Hamil_exp.subs(self.Z[sym]* I, self.Z[sym])
        # coeff = self.Hamil_exp.as_coefficients_dict()

        #  Reduce variables with >= power(1) to power(1)
        if pwr_args == True:
            self.Hamil_exp = self.Hamil_exp.replace(
                lambda expr: expr.is_Pow, lambda expr: expr.base**1)

        self.Hamil_exp = self.Hamil_exp.subs([(c, 0) for c in self.constants])

        # Convert to expression into a sympy poly to get list of monomial expression to build the QC
        # However for simplicity we will still reduce expressions with power > 1 to 1
        if pwr_args == False:
            temp = self.Hamil_exp.replace(
                lambda expr: expr.is_Pow, lambda expr: expr.base**1)
            self.quanCir_list = Poly(temp).monoms()
        else:
            temp = self.Hamil_exp.subs(I , 1)
            
            coeff = temp.as_coefficients_dict()
            gbl_phse = coeff.get(1)

            if gbl_phse != None:
                temp = temp - gbl_phse
            
            self.quanCir_list = Poly(temp).monoms()

    def perDitMap(self, gamma, p, k_dits, graph: nx.Graph, sub_expr={'i': 'i'}, barrier=False, initial_Hadamard=False):
        """ Experimental Function: maps the each variable/vertex to dits(multiple qubits) used for problems that require order and sequence.

            Dit map functions allows users to map a single variable to multiple qubits, this function is used for problems like TSP. However,
            the circuit generated in not optimized. 

            Parameters
            ----------
            gamma : list
                list of initial hyperparameters to generate the quantum circuit
            p : int
                number of p-steps 
            graph : nx.Graph
                a networkx graph that can be used to map the variables to k-dits
            k_dits : int
                number of qubit that each variable is mapped to
            sub_expr : dict
                expression that determines how the k-dits interact
            barrier : boolean
                set to True if a quantum barrier is to be added at the end of the circuit
            initial_hadamard : boolean
                If the circuit is to be set-up into equal superposition for all states.
        """
        
        assert p == len(gamma)

        self.quantum_circuit = []

        self.qubit_map = cir_build.circuit_builder.map_qubits(self.variables, k_dits, graph)

        no_qubits = len(graph.nodes) * k_dits

        for i in range(p):

            cir = QuantumCircuit(no_qubits)

            if i == 0 and initial_Hadamard == True:
                for j in range(no_qubits):
                    cir.h(j)
                cir.barrier() 

            for e in graph.edges:
                cir += cir_build.circuit_builder.generate_Ditcircuit(self.quanCir_list, gamma[i], self.qubit_map, e, sub_expr, k_dits, no_qubits)
            
            if barrier == True:
                cir.barrier()

            self.quantum_circuit.append(cir)

    def perQubitMap(self, gamma:list, p, barrier=False, initial_Hadamard=False):
        """ Maps each variable to a single qubit 

            perQubitMap function can be used for QUBO functions or algorithms that do not involve a graph.

            Parameters
            ----------
            gamma : list
                list of initial hyperparameters to initialize the circuit
            p : int
                number of p-steps
            barrier : boolean
                set to True if there is to be a quantum barrier added at the end of the circuit
            initial_hadamard : boolean
                If the circuit is to be set-up into equal superposition
        """

        assert p == len(gamma)

        self.quantum_circuit = []

        self.qubit_map = cir_build.circuit_builder.map_qubits(self.variables, 0)

        no_qubits = len(self.qubit_map.values())

        for i in range(p):
            cir = QuantumCircuit(no_qubits)

            if i == 0 and initial_Hadamard == True:
                for j in range(no_qubits):
                    cir.h(j)

            cir += cir_build.circuit_builder.generate_Zcircuit(self.quanCir_list, gamma[i], self.qubit_map)       

            if barrier == True:
                cir.barrier()

            self.quantum_circuit.append(cir)

    # Only for 2 variable Expressions since each edge is an interaction between 2 vertices(qubits)
    def perEdgeMap(self, gamma:list, p:int, graph:nx.Graph,  barrier=False, initial_Hadamard=False):
        """ Maps the qubits based on the graph and the hamiltonian generated.

            User have to be cautious that the hamiltonian generated only has 2 variables since this functions maps on an edge-edge basis, 
            which also allows for single Z gate per vertex for problems that are based on the vertices i.e max independent set. This function also 
            generates the full hamiltonian of the graph for all egde interaction ZZ Pauli expression which can useful for users who wish to use
            Tensorflow Quantum to find the optimal Hypeparameters.

            Parameters
            ----------
            gamma : list 
                list of gamma hyperparameters for the Pauli ZZ expression
            p : int
                number of p-steps
            graph : networkx graph 
                Graph is used to map the qubits
            barrier : boolean
                set to True if there is to be a quantum barrier added at the end of the circuit
            initial_hadamard : boolean
                If the circuit is to be set-up into equal superposition
        """

        assert p == len(gamma) 

        self.__add_defaultWeights(graph)

        self.full_hamiltonian = 0
        self.quantum_circuit = []

        self.qubit_map = cir_build.circuit_builder.map_qubits(self.variables, 0, graph)

        no_qubits = len(self.qubit_map.values())

        for i in range(p):
            cir = QuantumCircuit(no_qubits)

            if i == 0 and initial_Hadamard == True:
                for j in range(no_qubits):
                    cir.h(j)

            if len(self.variables) == 2:
                for e in graph.edges:
                    if i == 0:
                        temp = self.Hamil_exp
                        l = 0
                        for sym in self.Hamil_exp.free_symbols:
                            if not (sym == I):
                                temp = temp.subs(sym, symbols('Z_{}'.format(e[l])))
                                l = (l + 1) % 2
                            
                        self.full_hamiltonian += graph.get_edge_data(e[0],e[1])["weight"]* temp

                    cir += cir_build.circuit_builder.generate_Zcircuit(self.quanCir_list, gamma[i], self.qubit_map, e)
            else:
                if i == 0:
                    for v in graph.nodes:
                        temp = self.Hamil_exp
                        for sym in self.Hamil_exp.free_symbols:
                            if not (sym == I):
                                temp = temp.subs(sym, symbols('Z_{}'.format(v)))
                        self.full_hamiltonian += temp

                cir += cir_build.circuit_builder.generate_Zcircuit(self.quanCir_list, gamma[i], self.qubit_map, edge=(-1,-1))

            if barrier == True:
                cir.barrier()

            self.quantum_circuit.append(cir)

    def __add_defaultWeights(self, graph:nx.Graph, weights=1):
        for u,v in graph.edges:
            if not bool(graph.get_edge_data(u,v)):
                graph.add_edge(u, v, weight = weights)



class mixer_hamiltonian(hamiltonian):
    """
    Child class of hamiltonian, mixer hamiltonian generates the mixer/reference quantum circuits, however, it does not generate the Pauli expression.
    Users use this class to instantiate the class.

    Methods
    -------
    generalXMixer( betas:list, p:int, qubit_map:dict, measure=False)
        maps the each variable/vertex with RX gate
    
    controlledXMixer(beta:list, p:int, graph: nx.Graph, inverse:bool= False, measure=False)
        maps each vertex to controlled RX gates based on the neigbouring vertices.
    """

    def __init__(self, var=None, expr_str=None):
        super().__init__(expr_str, var)

    def generalXMixer(self, betas:list, p:int, qubit_map:dict, measure=False):
        """ Maps each of the variables/vertices to single RX gate to allow non-trivial dynamicity to the qubit states

            generalXmixer function is trivial to implement based on the number of qubits present. Users are also given 
            an option to measure all of the qubit values if they wish to run the circuit on a simulator/QPU.

            Parameters
            ----------
            betas : list
                list of betas to be used as rotation angles
            p : int
                number of p-steps
            qubit_map : dict
                dict of how each qubit is mapped, can be obtained from phase_hamiltonian object
            measure : boolean
                If the circuit is to be measured at the end
        """

        self.quantum_circuit = []

        for i in range(p):
            cir = QuantumCircuit()

            cir += cir_build.circuit_builder.generate_RXcircuit(qubit_map, betas[i])
                
            if measure == True and i == p - 1:
                cir.measure_all()

            self.quantum_circuit.append(cir)


    def controlledXMixer(self, betas:list, p:int, graph: nx.Graph, inverse:bool= False, measure=False):
        """ maps each vertex to controlled RX gates based on the neigbouring vertices.

            controlledXMixer function uses Breath First Search to find the neighbouring vertices as the control qubits for RX gate.
            User must be cautious with regards to the degree of each vertex, if the degree is too high > 3 the gate count of the circuit 
            will be very high which may lead to higher errors in the results especially if the circuit is run on an actual QPU. Users are
            also provided with an option to invert the controll qubits.

            Parameters
            ----------
            betas : list 
                list of betas to be used as rotation angles
            p : int
                number of p-steps
            graph : networkx graph
                networkx graph to be used to find neighbouring vertices
            inverse : boolean
                Option to invert the control qubits and invert back after the rotation
            measure : boolean
                If the circuit is to be measured at the end
        """

        self.quantum_circuit = []

        for i in range(p):
            cir = QuantumCircuit(len(graph.nodes))

            # Get all the q-regs
            quantum_regs = cir.qregs[0]

            for n in graph.nodes:
                bfs = dict(nx.traversal.bfs_successors(graph, n, depth_limit=1))
                for source in bfs:
                    if len(bfs[source]) > 0:
                        control_bits = list(quantum_regs[n] for n in bfs[source])
                        
                        if inverse == True:
                            cir.x(control_bits)

                        cir.mcrx(beta[i], control_bits, quantum_regs[int(source)])
                        
                        if inverse == True:
                            cir.x(control_bits)
                                         
            if measure == True and i == p - 1:
                classical_regs = ClassicalRegister(len(graph.nodes))
                cir.add_register(classical_regs)
                cir.measure(quantum_regs, classical_regs)

            self.quantum_circuit.append(cir)

    # Do not uncomment until you are ready for crazy!
    # def single_XYMixer(self, xy: str, beta: float, qubit_1: int, qubit_2: int, ancillary_qubit: int = None):
    #     if qubit_1 == qubit_2:
    #         raise ValueError(
    #             "Error: qubit_1 and qubit_2 cannot have the same int values.")
    #     else:
    #         if xy == "xx":
    #             self.mixer_circuit.h(qubit_1)
    #             self.mixer_circuit.h(qubit_2)
    #             self.mixer_circuit.cx(qubit_1, qubit_2)

    #             if ancillary_qubit == None:
    #                 self.mixer_circuit.rz(beta, qubit_2)
    #             else:
    #                 self.mixer_circuit.crz(beta, ancillary_qubit, qubit_2)

    #             self.mixer_circuit.cx(qubit_1, qubit_2)
    #             self.mixer_circuit.h(qubit_1)
    #             self.mixer_circuit.h(qubit_2)

    #         elif xy == 'yy':
    #             x_rot = np.pi / 2
    #             self.mixer_circuit.rx(x_rot, qubit_1)
    #             self.mixer_circuit.rx(x_rot, qubit_2)
    #             self.mixer_circuit.cx(qubit_1, qubit_2)

    #             if ancillary_qubit == None:
    #                 self.mixer_circuit.rz(beta, qubit_2)
    #             else:
    #                 self.mixer_circuit.crz(beta, ancillary_qubit, qubit_2)

    #             self.mixer_circuit.cx(qubit_1, qubit_2)
    #             self.mixer_circuit.rx(x_rot, qubit_1)
    #             self.mixer_circuit.rx(x_rot, qubit_2)

    #         else:
    #             raise ValueError(
    #                 "Incorrect xy string; it can only be either 'xx' or 'yy'.")

    #     return self.mixer_circuit
