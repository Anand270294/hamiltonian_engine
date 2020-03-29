from sympy import *
from sympy.abc import I
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
import networkx as nx
import numpy as np


# TODO: Reconstruct phase hamiltonians for k-dits & constants are regarded as variables->free_symbols
class hamiltonian:

    #I = sympy.abc.I
    X = {}
    Y = {}
    Z = {}
    constants = []

    qudit_map = {}
    symbolic_map = {}

    obj_exp = ""
    Hamil_exp = ""
    quanCir_list = []

    def __init__(self, expr_str, var):
        self.__checkVariables(expr_str, var)

        # Initialize expressions and variables
        self.expression_str = expr_str
        self.variables = var

        self.obj_exp = sympify(self.expression_str)

        # Number of pauli objects are limited to the number of variables/qubits being used
        i = 0
        for sym in self.obj_exp.free_symbols:
            if self.variables.count(str(sym)) > 0:
                self.Z[sym] = symbols('Z_{}'.format(i))
                self.X[sym] = symbols('X_{}'.format(i))
                self.Y[sym] = symbols('Y_{}'.format(i))
                self.symbolic_map[Not(sym)] = 0.5 * (I + self.Z[sym])
                self.symbolic_map[sym] = 0.5*(I - self.Z[sym])
                i += 1
            else:
                self.symbolic_map[sym] = sym * I
                self.constants.append(sym)

    # Need to check more cases so as to prevent future errors
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
        return self.Hamil_exp

    def get_qclist(self):
        return self.quanCir_list

    def get_exprstr(self):
        return self.expression_str

    def get_variables(self):
        return self.variables

    def get_qubitMap(self):
        return self.qubit_map


class phase_hamiltonian(hamiltonian):
    def __init__(self, expr_str, var):
        super().__init__(expr_str, var)

    def Hamify(self, pwr_args=True, boolean=False, global_phse=True):
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

        # # Remove all Identity matrices and global phases
        self.Hamil_exp = self.Hamil_exp.subs(I, 1)
        coeff = self.Hamil_exp.as_coefficients_dict()

        # # Reduce variables with >= power(1) to power(1)
        if pwr_args == True:
            self.Hamil_exp = self.Hamil_exp.replace(
                lambda expr: expr.is_Pow, lambda expr: expr.base**1)

        # Remove the global phase of the expression as it will not affect the outcome
        if global_phse == True:
            gbl_phse = coeff.get(1)
            if gbl_phse != None:
                self.Hamil_exp = self.Hamil_exp - gbl_phse

            self.Hamil_exp = self.Hamil_exp.subs(
                [(c, 0) for c in self.constants])

        # Convert to expression into a sympy poly to get list of monomial expression to build the QC
        # However for simplicity we will still reduce expressions with power > 1 to 1
        if pwr_args == False:
            temp = self.Hamil_exp.replace(
                lambda expr: expr.is_Pow, lambda expr: expr.base**1)
            self.quanCir_list = Poly(temp).monoms()
        else:
            self.quanCir_list = Poly(self.Hamil_exp).monoms()

    # multidimensional qubit mapping for problems involving k-dits
    def perDitMap(self, gamma, p, k_dits, graph: nx.Graph, sub_expr={'i': 'i'}, barrier=False, initial_Hadamard=False):
        # Pre-processing before mapping the qudits
        no_of_qubits = len(graph.nodes) * k_dits

        self.p_hamilCir = QuantumCircuit(no_of_qubits)

        self._generate_quditMap(k_dits, graph, no_of_qubits)

        if initial_Hadamard == True:
            for i in range(no_of_qubits):
                self.p_hamilCir.h(i)
            self.p_hamilCir.barrier()

        for e in graph.edges:
            for i in range(k_dits):
                nxt_i = eval(sub_expr['i']) % k_dits

                self.p_hamilCir.cx(
                    self.qudit_map[e[0]][i], self.qudit_map[e[1]][nxt_i])
                self.p_hamilCir.rz(gamma, self.qudit_map[e[1]][nxt_i])
                self.p_hamilCir.cx(
                    self.qudit_map[e[0]][i], self.qudit_map[e[1]][nxt_i])

        if barrier == True:
            self.p_hamilCir.barrier()

        return self.p_hamilCir

    def _generate_quditMap(self, k_dits, graph: nx.Graph, no_of_qubits):
        i = 0
        for v in graph.nodes:
            qbits = tuple()
            for j in range(k_dits):
                qbits = qbits + (i,)
                i += 1
            self.qudit_map[v] = qbits
        assert i == no_of_qubits

    # Map the qubits directly to each variable
    def perQubitMap(self, gamma, p, barrier=False, initial_Hadamard=False):
        self.p_hamilCir = QuantumCircuit(
            len(self.variables), len(self.variables))

        if initial_Hadamard == True:
            for i in range(len(self.variables)):
                self.p_hamilCir.h(i)
            self.p_hamilCir.barrier()

        for sub_cir in self.quanCir_list:
            if sum(sub_cir) > 1:
                indices = [i for i, x in enumerate(sub_cir) if x == 1]
                for i in range(len(indices)):
                    if i == len(indices) - 1:
                        self.p_hamilCir.rz(gamma, indices[i])
                    else:
                        self.p_hamilCir.cx(indices[i], indices[i + 1])

                for i in range(len(indices)-1, 0, -1):
                    self.p_hamilCir.cx(indices[i-1], indices[i])

            else:
                self.p_hamilCir.rz(gamma, sub_cir.index(1))

        if barrier == True:
            self.p_hamilCir.barrier()

        return self.p_hamilCir

    # Only for 2 variable Expressions since each edge is an interaction between 2 vertices(qubits)
    def perEdgeMap(self, gamma:list, p:int, G:nx.Graph,  barrier=False, initial_Hadamard=False):
        self.p_hamilCir = QuantumCircuit(len(G.nodes), len(G.nodes))

        if initial_Hadamard == True:
            for i in range(len(G.nodes)):
                self.p_hamilCir.h(i)
        self.p_hamilCir.barrier()

        for i in range(p):
            for e in G.edges:
                for sub_cir in self.quanCir_list:
                    if sum(sub_cir) > 1:
                        self.p_hamilCir.cx(e[0], e[1])
                        self.p_hamilCir.rz(gamma[i], e[1])
                        self.p_hamilCir.cx(e[0], e[1])
                    else:
                        self.p_hamilCir.rz(gamma[i], e[sub_cir.index(1)])

        if barrier == True:
            self.p_hamilCir.barrier()

        return self.p_hamilCir

    def get_objFun(self):
        return super().get_objFun()

    def get_pHamil(self):
        return super().get_pHamil()

    def get_qclist(self):
        return super().get_qclist()

    def get_exprstring(self):
        return super().get_exprstr()

    def get_variablesList(self):
        return super().get_variables()

    def get_qubitMap(self):
        return super().get_qubitMap()


class mixer_hamiltonian(hamiltonian):
    def __init__(self, phase_ham:phase_hamiltonian):
        super().__init__(phase_ham.get_exprstring(), phase_ham.get_variablesList())

    # Mixer Hamiltonian for qubits to have dynamicity between {0,1}
    def generalXMixer(self, betas:list, p:int, measure=False, graph_map=False, graph: nx.Graph = None):
        if graph_map == False:
            self.mixer_circuit = QuantumCircuit(
                len(self.variables), len(self.variables))
            for i in range(len(self.variables)):
                self.mixer_circuit.rx(betas[0], i)
        else:
            if graph != None:
                self.mixer_circuit = QuantumCircuit(
                    len(graph.nodes), len(graph.nodes))
                
                for i in range(p):
                    for v in range(len(graph.nodes)):
                        self.mixer_circuit.rx(betas[i], v)
            else:
                raise ValueError(
                    'Missing Argument: {} for "graph:nx.Graph"'.format(graph))

        if measure == True:
            self.mixer_circuit.barrier()
            if graph_map == False:
                self.mixer_circuit.measure(
                    range(len(self.variables)), range(len(self.variables)))
            else:
                self.mixer_circuit.measure(
                    range(len(graph.nodes)), range(len(graph.nodes)))
        return self.mixer_circuit

    def controlledXMixer(self, beta, q, graph: nx.Graph, measure=False):
        # allow for ancillary qubits so that controlled rotations can be performed.
        # Include the controlled X-not version by adding X gates to each side of the controlled qubit line.
        self.mixer_circuit = QuantumCircuit(
            len(graph.nodes) + 1, len(graph.nodes))

        # Get all the q-regs
        quantum_regs = self.mixer_circuit.qregs

        # Declare last qreg to be the ancillary qubit
        ancilla_qubit = quantum_regs[len(quantum_regs) - 1]

        for n in graph.nodes:
            bfs = dict(nx.traversal.bfs_successors(graph, n, depth_limit=1))
            for source in bfs:
                if len(bfs[source]) != 0:
                    control_bits = list(quantum_regs[n] for n in bfs[source])
                    self.mixer_circuit.mct(
                        control_bits, ancilla_qubit, None, mode="noancilla")
                    self.mixer_circuit.mcrx(
                        beta, [ancilla_qubit], quantum_regs[int(source)])
                    self.mixer_circuit.mct(
                        control_bits, ancilla_qubit, None, mode='noancilla')
                    self.mixer_circuit.barrier()
                else:
                    self.mixer_circuit.rx(beta, int(source))

            if measure == True:
                self.mixer_circuit.barrier()
                self.mixer_circuit.measure(
                    range(len(graph.nodes)), range(len(graph.nodes)))

            return self.mixer_circuit

    def single_XYMixer(self, xy: str, beta: float, qubit_1: int, qubit_2: int, ancillary_qubit: int = None):
        if qubit_1 == qubit_2:
            raise ValueError(
                "Error: qubit_1 and qubit_2 cannot have the same int values.")
        else:
            if xy == "xx":
                self.mixer_circuit.h(qubit_1)
                self.mixer_circuit.h(qubit_2)
                self.mixer_circuit.cx(qubit_1, qubit_2)

                if ancillary_qubit == None:
                    self.mixer_circuit.rz(beta, qubit_2)
                else:
                    self.mixer_circuit.crz(beta, ancillary_qubit, qubit_2)

                self.mixer_circuit.cx(qubit_1, qubit_2)
                self.mixer_circuit.h(qubit_1)
                self.mixer_circuit.h(qubit_2)

            elif xy == 'yy':
                x_rot = np.pi / 2
                self.mixer_circuit.rx(x_rot, qubit_1)
                self.mixer_circuit.rx(x_rot, qubit_2)
                self.mixer_circuit.cx(qubit_1, qubit_2)

                if ancillary_qubit == None:
                    self.mixer_circuit.rz(beta, qubit_2)
                else:
                    self.mixer_circuit.crz(beta, ancillary_qubit, qubit_2)

                self.mixer_circuit.cx(qubit_1, qubit_2)
                self.mixer_circuit.rx(x_rot, qubit_1)
                self.mixer_circuit.rx(x_rot, qubit_2)

            else:
                raise ValueError(
                    "Incorrect xy string; it can only be either 'xx' or 'yy'.")

        return self.mixer_circuit
