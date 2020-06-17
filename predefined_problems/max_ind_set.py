import os 
os.path.sys.path.append('../hamiltonian_engine/')
from hamiltonian import phase_hamiltonian as phs_ham
from hamiltonian import mixer_hamiltonian as mix_ham
from expectation_value import expectation_value as ex_v

from qiskit import *
from qiskit.tools.monitor import job_monitor
import networkx as nx


class max_ind_set:
    objective_function = 'x_1' # Try to state the actual functions 
    variables = ['x_1']
    graph = None
    shots = 0
    qubit_map = None

    def __init__(self, p:int, graph:nx.Graph):
        # self.shots = shots
        self.graph = graph
        self.p = p


        self.phse_ham = phs_ham(self.objective_function,self.variables)
        self.expectation = ex_v(self.objective_function, self.variables, is_graph=True)

        self.mx_ham = mix_ham()

        # generate Phase Hamiltonian
        self.phse_ham.Hamify(boolean=True)

    def setup_device(self, run_function, function_args:dict):
        self.qpu_execution = run_function
        self.qpu_args = function_args

    def generate_quantumCircuit(self, hyperparams:list):
        if self.graph == None:
            raise ValueError('Missing Argument: {} for "graph:nx.Graph"')
        else:
            assert len(hyperparams) == 2*self.p
            
            l = len(hyperparams)
            gammas = hyperparams[:l//2]
            betas  = hyperparams[l//2:]

            self.phse_ham.perEdgeMap(gammas, self.p, self.graph, True, True)

            self.qubit_map = self.phse_ham.get_qubitMap()
            self.expectation.use_qubitMap(self.qubit_map)

            self.mx_ham.controlledXMixer(betas, self.p, self.graph, True, True)

            self.circuit = self.phse_ham / self.mx_ham

            return self.circuit.draw(output='mpl')
    
    def run_circuit(self, shots=1024):
        
        job = self.qpu_execution(self.circuit, **self.qpu_args)

        job_monitor(job, quiet=True)

        results = job.result()

        print('Run Complete! job_id : {}'.format(job.job_id()))

        print("Expectation Value : {}".format(self.expectation.get_expectationValue(results, self.qpu_args['shots'], self.graph)))

        return results

    def MAX_IND(self, hyperparameters:list):

        self.generate_quantumCircuit(hyperparameters)

        job = self.qpu_execution(self.circuit, **self.qpu_args)

        job_monitor(job, quiet=True)

        results = job.result()

        res_maxcut = self.expectation.get_expectationValue(results, self.qpu_args['shots'], self.graph)

        return -1 * res_maxcut


    def run_QAOA(self, opt_function, **kwargs):
        
        res = opt_function(self.MAX_IND, **kwargs)

        opt_hyperparameter = res.x

        self.generate_quantumCircuit(opt_hyperparameter)

        results = self.run_circuit()

        res_maxcut = self.expectation.get_expectationValue(results, self.qpu_args['shots'], self.graph)

        return {'expectation': res_maxcut, 'optimal_parameters': opt_hyperparameter, 'QPU_data':results , 'optimizer_data': res }









    