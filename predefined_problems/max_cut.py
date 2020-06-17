import os 
os.path.sys.path.append('../hamiltonian_engine/')
from hamiltonian import phase_hamiltonian as phs_ham
from hamiltonian import mixer_hamiltonian as mix_ham
from expectation_value import expectation_value as ex_v

from qiskit import *
from qiskit.tools.monitor import job_monitor
import networkx as nx

# TODO: Change class to MAX-CUT then add different sub-class types

class max_cut:
    objective_function = '~x_1 & x_2 | ~x_2 & x_1' # Try to state the actual functions 
    objective_functionDi = '~x_1 & x_2'
    variables = ['x_1','x_2']
    graph = None
    shots = 0
    

    def __init__(self, p:int, graph:nx.Graph, directed=False):
        # self.shots = shots
        self.graph = graph
        self.directed = directed
        self.p = p

        if self.directed == False:
            self.phse_ham = phs_ham(self.objective_function,self.variables)
            self.expectation = ex_v(self.objective_function, self.variables, is_graph=True)
        else:
            self.phse_ham = phs_ham(self.objective_functionDi, self.variables)
            self.expectation = ex_v(self.objective_functionDi, self.variables, is_graph=True)
            assert isinstance(self.graph, nx.DiGraph)

        self.mx_ham = mix_ham()

        # generate Phase Hamiltonian
        self.phse_ham.Hamify(boolean=True)

    def setup_device(self, run_function, function_args:dict):
        self.qpu_execution = run_function
        self.qpu_args = function_args


    def generate_quantumCircuit(self, graph:nx.Graph, hyperparams:list):
        if graph == None:
            raise ValueError('Missing Argument: {} for "graph:nx.Graph"'.format(graph))
        else:
            assert len(hyperparams) == 2*self.p
            
            l = len(hyperparams)
            gammas = hyperparams[:l//2]
            betas  = hyperparams[l//2:]

            self.phse_ham.perEdgeMap(gammas, self.p, graph, True, True)

            phse_map = self.phse_ham.qubit_map

            self.mx_ham.generalXMixer(betas, self.p, phse_map,True)

            self.circuit = self.phse_ham / self.mx_ham
    
    def run_circuit(self):

        job = self.qpu_execution(self.circuit, **self.qpu_args)

        job_monitor(job, quiet=True)

        results = job.result()

        print('Run Complete! job_id : {}'.format(job.job_id()))

        print("Expectation Value : {}".format(self.expectation.get_expectationValue(results,self.qpu_args['shots'],self.graph)))

        return results

    def MAX_CUT(self, hyperparameters:list):
        
        self.generate_quantumCircuit(self.graph, hyperparameters)

        job = self.qpu_execution(self.circuit, **self.qpu_args)

        job_monitor(job, quiet=True)

        results = job.result()

        res_maxcut = self.expectation.get_expectationValue(results, self.qpu_args['shots'], self.graph)

        return -1 * res_maxcut


    def run_QAOA(self, opt_function, **kwargs):
        
        res = opt_function(self.MAX_CUT, **kwargs)

        opt_hyperparameter = res.x

        self.generate_quantumCircuit(self.graph,  opt_hyperparameter)

        results = self.run_circuit()

        res_maxcut = self.expectation.get_expectationValue(results,self.qpu_args['shots'],self.graph)

        return {'expectation': res_maxcut, 'optimal_parameters': opt_hyperparameter, 'QPU_data':results , 'optimizer_data': res }

