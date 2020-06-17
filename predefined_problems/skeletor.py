import networkx as nx
from qiskit import *
from qiskit.tools.monitor import job_monitor

import os
os.path.abspath(os.curdir)
os.path.sys.path.append('../hamiltonian_engine/')
from expectation_value import expectation_value as ex_v
from hamiltonian import mixer_hamiltonian as mix_ham
from hamiltonian import phase_hamiltonian as phs_ham



class skeletor:
    objective_function = None 
    variables = None
    graph = None
    shots = 0

    def __init__(self, p: int, obj_fun: str, variables: list, boolean: bool, graph: nx.Graph = None):
        # self.shots = shots
        self.graph = graph
        self.p = p
        self.objective_function = obj_fun
        self.variables = variables

        self.phse_ham = phs_ham(self.objective_function, self.variables)

        # generate Phase Hamiltonian
        self.phse_ham.Hamify(boolean=boolean)

        if graph != None:
            self.expectation = ex_v(
                self.objective_function, self.variables, is_graph=True)
        else:
            self.expectation = ex_v(
                self.objective_function, self.variables, is_graph=False)

        self.mx_ham = mix_ham()

    def get_objFun(self):
        return self.phse_ham.get_objFun()

    def get_pHamil(self):
        return self.phse_ham.get_pHamil()

    def set_upMixerHamiltonian(self, func, inverse=None):
        self.mx_function = (func, inverse)
    
    def setup_device(self, run_function, function_args:dict):
        self.qpu_execution = run_function
        self.qpu_args = function_args

    def generate_quantumCircuit(self, hyperparams: list):
        assert len(hyperparams) == 2*self.p

        l = len(hyperparams)
        gammas = hyperparams[:l//2]
        betas = hyperparams[l//2:]

        if self.graph != None:
            self.phse_ham.perEdgeMap(gammas, self.p, self.graph, True, True)
        else:
            self.phse_ham.perQubitMap(gammas, self.p, True, True)

        phse_map = self.phse_ham.qubit_map

        self.expectation.use_qubitMap(phse_map)

        if self.mx_function[0] == "general":
            self.mx_ham.generalXMixer(betas, self.p, phse_map, True)
        elif self.mx_function[0] == "controlled":
            self.mx_ham.controlledXMixer(betas, self.p, self.graph, inverse=self.mx_function[1], measure=True)

        self.circuit = self.phse_ham / self.mx_ham

        return self.circuit.draw(output='mpl')

    def run_circuit(self):

        job = self.qpu_execution(self.circuit, **self.qpu_args)

        job_monitor(job, quiet=True)

        results = job.result()

        print('Run Complete! job_id : {}'.format(job.job_id()))

        print("Expectation Value : {}".format(self.expectation.get_expectationValue(results, self.qpu_args['shots'], self.graph)))

        return results

    def run_skeletor(self, hyperparameters: list):
        
        self.generate_quantumCircuit(hyperparameters)

        job = self.qpu_execution(self.circuit, **self.qpu_args)

        job_monitor(job, quiet=True)

        results = job.result()

        res_maxcut = self.expectation.get_expectationValue(results, self.qpu_args['shots'], self.graph)

        return -1 * res_maxcut

        
    def run_QAOA(self, opt_function, **kwargs):

        res = opt_function(self.run_skeletor, **kwargs)

        opt_hyperparameter = res.x

        self.generate_quantumCircuit(opt_hyperparameter)

        results = self.run_circuit()

        res_maxcut = self.expectation.get_expectationValue(results,self.qpu_args['shots'],self.graph)

        return {'expectation': res_maxcut, 'optimal_parameters': opt_hyperparameter, 'QPU_data':results , 'optimizer_data': res }

