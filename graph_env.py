import numpy as np
import gym
import time
from matplotlib import pyplot as plt
from PyClifford_old.src import *
from itertools import product

from utils import cprint
from static_env import StaticEnv

from qiskit.algorithms import NumPyMinimumEigensolver 
from qiskit_nature.algorithms import GroundStateEigensolver
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.drivers.second_quantization import ElectronicStructureMoleculeDriver, ElectronicStructureDriverType 
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper
from qiskit_nature.transformers.second_quantization.electronic import FreezeCoreTransformer
from qiskit_nature.drivers import Molecule
import qiskit_nature.settings
from pyscf import scf, gto, fci
from copy import deepcopy

qiskit_nature.settings.dict_aux_operators = False

POSSIBLEUNITARIES = ['H', 'S', 'HS', 'SH', 'SS', 'HSH', 'SHS', 'SSH', 'HSS', 'SSS', 'SSSH', 'HSSS', 'HSHS', 'SHSH', 'HSSSH', 'SHSSH', 'SSHSH', 'SHSHS', 'HSSHS', 'SHSSS', 'SSSHS', 'SSHSS', 'HSHSH']

N_UNITARIES = len(POSSIBLEUNITARIES)

def binary_sum(x):
    # Summing in pyclifford seems to take quadratic time. More efficient to do a recursive sum over halves of the list.
    if len(x) < 30:
        return sum(x)
    else:
        return binary_sum(x[::2]) + binary_sum(x[1::2])

def calculate_hamiltonian(geometry, use_pyscf=False):
    molecule = Molecule(geometry=geometry, charge=0, multiplicity=1)

    driver = ElectronicStructureMoleculeDriver(
        molecule, basis="sto3g", driver_type=ElectronicStructureDriverType.PYSCF
    )

    es_problem = ElectronicStructureProblem(driver, transformers=[FreezeCoreTransformer()])
    
    qubit_converter = QubitConverter(ParityMapper(), two_qubit_reduction=True)
    second_q_op = es_problem.second_q_ops()
    qubit_op = qubit_converter.convert(second_q_op[0], num_particles=es_problem.num_particles).to_pauli_op().oplist
    nuclear_repuls = es_problem.grouped_property_transformed.get_property("ElectronicEnergy").nuclear_repulsion_energy
    core_shift = es_problem.grouped_property_transformed.get_property("ElectronicEnergy")._shift['FreezeCoreTransformer']
    
    is_identity = lambda x: str(x.primitive).count('I') == len(str(x.primitive))
    const_shift = sum([x.coeff for x in qubit_op if is_identity(x)])
    pyc_hamiltonian = binary_sum([x.coeff * paulialg.pauli(str(x.primitive)) for x in qubit_op if not is_identity(x)])
    if pyc_hamiltonian.N >= 14 or use_pyscf:
        mol_s = '; '.join([x + ' ' + ' '.join(map(str, y)) for x, y in geometry])
        mol = gto.M(atom=mol_s, basis='sto3g')
        rhf = scf.RHF(mol)
        # rhf.kernel()
        ee_hf = rhf.kernel()
        full_config = fci.FCI(rhf)
        E0 = full_config.kernel()[0]
        return pyc_hamiltonian, (E0 - (nuclear_repuls + core_shift + const_shift)).real, (nuclear_repuls + const_shift + core_shift).real, ee_hf
    else:
        ee_hf = None
        numpy_solver = NumPyMinimumEigensolver()
        calc = GroundStateEigensolver(qubit_converter, numpy_solver)
        res = calc.solve(es_problem)
        return pyc_hamiltonian, (min(res.total_energies) - (nuclear_repuls + const_shift + core_shift)).real, (nuclear_repuls + const_shift + core_shift).real, ee_hf


def create_stabilizer_state(adj_matrix):
        
    n = len(adj_matrix)
    s_v = []
    
    for j in range(n):
        pauli = [''] * n
        for k in range(n):
            pauli[k] = 'Z' if adj_matrix[j][k] else 'I'
        pauli[j] = 'X'
        s_v.append("".join(pauli))
        
    return stabilizer.stabilizer_state(s_v)

def gate(qubit, type):
    H = stabilizer.stabilizer_state('X').to_map()
    I = stabilizer.stabilizer_state('I').to_map()
    if type == 'H':
        gate = circuit.CliffordGate(qubit)
        gate.set_forward_map(H)
        return gate
    if type == 'S':
        gate = circuit.CliffordGate(qubit)
        gate.set_generator(paulialg.pauli('Z'))
        return gate
    if type == 'I':
        gate = circuit.CliffordGate(qubit)
        gate.set_forward_map(I)
        return gate
    raise Exception('Invalid gate type.')

def construct_circ(gates):
    circ = circuit.CliffordCircuit()
    for gate in gates:
        circ.take(gate)
    return circ

def construct_all_circuits(qubits):
    circuits = [] # [[qubit0circ0, qubit0circ1, ...], [qubit1circ0, qubit1circ1, ...]
    for i in range(qubits):
        qubit_circuits = []
        for op in POSSIBLEUNITARIES:
            qubit_circuits.append(construct_circ([gate(i, k) for k in op]))
        circuits.append(qubit_circuits)
    return list(np.array(circuits).flatten())

def create_edges(qubits):
    edges = []
    for i in range(qubits):
        for j in range(i + 1, qubits):
            edge = np.zeros((qubits, qubits))
            edge[i][j] = 1
            edge[j][i] = 1
            edges.append(edge)
    return edges

MOL = "H2O"

BOND_DIS = 1.7

if MOL == "H2":
    PYC_HAMILTONIAN, EE_EXACT, EE_NUC, EE_HF = calculate_hamiltonian([["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, BOND_DIS]]], True)
elif MOL == "H2O":
    # c1 = np.array([-0.0399, -0.0038, 0.0])
    # c2 = np.array([1.5780, 0.8540, 0.0])
    # c3 = np.array([2.7909, -0.5159, 0.0])
    c1 = np.array([BOND_DIS * np.cos(np.pi * 104.5 / 180), BOND_DIS * np.sin(np.pi * 104.5 / 180), 0.0])
    c2 = np.array([0.0, 0.0, 0.0])
    c3 = np.array([BOND_DIS, 0.0, 0.0])
    bond1 = np.linalg.norm(c1 - c2)
    bond2 = np.linalg.norm(c3 - c2) 
    PYC_HAMILTONIAN, EE_EXACT, EE_NUC, EE_HF = calculate_hamiltonian([["H", c1], ["O", c2], ["H", c3]], True)

N_QUBITS = PYC_HAMILTONIAN.N

EDGES = create_edges(N_QUBITS)

N_EDGES = len(EDGES)

EDGE_LOCATIONS = [np.where(arr == 1)[0] for arr in EDGES]

MAX_STEPS = N_QUBITS + N_EDGES

ACTIONS = construct_all_circuits(N_QUBITS) + EDGES

N_ACTIONS = len(ACTIONS)

INIT_ENERGY = np.real(create_stabilizer_state(np.zeros((N_QUBITS, N_QUBITS))).expect(PYC_HAMILTONIAN))

class GraphEnv(gym.Env, StaticEnv):
    """
    gym environment with the goal to navigate the player from its
    starting position to the lowest energy node by choosing a graph state and local unitary. 
    Rewards are defined as the negative of the energy between states minus a penalty for each step. 
    The player starts with no connected nodes. 
    """ 
    n_actions = len(ACTIONS)

    def __init__(self):
        
        self.shape = (N_QUBITS, N_QUBITS)

        self.graph = np.zeros(self.shape)

        self.unitaries = np.zeros((N_QUBITS, N_UNITARIES))

        self.step_idx = 0

        self.stabilizer_state = create_stabilizer_state(self.graph)

        self.min_stabilizer_state = None

        self.energy = np.real(self.stabilizer_state.expect(PYC_HAMILTONIAN))

        self.min_energy = 1e8

    def reset(self):

        self.graph = np.zeros(self.shape)

        self.unitaries = np.zeros((N_QUBITS, N_UNITARIES))
        self.step_idx = 0
        state = np.append(self.unitaries, self.graph)
        self.stabilizer_state = create_stabilizer_state(self.graph)
        self.energy = INIT_ENERGY
        self.min_energy = 1e8
        return state.astype("float32"), 0, False, None

    def step(self, action):
        self.step_idx += 1

        if action < N_QUBITS * N_UNITARIES and (self.unitaries[action // N_UNITARIES] == 0).all():
            self.unitaries[action // N_UNITARIES][action % N_UNITARIES] = 1
            ACTIONS[action].forward(self.stabilizer_state)
        elif action >= N_QUBITS * N_UNITARIES:
            temp_graph = np.add(self.graph, ACTIONS[action])
            if (temp_graph > 1).any(): return np.append(self.unitaries, self.graph).astype("float32"), -self.min_energy, self.step_idx >= MAX_STEPS, None # reward = -0.5 for already connected edge, not done
            self.graph = temp_graph
            self.stabilizer_state = create_stabilizer_state(self.graph)
            for i in range(len(self.unitaries)):
                pos = self.unitaries[i].argmax()
                if self.unitaries[i][pos]: ACTIONS[N_UNITARIES * i + pos].forward(self.stabilizer_state)
        else: return np.append(self.unitaries, self.graph).astype("float32"), -self.min_energy, self.step_idx >= MAX_STEPS, None # reward = -0.5 for invalid action, not done
        
        e_after = np.real(self.stabilizer_state.expect(PYC_HAMILTONIAN))

        if e_after < self.min_energy: 
            self.min_energy = e_after
            self.min_stabilizer_state = self.stabilizer_state

        reward = -self.min_energy
        self.energy = e_after

        state = np.append(self.unitaries, self.graph)
        done = self.step_idx >= MAX_STEPS or (np.add(self.graph, np.diag(np.ones(N_QUBITS))).all() and self.unitaries.any(1).all())
        return state.astype("float32"), reward, done, None

    def render(self, mode='human'):
        print("Step, Energy:", self.step_idx, self.energy)
        print(np.append(self.unitaries, self.graph))

    @staticmethod
    def next_state(state, action, shape=(N_QUBITS, N_QUBITS)):
        unitaries = state.copy()
        graph = state.copy()
        unitaries = (unitaries[:N_UNITARIES * shape[0]]).reshape((shape[0], N_UNITARIES))
        graph = (graph[N_UNITARIES * shape[0]:]).reshape(shape)
        
        if action < shape[0] * N_UNITARIES and (unitaries[action // N_UNITARIES] == 0).all():
            unitaries[action // N_UNITARIES][action % N_UNITARIES] = 1
        elif action >= shape[0] * N_UNITARIES:
            graph = np.add(graph, ACTIONS[action])
            if (graph > 1).any(): return state.astype("float32")
        else: return state.astype("float32")

        return np.append(unitaries, graph).astype("float32")

    @staticmethod
    def is_done_state(state, step_idx, shape=(N_QUBITS, N_QUBITS)):
        unitaries = state.copy()
        graph = state.copy()
        unitaries = (unitaries[:N_UNITARIES * shape[0]]).reshape((shape[0], N_UNITARIES))
        graph = (graph[N_UNITARIES * shape[0]:]).reshape(shape)
        return step_idx >= MAX_STEPS or (np.add(graph, np.diag(np.ones(N_QUBITS))).all() and unitaries.any(1).all())

    @staticmethod
    def initial_state(shape=(N_QUBITS, N_QUBITS)):
        return np.zeros(shape[0] * N_UNITARIES + shape[0] * shape[1])

    @staticmethod
    def get_obs_for_states(states):
        return np.array(states).astype("float32")

    
    @staticmethod
    def get_return(parent_states, shape=(N_QUBITS, N_QUBITS)):
        states = deepcopy(parent_states)
        states_graph = deepcopy(parent_states)
        energies = []
        for i in range(len(states)):
            unitaries = (states[i][:N_UNITARIES * shape[0]]).reshape((shape[0], N_UNITARIES))
            graph = (states_graph[i][N_UNITARIES * shape[0]:]).reshape(shape)
            stabilizer_state = create_stabilizer_state(graph)
            for i in range(len(unitaries)):
                pos = unitaries[i].argmax()
                if unitaries[i][pos]: ACTIONS[N_UNITARIES * i + pos].forward(stabilizer_state)
            current_energy = np.real(stabilizer_state.expect(PYC_HAMILTONIAN))
            energies.append(current_energy)
        return -min(energies)
    
    # @staticmethod
    # def get_return(state, step_idx, shape=(N_QUBITS, N_QUBITS)):
    #     unitaries = deepcopy(state)
    #     graph = deepcopy(state)
    #     unitaries = (unitaries[:N_UNITARIES * shape[0]]).reshape((shape[0], N_UNITARIES))
    #     graph = (graph[N_UNITARIES * shape[0]:]).reshape(shape)
    #     stabilizer_state = create_stabilizer_state(graph)
    #     for i in range(len(unitaries)):
    #         pos = unitaries[i].argmax()
    #         if unitaries[i][pos]: ACTIONS[N_UNITARIES * i + pos].forward(stabilizer_state)
    #     current_energy = np.real(stabilizer_state.expect(PYC_HAMILTONIAN))
    #     return INIT_ENERGY - current_energy
    
    @staticmethod
    def get_current_energy(state, shape=(N_QUBITS, N_QUBITS)):
        unitaries = state.copy()
        graph = state.copy()
        unitaries = (unitaries[:N_UNITARIES * shape[0]]).reshape((shape[0], N_UNITARIES))
        graph = (graph[N_UNITARIES * shape[0]:]).reshape(shape)

        stabilizer_state = create_stabilizer_state(graph)
        for i in range(len(unitaries)):
            pos = unitaries[i].argmax()
            if unitaries[i][pos]: ACTIONS[N_UNITARIES * i + pos].forward(stabilizer_state)
        current_energy = np.real(stabilizer_state.expect(PYC_HAMILTONIAN))
        return current_energy
    
    @staticmethod
    def remove_invalid_actions(state, weights):
        unitaries = state.copy()
        graph = state.copy()
        w = weights.copy()

        unitaries = (unitaries[:N_UNITARIES * N_QUBITS]).reshape((N_QUBITS, N_UNITARIES))
        graph = (graph[N_UNITARIES * N_QUBITS:]).reshape((N_QUBITS, N_QUBITS))
        w[np.append(np.kron((unitaries > 0).any(1), [True] * N_UNITARIES), [False] * N_EDGES)] = 0
        

        graph_edges = np.transpose(np.nonzero(graph))
        for edge_loc in range(len(EDGE_LOCATIONS)):
            if np.any(np.all(np.isin(graph_edges, EDGE_LOCATIONS[edge_loc]), axis=1)):
                w[N_UNITARIES * N_QUBITS + edge_loc] = 0
        return w

    @staticmethod
    def get_n_legal_actions(state, shape=(N_QUBITS, N_QUBITS)):
        unitaries = state.copy()
        graph = state.copy()
        unitaries = (unitaries[:N_UNITARIES * shape[0]]).reshape((shape[0], N_UNITARIES))
        graph = (graph[N_UNITARIES * shape[0]:]).reshape(shape)
        
        weights = GraphEnv.remove_invalid_actions(state, np.ones(N_ACTIONS))

        legal = np.count_nonzero(weights)

        return legal

    def get_min_energy(self):
        return self.min_energy + EE_NUC

if __name__ == '__main__':
    env = GraphEnv()
    env.render()
