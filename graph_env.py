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
    
    print('Done calculating qubit_op.')
    is_identity = lambda x: str(x.primitive).count('I') == len(str(x.primitive))
    const_shift = sum([x.coeff for x in qubit_op if is_identity(x)])
    pyc_hamiltonian = binary_sum([x.coeff * paulialg.pauli(str(x.primitive)) for x in qubit_op if not is_identity(x)])
    print('Done calculating pyclifford Hamiltonian.')
    if pyc_hamiltonian.N >= 14 or use_pyscf:
        print("Using pyscf ground state estimate")
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
        print("Using exact ground state")
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
    if type == 'H':
        gate = circuit.CliffordGate(qubit)
        gate.set_forward_map(H)
        return gate
    if type == 'S':
        gate = circuit.CliffordGate(qubit)
        gate.set_generator(paulialg.pauli('Z'))
        return gate
    raise Exception('Invalid gate type.')

def construct_circ(gates):
    circ = circuit.CliffordCircuit()
    for gate in gates:
        circ.take(gate)
    return circ

def construct_all_circuits(qubits):
    possibleops = ['', 'H', 'S', 'HS', 'SH', 'SS', 'HSH', 'SHS', 'SSH', 'HSS', 'SSS', 'SSSH', 'HSSS', 'HSHS', 'SHSH', 'HSSSH', 'SHSSH', 'SSHSH', 'SHSHS', 'HSSHS', 'SHSSS', 'SSSHS', 'SSHSS', 'HSHSH']
    circuits = [] # [[qubit0circ0, qubit0circ1, ...], [qubit1circ0, qubit1circ1, ...]
    for i in range(qubits):
        qubit_circuits = []
        for op in possibleops:
            qubit_circuits.extend(construct_circ([gate(i, k) for k in op]))
        circuits + qubit_circuits
    return circuits

def create_edges(qubits):
    edges = []
    for i in range(qubits):
        for j in range(i + 1, qubits):
            edge = np.zeros((qubits, qubits))
            edge[i][j] = 1
            edge[j][i] = 1
            edges.append(edge)
    return edges

class GraphEnv(gym.Env, StaticEnv):
    """
    gym environment with the goal to navigate the player from its
    starting position to the lowest energy node by choosing a graph state and local unitary. 
    Rewards are defined as the negative of the energy between states minus a penalty for each step. 
    The player starts with no connected nodes. 
    """ 

    def __init__(self):
        mol = "H2O"
        bond_dis = 1.6
        if mol == "H2":
            self.pyc_hamiltonian, self.ee_exact, self.ee_nuc, self.ee_hf = calculate_hamiltonian([["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, bond_dis]]], True)
        elif mol == "H2O":
            # c1 = np.array([-0.0399, -0.0038, 0.0])
            # c2 = np.array([1.5780, 0.8540, 0.0])
            # c3 = np.array([2.7909, -0.5159, 0.0])
            c1 = np.array([bond_dis * np.cos(np.pi * 104.5 / 180), bond_dis * np.sin(np.pi * 104.5 / 180), 0.0])
            c2 = np.array([0.0, 0.0, 0.0])
            c3 = np.array([bond_dis, 0.0, 0.0])
            bond1 = np.linalg.norm(c1 - c2)
            bond2 = np.linalg.norm(c3 - c2) 
            print("bond length:", bond1, bond2)
            self.pyc_hamiltonian, self.ee_exact, self.ee_nuc, self.ee_hf = calculate_hamiltonian([["H", c1], ["O", c2], ["H", c3]], True)
        
        self.n_qubits = self.pyc_hamiltonian[0].N

        self.actions = construct_all_circuits(self.n_qubits) + create_edges(self.n_qubits)
        
        self.shape = (self.n_qubits, self.n_qubits)

        self.graph = np.zeros(self.shape)

        self.unitaries = [''] * 10

        self.step_idx = 0

        self.state = create_stabilizer_state(self.graph)

        self.energy = np.real(self.state.expect(self.pyc_hamiltonian))

    def calculate_energy(self):
        newstate = create_stabilizer_state(self.graph)

    def reset(self):

        self.graph = np.zeros(self.shape)

        self.unitaries = [''] * 10
        
        self.step_idx = 0
        state = self.pos[0] * self.shape[0] + self.pos[1]
        return state, 0, False, None

    def step(self, action):
        self.step_idx += 1

        if action < self.n_qubits * 24 and self.unitaries[action // 24] == '':
            self.unitaries[action // 24] = self.actions[action]
            self.actions[action].forward(self.state)

        elif action >= self.n_qubits * 24:
            temp_graph = np.add(self.graph, self.actions[action])
            if (temp_graph > 1).any(): return state, -0.5, False, None # reward = -0.5 for already connected edge, not done
            self.graph = temp_graph
            self.state = create_stabilizer_state(self.graph)
            for unitary in self.unitaries:
                if unitary: self.state = unitary.forward(self.state)

        else: return state, -0.5, False, None # reward = -0.5 for invalid action, not done
        
        e_after = np.real(self.state.expect(self.pyc_hamiltonian))
        reward = self.energy - e_after - 0.5   # -0.5 for encouraging speed
        state = self.pos[0] * self.shape[0] + self.pos[1]
        done = self.pos == (0, 6) or self.step_idx == self.ep_length
        return state, reward, done, None

    def render(self, mode='human'):
        if mode != 'human':
            print(self.pos)
            return
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                end = " " if y < self.shape[0] - 1 else ""
                bg_color = self.altitude_colors[self.altitudes[x][y]]
                color = "white" if bg_color == "black" else "black"
                if self.pos == (x, y):
                    cprint(" P ", "red", bg_color)
                else:
                    cprint(f" {self.altitudes[x][y]} ", color, bg_color)
            print()

    @staticmethod
    def next_state(state, action, shape=(7, 7)):
        
        pos = np.unravel_index(state, shape)
        if action == UP:
            pos = (pos[0] - 1, pos[1])
        if action == DOWN:
            pos = (pos[0] + 1, pos[1])
        if action == LEFT:
            pos = (pos[0], pos[1] - 1)
        if action == RIGHT:
            pos = (pos[0], pos[1] + 1)
        pos = HillClimbingEnv._limit_coordinates(pos, shape)
        return pos[0] * shape[0] + pos[1]

    @staticmethod
    def is_done_state(state, step_idx, shape=(7, 7)):
        return np.unravel_index(state, shape) == (0, 6) or step_idx >= 15

    @staticmethod
    def initial_state(shape=(7, 7)):
        return (shape[0] - 1) * shape[0]

    @staticmethod
    def get_obs_for_states(states):
        return np.array(states)

    @staticmethod
    def get_return(state, step_idx, shape=(7, 7)):
        row, col = np.unravel_index(state, shape)
        return HillClimbingEnv.altitudes[row][col] - step_idx * 0.5

    @staticmethod
    def _limit_coordinates(coord, shape):
        """
        Prevent the agent from falling out of the grid world.
        Adapted from
        https://github.com/openai/gym/blob/master/gym/envs/toy_text/cliffwalking.py
        """
        coord = list(coord)
        coord[0] = min(coord[0], shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return tuple(coord)


if __name__ == '__main__':
    env = HillClimbingEnv()
    env.render()
    print(env.step(UP))
    env.render()
    print(env.step(RIGHT))
    env.render()
    print(env.step(DOWN))
    env.render()
    print(env.step(LEFT))
    env.render()
