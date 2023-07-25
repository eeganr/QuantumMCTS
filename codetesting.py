import numpy as np
import shutup
shutup.please()
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
    possibleops = ['I', 'H', 'S', 'HS', 'SH', 'SS', 'HSH', 'SHS', 'SSH', 'HSS', 'SSS', 'SSSH', 'HSSS', 'HSHS', 'SHSH', 'HSSSH', 'SHSSH', 'SSHSH', 'SHSHS', 'HSSHS', 'SHSSS', 'SSSHS', 'SSHSS', 'HSHSH']
    circuits = [] # [[qubit0circ0, qubit0circ1, ...], [qubit1circ0, qubit1circ1, ...]
    for i in range(qubits):
        qubit_circuits = []
        for op in possibleops:
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

ACTIONS = construct_all_circuits(10) + create_edges(10)

print(len(ACTIONS))