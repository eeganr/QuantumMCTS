import numpy as np
from PyClifford_old.src import *

def create_edges(qubits):
    edges = []
    for i in range(qubits):
        for j in range(i + 1, qubits):
            edge = np.zeros((qubits, qubits))
            edge[i][j] = 1
            edge[j][i] = 1
            edges.append(edge)
    return edges

def construct_all_circuits(qubits):
    possibleops = ['', 'H', 'S', 'HS', 'SH', 'SS', 'HSH', 'SHS', 'SSH', 'HSS', 'SSS', 'SSSH', 'HSSS', 'HSHS', 'SHSH', 'HSSSH', 'SHSSH', 'SSHSH', 'SHSHS', 'HSSHS', 'SHSSS', 'SSSHS', 'SSHSS', 'HSHSH']
    circuits = [] # [[qubit0circ0, qubit0circ1, ...], [qubit1circ0, qubit1circ1, ...]
    for i in range(qubits):
        qubit_circuits = []
        for op in possibleops:
            qubit_circuits.append(construct_circ([gate(i, k) for k in op]))
        circuits.extend(qubit_circuits)
    return circuits

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

x = construct_all_circuits(3)

psi = stabilizer.random_clifford_state(3)
psi2 = psi.copy()

x[5].forward(psi)
x[27].forward(psi)
x[28].forward(psi)
x[14].forward(psi)
x[59].forward(psi)

x[5].forward(psi2)
x[59].forward(psi2)
x[14].forward(psi2)
x[27].forward(psi2)
x[28].forward(psi2)

print(psi)
print(psi2)