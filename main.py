"""
Example program that uses the single-player MCTS algorithm to train an agent
to master the HillClimbingEnvironment, in which the agent has to reach the
highest point on a map.
"""
import shutup
shutup.please()
import time
import numpy as np
import matplotlib.pyplot as plt

from trainer import Trainer
from policy import HillClimbingPolicy, GraphPolicy
from replay_memory import ReplayMemory
from hill_climbing_env import HillClimbingEnv
from graph_env import GraphEnv
from mcts import execute_episode

use_case = 'graph'

if use_case == 'graph':
    N_QUBITS = 2
    N_UNITARIES = 23
    ENV = GraphEnv
    POLICY = GraphPolicy
    N_ACTIONS = 23 * N_QUBITS + N_QUBITS * (N_QUBITS - 1) // 2
    N_OBS = 23 * N_QUBITS + N_QUBITS * N_QUBITS
    OBS_TYPE = np.float32
    OBTYPE = [N_OBS]
else:
    ENV = HillClimbingEnv
    POLICY = HillClimbingPolicy
    N_ACTIONS = 4
    N_OBS = 49
    OBS_TYPE = np.long
    OBTYPE = []

def log(test_env, iteration, step_idx, total_rew):

    time.sleep(0.1)
    print()
    test_env.render()
    print(f"Training Episodes: {iteration}")
    print(f"Step: {step_idx}")
    print(f"Return: {total_rew}")


if __name__ == '__main__':
    n_actions = N_ACTIONS
    n_obs = N_OBS

    trainer = Trainer(lambda: POLICY(n_obs, 20, n_actions))
    network = trainer.step_model

    mem = ReplayMemory(200,
                       {"ob": OBS_TYPE,
                        "pi": np.float32,
                        "return": np.float32},
                       {"ob": OBTYPE,
                        "pi": [n_actions],
                        "return": []})

    def test_agent(iteration):
        test_env = ENV()
        total_rew = 0
        state, reward, done, _ = test_env.reset()
        step_idx = 0
        while not done:
            log(test_env, iteration, step_idx, total_rew)
            p, _ = network.step(np.array([state]))

            p = p[0]

            p = ENV.remove_invalid_actions(state, p)
            
            # print(p)
            action = np.argmax(p)
            print('Action:', action)
            state, reward, done, _ = test_env.step(action)
            step_idx += 1
            total_rew += reward
        
        log(test_env, iteration, step_idx, total_rew)
        print("min_energy:", test_env.get_min_energy())
        print("min_state", test_env.min_stabilizer_state)

    value_losses = []
    policy_losses = []

    for i in range(1000):
        print(i)
        print("Training Iteration:", i)
        if i % 50 == 0:
            test_agent(i)
            plt.plot(value_losses, label="value loss")
            plt.plot(policy_losses, label="policy loss")
            plt.legend()
            plt.show()

        obs, pis, returns, total_reward, done_state = execute_episode(network,
                                                                      32,
                                                                      ENV)
        
        mem.add_all({"ob": obs, "pi": pis, "return": returns})

        np.set_printoptions(threshold=np.inf)
        batch = mem.get_minibatch()
        # print(batch)
        # print("Length of Batch:", len(batch["ob"]))
        # print("Unique obs in batch:", len(np.unique(batch["ob"], axis=0)))
        # print("unique batch:", np.unique(batch["ob"], axis=0))

        vl, pl = trainer.train(batch["ob"], batch["pi"], batch["return"])
        value_losses.append(vl)
        policy_losses.append(pl)
        