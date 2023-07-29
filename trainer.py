import torch
import torch.nn as nn
import torch.nn.functional as F
from graph_env import GraphEnv
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Trainer:
    """
    Trainer for an MCTS policy network. Trains the network to minimize
    the difference between the value estimate and the actual returns and
    the difference between the policy estimate and the refined policy estimates
    derived via the tree search.
    """

    def __init__(self, Policy, learning_rate=0.01):

        self.step_model = Policy()

        value_criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.step_model.parameters(),
                                     lr=learning_rate)

        def train(obs, search_pis, returns):

            search_pis = np.array([GraphEnv.remove_invalid_actions(obs[i], search_pis[i]) for i in range(len(search_pis))])

            obs = torch.from_numpy(obs)
            search_pis = torch.from_numpy(search_pis)
            returns = torch.from_numpy(returns)

            for i in range(1):
                optimizer.zero_grad()
                logits, policy, value = self.step_model(obs)

                logsoftmax = nn.LogSoftmax(dim=1)
                # print(search_pis)
                # print(logsoftmax(logits))
                # exit()

                # zip obs and search_pis and print them
                # print("---DEBUG---")
                # for i in range(len(obs)):
                #     print(obs[i], search_pis[i])
                # print("---DEBUG---")

                policy_loss = torch.mean(torch.sum(-search_pis * logsoftmax(logits), dim=1))
                # policy_loss = F.cross_entropy(logits, search_pis)
                value_loss = 10 * value_criterion(value, returns)
                # value_loss = torch.tensor([0])
                loss = policy_loss + value_loss
                # print policy and value loss separately
                # print(policy_loss.data.numpy(), value_loss.data.numpy())
                loss.backward()
                optimizer.step()
                # print(loss.item())

            # exit()
            
            # exit()

            return value_loss.data.numpy(), policy_loss.data.numpy()

        self.train = train
