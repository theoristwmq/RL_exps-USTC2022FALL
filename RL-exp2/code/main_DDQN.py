# Nota that this network won't work because the reward is always 1
"""Double DQN"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from torch.autograd import Variable
import random
from torch.utils.tensorboard import SummaryWriter
from make_env import make_gymenv
from collections import deque

DEVICE='cuda:0' if torch.cuda.is_available() else 'cpu' # cuda:0
# hyper-parameters
BATCH_SIZE = 32
LR = 0.0001
GAMMA = 0.99
SAVING_IETRATION = 1000
MEMORY_CAPACITY = 72000
Q_NETWORK_ITERATION = 100

env = make_gymenv('env.yaml')


# env = env.unwrapped
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape

class AtariNet(nn.Module):
    def __init__(self, num_inputs=4,
                 ):
        super(AtariNet, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        # self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.linear = nn.Linear(3136, 512)
        self.linear2 = nn.Linear(512, NUM_ACTIONS)
        self.linear3 = nn.Linear(512, 1)


    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), inplace=False)
        x = F.leaky_relu(self.conv2(x), inplace=False)
        x = F.leaky_relu(self.conv3(x), inplace=False)

        # x4 = F.relu(self.conv4(x3), inplace=False)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.linear2(x)
        return x


class Data:
    def __init__(self, s, r, d, ns):
        self.s = s
        self.r = r
        self.d = d
        self.ns = ns

class Memory:
    def __init__(self, n):
        self.n = n
        self.memory = deque(maxlen=self.n)

    # TODO
    def set(self, data):
        state = data.s.__array__()
        next_state = data.ns.__array__()

        state = torch.FloatTensor(state)
        action = torch.LongTensor([data.r])
        reward = torch.FloatTensor(np.array([data.d]))
        next_state = torch.FloatTensor(next_state)

        self.memory.append((state, action, reward, next_state,))

    # TODO
    def get(self, index):
        batch = random.choices(self.memory, k=len(index))
        state, action, reward, next_state = map(torch.stack, zip(*batch))
        return state.squeeze(), action.squeeze(), reward.squeeze(), next_state.squeeze()


class DQN():
    """docstring for DQN"""
    def __init__(self):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = AtariNet().cuda(), AtariNet().cuda()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = Memory(MEMORY_CAPACITY)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state, EPISILO = 1.0):
        state = torch.tensor(state, device=DEVICE, dtype=torch.float)
        if np.random.random() > EPISILO:# greedy policy
            action_value = self.eval_net.cuda().forward(state)
            action = torch.max(action_value, 1)[1].data.cpu().numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else: # random policy
            action = np.random.randint(0,NUM_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, data):
        self.memory.set(data)
        self.memory_counter += 1

    def learn(self):
        # update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        if self.learn_step_counter % SAVING_IETRATION == 0:
            self.save_train_model(self.learn_step_counter)

        self.learn_step_counter += 1

        # sample batch from memory
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        # TODO
        batch_memory = self.memory.get(index=sample_index)
        b_s, b_a, b_r, b_ns = batch_memory

        b_s = Variable(b_s)
        b_a = Variable(b_a)
        b_r = Variable(b_r)
        b_ns = Variable(b_ns)

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s.cuda()).gather(1, b_a.unsqueeze(1).cuda())  # shape (batch, 1)

        q_eval_test = self.eval_net(b_s.cuda())
        # argmax axis = 0 means column , 1 means row
        # we choose the max acion value , the action is column , so axis = 1
        Q1_argmax = np.argmax(q_eval_test.data.cpu().numpy(), axis=1)

        # q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_next = self.target_net(b_ns.cuda())

        q_next_numpy = q_next.data.cpu().numpy()

        q_update = np.zeros((BATCH_SIZE, 1))
        for iii in range(BATCH_SIZE):
            q_update[iii] = q_next_numpy[iii, Q1_argmax[iii]]

        q_update = GAMMA * q_update
        q_update = torch.FloatTensor(q_update)

        variable11 = Variable(q_update)
        q_target = b_r.unsqueeze(1) + variable11
        # q_target = b_r + GAMMA * q_next.max(1)[0]   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target.cuda())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_train_model(self, epoch):
        torch.save(self.eval_net.state_dict(), "./data/" + str(epoch) + ".pt")

    def load_model(self,file):
        self.eval_net.cuda().load_state_dict(torch.load(file))

def main():
    dqn = DQN()
    episodes = 400000
    EPISILO = 1.0
    writer = SummaryWriter('./log/dqn')
    TEST = False
    if TEST:
        dqn.load_model('./model/5815000.pt')
    print("Collecting Experience....")
    for i in range(episodes):
        print("EPISODE: ", i)
        state = env.reset() # [1, 4, 84, 84]
        EPISILO = max(0.1, EPISILO * (1 - i / episodes))
        ep_reward = 0
        index_train = 0
        while True:
            action = dqn.choose_action(state, EPISILO if not TEST else 0)
            next_state, reward, done, info = env.step(action)
            dqn.store_transition(Data(state, action, reward, next_state))
            ep_reward += reward

            if dqn.memory_counter >= MEMORY_CAPACITY and not TEST:
                if index_train == 7:
                    dqn.learn()
                    index_train = 0
                else:
                    index_train += 1

                if done:
                    print("episode: {} , the episode reward is {}".format(i, np.round(ep_reward, 3)))
            if done[0]:
                break
            state = next_state
        writer.add_scalar('reward', ep_reward, global_step=i)


if __name__ == '__main__':
    main()