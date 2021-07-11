from collections import namedtuple
import random
import numpy as np
from DQN.atari_wrappers import wrap_deepmind, make_atari
from DQN.models import QNet, DuelingQNet
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.01
EPS_DECAY = 1000000
TARGET_UPDATE = 1000
RENDER = False
lr = 0.0000625
INITIAL_MEMORY = 10000
MEMORY_SIZE = 200000
POLICY_UPDATE = 4

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def get_state(obs):
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0)


class DqnAgent:
    def __init__(self, env_raw, model_path, model="DQN"):
        # environment settings
        self.env_raw = env_raw
        self.env = wrap_deepmind(env_raw, frame_stack=True)
        self.action_size = self.env.action_space.n
        # training settings
        self.eps = EPS_START
        self.memory = ReplayMemory(MEMORY_SIZE)
        # model save path
        self.model_path = model_path

        # network settings
        if model == "Dueling_DQN":
            self.policy_net = DuelingQNet(action_size=self.action_size).to(device)
            self.target_net = DuelingQNet(action_size=self.action_size).to(device)
        else:  # default DQN
            self.policy_net = QNet(action_size=self.action_size).to(device)
            self.target_net = QNet(action_size=self.action_size).to(device)

        self.policy_net.apply(self.policy_net.init_weights)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, eps=1.5e-4)

    def select_action(self, state):
        self.eps -= (EPS_START - EPS_END) / EPS_DECAY
        self.eps = max(self.eps, EPS_END)

        if random.random() > self.eps:
            with torch.no_grad():
                return self.policy_net(state.to(device)).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_size)]], device=device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)

        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)
        next_state_batch = torch.cat(batch.next_state).to(device)
        done_Batch = torch.cat(batch.done).to(device)

        Q = self.policy_net(state_batch).gather(1, action_batch)
        target_Q = self.target_net(next_state_batch).max(1)[0].view(-1, 1)
        expected_Q = (target_Q * GAMMA) * done_Batch + reward_batch

        loss = F.smooth_l1_loss(Q, expected_Q)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train(self, n_episodes):
        reward_list = []
        step = 0
        for i_episode in tqdm(range(n_episodes)):
            obs = self.env.reset()
            state = get_state(obs)

            reward_sum = 0
            while True:
                step += 1
                action = self.select_action(state)
                obs, reward, done, _ = self.env.step(action)
                next_state = get_state(obs)
                reward_sum += reward

                reward = torch.tensor(np.array([reward]).reshape(1, -1), device='cpu', dtype=torch.float32)
                done_ = torch.tensor(np.array([not done]).reshape(1, -1), device='cpu', dtype=torch.long)

                self.memory.push(state, action.to('cpu'), next_state, reward, done_)
                state = next_state

                if step > INITIAL_MEMORY and step % POLICY_UPDATE == 0:
                    self.optimize_model()

                    if step % TARGET_UPDATE == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())

                if done:
                    reward_list.append(reward_sum)
                    break

            if (i_episode + 1) % 50 == 0:
                print('Total steps: {} Episode: {} Reward: {.2f}'.format(step, i_episode, reward_sum))
                # self.eval(n_episodes=5)

            if (i_episode + 1) % 500 == 0:
                torch.save(self.policy_net.state_dict(), self.model_path)

        self.env.close()
        return reward_list

    def eval(self, n_episodes):
        env = wrap_deepmind(self.env_raw, clip_rewards=False, frame_stack=True)
        policy = self.policy_net

        r = []
        for episode in range(n_episodes):
            obs = env.reset()
            state = get_state(obs)
            reward_sum = 0
            while True:
                action = policy(state.to(device)).max(1)[1].view(1, 1)
                obs, reward, done, _ = env.step(action)
                next_state = get_state(obs)

                reward_sum += reward
                state = next_state

                if done:
                    break
            r.append(reward_sum)

        print("Evaluate reward: %f" % np.mean(r))
        env.close()

        return np.mean(r)


def run_dqn(env_name, model="DQN", model_path=None):
    if model_path is None:
        model_path = 'trained_models/' + env_name + '_model.pth'

    env_raw = make_atari(env_name, max_episode_steps=10000)
    agent = DqnAgent(env_raw, model_path, model)

    train_r_list = agent.train(n_episodes=3000)
    np.save(env_name + '_train', np.array(train_r_list))
    torch.save(agent.policy_net.state_dict(), model_path)
