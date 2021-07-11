import time
import torch
import numpy as np
from tqdm import tqdm
from DQN.run_dqn import get_state
from DQN.atari_wrappers import make_atari, wrap_deepmind
from DQN.models import QNet, DuelingQNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_dqn(env_name, n_episodes=100, model_path=None, model='DQN', render=False):
    if model_path is None:
        model_path = 'trained_models/' + env_name + '_model.pth'

    env_raw = make_atari(env_name, max_episode_steps=10000)
    env = wrap_deepmind(env_raw, clip_rewards=False, frame_stack=True)

    if model == 'Dueling_DQN':
        net = DuelingQNet(action_size=env.action_space.n).to(device)
    else:  # default DQN
        net = QNet(action_size=env.action_space.n).to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()

    reward_list = []
    for i_episode in tqdm(range(n_episodes)):
        obs = env.reset()
        state = get_state(obs)
        reward_sum = 0
        while True:
            if render:
                env.render()
                time.sleep(0.02)

            action = net(state.to(device)).max(1)[1].view(1, 1)
            obs, reward, done, _ = env.step(action)
            next_state = get_state(obs)

            reward_sum += reward
            state = next_state

            if done:
                break
        reward_list.append(reward_sum)

    env.close()

    reward_list = np.array(reward_list)
    avg_r = np.mean(reward_list)
    std_r = np.std(reward_list)
    print("Reward sum avg: ", avg_r, "std: ", std_r)


