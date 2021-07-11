import os
import argparse
from DQN.run_dqn import run_dqn
from DQN.test_dqn import test_dqn
from PPO.run_ppo import run_ppo
from PPO.test_ppo import test_ppo
from SAC.run_sac import run_sac
from SAC.test_sac import test_sac

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='PongNoFrameskip-v4', help='Name of the environment to run')
parser.add_argument('--method', type=str, default='SAC', help='Method to use in MuJoCo environments, PPO or SAC')
parser.add_argument('--train', action='store_true', help='Train the model')
args = parser.parse_args()

if args.env_name in ['PongNoFrameskip-v4', 'BoxingNoFrameskip-v4']:
    os.chdir("DQN")
    print('Environment: ' + args.env_name + ' Method: DQN')
    if args.train:
        print('Train start!')
        run_dqn(args.env_name)
    else:
        print('Test start!')
        test_dqn(args.env_name)
elif args.env_name in ['Hopper-v2', 'HalfCheetah-v2', 'Ant-v2']:
    if args.method == 'SAC':
        os.chdir('SAC')
        print('Environment: ' + args.env_name + ' Method: SAC')
        if args.train:
            print('Train start!')
            run_sac(args.env_name)
        else:
            print('Test start!')
            test_sac(args.env_name)
    elif args.method == 'PPO':
        os.chdir('PPO')
        print('Environment: ' + args.env_name + ' Method: PPO')
        if args.train:
            print('Train start!')
            run_ppo(args.env_name)
        else:
            print('Test start!')
            test_ppo(args.env_name)
    else:
        print('Environment: ' + args.env_name + ' does not support method '+args.method)
else:
    print('Environment ' + args.env_name + ' not support!')
