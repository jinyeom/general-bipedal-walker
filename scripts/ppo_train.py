# Copied from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

import copy
import glob
import os
from time import time
from collections import deque

import gym
import numpy as np
import torch as pt
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage

def main(args):
  pt.manual_seed(args.seed)
  pt.cuda.manual_seed_all(args.seed)

  pt.set_num_threads(1)
  device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')

  envs = make_vec_envs(args.env_name, args.seed, args.num_processes, args.gamma, args.log_path, device, False)

  actor_critic = Policy(envs.observation_space.shape, envs.action_space, 
                        base_kwargs={'recurrent': args.recurrent_policy}).to(device)

  agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch, args.value_loss_coef, 
                   args.entropy_coef, lr=args.lr, eps=args.eps, max_grad_norm=args.max_grad_norm)

  rollouts = RolloutStorage(args.num_steps, args.num_processes, envs.observation_space.shape, 
                            envs.action_space, actor_critic.recurrent_hidden_state_size)

  writer = SummaryWriter()

  obs = envs.reset()
  rollouts.obs[0].copy_(obs)
  rollouts.to(device)

  num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

  start = time()

  for j in tqdm(range(num_updates), desc='Updates'):
    if args.use_linear_lr_decay:
      utils.update_linear_schedule(agent.optimizer, j, num_updates, args.lr)

    episode_rewards = []

    for step in range(args.num_steps):
      with pt.no_grad():
        value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
          rollouts.obs[step], 
          rollouts.recurrent_hidden_states[step],
          rollouts.masks[step])

      obs, reward, done, infos = envs.step(action)
      episode_rewards.append(reward)

      # If done then clean the history of observations.
      masks = pt.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
      bad_masks = pt.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
      rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, bad_masks)

    episode_rewards = sum(episode_rewards)

    with pt.no_grad():
      next_value = actor_critic.get_value(rollouts.obs[-1], rollouts.recurrent_hidden_states[-1], rollouts.masks[-1]).detach()

    rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits)
    value_loss, action_loss, dist_entropy = agent.update(rollouts)
    rollouts.after_update()

    # save for every interval-th episode or for the last epoch
    if j % args.save_interval == 0 or j == num_updates - 1:
      try:
        os.makedirs(args.save_path)
      except OSError:
        pass

      pt.save([actor_critic, getattr(utils.get_vec_normalize(envs), 'ob_rms', None)], 
              os.path.join(args.save_path, f'PPO_{args.env_name}.pt'))

    end = time()
    total_num_steps = (j + 1) * args.num_processes * args.num_steps
    writer.add_scalar('Train/FPS', int(total_num_steps / (end - start)), j)

    writer.add_scalars('Train/reward', {
      'mean': pt.mean(episode_rewards),
      'median': pt.median(episode_rewards),
    }, j)
    writer.add_scalar('Train/loss/policy', action_loss, j) 
    writer.add_scalar('Train/loss/value', value_loss, j) 
    writer.add_scalar('Train/loss/entropy', dist_entropy, j)

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
  parser.add_argument('--eps', type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
  parser.add_argument('--alpha', type=float, default=0.99, help='RMSprop optimizer apha (default: 0.99)')
  parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
  parser.add_argument('--use-gae', action='store_true', default=False, help='use generalized advantage estimation')
  parser.add_argument('--gae-lambda', type=float, default=0.95, help='gae lambda parameter (default: 0.95)')
  parser.add_argument('--entropy-coef', type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
  parser.add_argument('--value-loss-coef', type=float, default=0.5, help='value loss coefficient (default: 0.5)')
  parser.add_argument('--max-grad-norm', type=float, default=0.5, help='max norm of gradients (default: 0.5)')
  parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
  parser.add_argument('--num-processes', type=int, default=16, help='how many training CPU processes to use (default: 16)')
  parser.add_argument('--num-steps', type=int, default=5, help='number of forward steps (default: 5)')
  parser.add_argument('--ppo-epoch', type=int, default=4, help='number of ppo epochs (default: 4)')
  parser.add_argument('--num-mini-batch', type=int, default=32, help='number of batches for ppo (default: 32)')
  parser.add_argument('--clip-param', type=float, default=0.2, help='ppo clip parameter (default: 0.2)')
  parser.add_argument('--save-interval', type=int, default=100, help='save interval, one save per n updates (default: 100)')
  parser.add_argument('--num-env-steps', type=int, default=10e6, help='number of environment steps to train (default: 10e6)')
  parser.add_argument('--env-name', default='PongNoFrameskip-v4', help='environment to train on (default: PongNoFrameskip-v4)')
  parser.add_argument('--log-path', default='/tmp/gym/', help='directory to save agent logs (default: /tmp/gym)')
  parser.add_argument('--save-path', default='./trained_models/', help='directory to save agent logs (default: ./trained_models/)')
  parser.add_argument('--use-proper-time-limits', action='store_true', default=False, help='compute returns taking into account time limits')
  parser.add_argument('--recurrent-policy', action='store_true', default=False, help='use a recurrent policy')
  parser.add_argument('--use-linear-lr-decay', action='store_true', default=False, help='use a linear schedule on the learning rate')
  args = parser.parse_args()
  main(args)