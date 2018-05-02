import time

from agent import RLAgent
from trigger import NoTrigger, MACTrigger, LTATrigger, LTAMACTrigger
from file_manager import FileManager
from epsilon_policy import LinearlyDecayingEpsilonPolicy, ExponentiallyDecayingEpsilonPolicy
from environment import Environment
from config import config2, agent_config, train_config
from plot import plot_reward
import nel
import sys

import argparse
from collections import deque
import random
from six.moves import cPickle
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
import os


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)


def compute_td_loss(batch_size, agent, replay_buffer, gamma, optimizer, use_gpu=False):
    # Sample a random minibatch from the replay history.
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    if use_gpu:
        state = Variable(torch.FloatTensor(np.float32(state)).cuda())
        next_state = Variable(torch.FloatTensor(np.float32(next_state)).cuda())
        action = Variable(torch.LongTensor(action).cuda())
        reward = Variable(torch.FloatTensor(reward).cuda())
        done = Variable(torch.FloatTensor(done).cuda())
    else:
        state = Variable(torch.FloatTensor(np.float32(state)))
        next_state = Variable(torch.FloatTensor(np.float32(next_state)))
        action = Variable(torch.LongTensor(action))
        reward = Variable(torch.FloatTensor(reward))
        done = Variable(torch.FloatTensor(done))

    q_values = agent.policy(state)
    q_values_target = agent.target(next_state)
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = q_values_target.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    # loss = F.smooth_l1_loss(q_value,  Variable(expected_q_value.data))
    if use_gpu:
        loss = F.mse_loss(q_value,  Variable(expected_q_value.data.cuda()))
    else:
        loss = F.mse_loss(q_value,  Variable(expected_q_value.data))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.cpu()


def plot_setup():
    # plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    p1, = ax1.plot([])
    p2, = ax2.plot([])
    ax2.set_title('loss')
    print("SETUP")
    fig.canvas.draw()

    def update(frame_idx, rewards, losses):
        p1.set_xdata(range(len(rewards)))
        p1.set_ydata(rewards)

        ax1.set_title('frame %s. reward: %s' %
                      (frame_idx, np.mean([rewards[i] for i in range(-10, 0)])))
        p2.set_xdata(range(len(losses)))
        p2.set_ydata(losses)
        ax1.set_xlim([0, len(rewards)])
        ax1.set_ylim([min(rewards), max(rewards) + 10])
        ax2.set_xlim([0, len(losses)])
        ax2.set_ylim([min(losses), max(losses)])
        print(max(losses))
        ax2.set_yscale('log')
        plt.draw()
        plt.pause(0.0001)

    def save(fname):
        fig.savefig(fname)

    return update, save


def plot(frame_idx, rewards, losses):
    # clear_output(True)
    fig = plt.figure(figsize=(20, 5))
    plt.subplot(121)
    plt.title('frame %s. reward: %s' %
              (frame_idx, np.mean([rewards[i] for i in range(-10, 0)])))
    plt.plot(rewards)
    plt.subplot(122)
    plt.title('loss')
    plt.plot(losses)
    plt.show()


def train(agent, env, actions, optimizer, epsilon_policy, trigger_mechanism, trigger_file, reward_file, loss_file, epsilon_file, use_gpu):
    log_num_steps = 1000

    num_steps_save_training_run = train_config['num_steps_save_training_run']
    policy_update_frequency = train_config['policy_update_frequency']
    target_update_frequency = train_config['target_update_frequency']
    batch_size = train_config['batch_size']
    training_steps = 0
    replay = ReplayBuffer(train_config['replay_buffer_capacity'])
    discount_factor = train_config['discount_factor']
    max_steps = train_config['max_steps']
    tr_reward = 0
    agent.update_target()
    triggered = False

    reward = 0
    loss = Variable(torch.Tensor([0]))
    epsilon = 1.0

    last_log_time = time.time()

    for training_steps in range(max_steps):
        # Update current exploration parameter epsilon, which is discounted
        # with time.

        if training_steps > 0:
            triggered = trigger_mechanism.should_trigger(reward, loss.data.item())
        if triggered:
            trigger_file.write_line(str(training_steps))
        epsilon = epsilon_policy.get_epsilon(epsilon, triggered)
        epsilon_file.write_line(str(epsilon))

        add_to_replay = len(agent.prev_states) >= 1

        # Get current state.
        s1 = agent.get_state()

        # Make a step.
        action, reward = agent.step(epsilon)

        # Update state according to step.
        s2 = agent.get_state()

        # Accumulate all rewards.
        tr_reward += reward
        reward_file.write_line(str(reward))

        # Add to memory current state, action it took, reward and new state.
        if add_to_replay:
            # enum issue in server machine
            replay.push(s1, action.value, reward, s2, False)

        # Update the network parameter every update_frequency steps.
        if training_steps % policy_update_frequency == 0:
            if batch_size < len(replay):
                # Compute loss and update parameters.
                loss = compute_td_loss(
                    batch_size, agent, replay, discount_factor, optimizer,
                    use_gpu)
                loss_file.write_line(str(loss.data.item()))

        if training_steps % log_num_steps == 0 and training_steps > 0:
            current_time = time.time()
            elapsed_time = current_time - last_log_time
            last_log_time = current_time
            num_steps_per_second = float(log_num_steps) / elapsed_time
            print(
                'Step: %7d, %6.2f # steps / second, Loss: %10.8f' %
                (training_steps, num_steps_per_second, loss.data.item()))

        if training_steps % target_update_frequency == 0:
            agent.update_target()

        model_path = 'outputs/models/NELQ_' + str(training_steps)

        if training_steps % num_steps_save_training_run == 0:
            agent.save(model_path)

    model_path = 'outputs/models/NELQ_' + str(training_steps)
    agent.save(model_path)
    trigger_file.close()
    reward_file.close()
    loss_file.close()
    epsilon_file.close()

def setup_output_dir():
    m_dir = 'outputs/models'
    if not os.path.exists(m_dir):
        os.makedirs(m_dir)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', dest='id',
                        type=int, default=0,
                        help="Experiment ID")
    parser.add_argument('--dir', dest='dir',
                        type=str, default='./',
                        help="Result Directory")
    parser.add_argument('--trigger', dest='trigger',
                        type=str, default='no',
                        help="Trigger Mechanism")
    parser.add_argument('--ep', dest='ep',
                        type=str, default='linear',
                        help="Epsilon Policy")
    parser.add_argument('--use-gpu', dest='use_gpu',
                        type=bool, default=False,
                        help="Indicates whether or not to use a GPU.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    id = args.id
    dir = args.dir
    trigger_fn = dir + 'trigger_' + str(id)
    reward_fn = dir + 'reward_' + str(id)
    loss_fn = dir + 'loss_' + str(id)
    epsilon_fn = dir + 'epsilon_' + str(id)
    use_gpu = args.use_gpu

    trigger = args.trigger
    ep = args.ep

    env = Environment(config2)
    from agent import actions
    state_size = (config2.vision_range*2 + 1)**2 * config2.color_num_dims + config2.scent_num_dims + len(actions)
    agent = RLAgent(env, state_size=state_size, use_gpu=use_gpu)

    optimizer = optim.Adam(agent.policy.parameters(),
        lr=agent_config['learning_rate'])

    setup_output_dir()


    EPS_START = 1.
    EPS_END = .1
    EPS_DECAY_START = 1000.
    EPS_DELTA = .00002
    trigger_file = FileManager(trigger_fn, buffer_size=1)
    reward_file = FileManager(reward_fn)
    loss_file = FileManager(loss_fn)
    epsilon_file = FileManager(epsilon_fn)


    if ep == 'linear':
        epsilon_policy = LinearlyDecayingEpsilonPolicy(EPS_DECAY_START, EPS_START, EPS_DELTA, lower_bound=EPS_END)
    else:
        epsilon_policy = ExponentiallyDecayingEpsilonPolicy(EPS_DECAY_START, EPS_START, EPS_DELTA, lower_bound=EPS_END)

    if trigger == 'no':
        trigger_mechanism = NoTrigger()
    elif trigger == 'mac':
        trigger_mechanism = MACTrigger()
    elif trigger == 'lta':
        trigger_mechanism = LTATrigger()
    else:
        trigger_mechanism = LTAMACTrigger()

    train(agent, env, [0, 1, 2, 3], optimizer, epsilon_policy, trigger_mechanism,
          trigger_file, reward_file, loss_file, epsilon_file, use_gpu)


if __name__ == '__main__':
    main()
