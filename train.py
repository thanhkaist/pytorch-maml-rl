import gym
import torch
import json
import os
import yaml
from tqdm import trange
import numpy as np

import maml_rl.envs
from maml_rl.metalearners import MAMLTRPO
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.samplers import MultiTaskSampler
from maml_rl.utils.helpers import get_policy_for_env, get_input_size
from maml_rl.utils.reinforcement_learning import get_returns
from torch.utils.tensorboard import SummaryWriter


class PriorTaskBuffer:
    def __init__(self, size = 100):
        self.size = size
        self.losses = np.zeros(self.size) -1000
        self.task_buffer = np.array([dict() for i in range(self.size)])


    def select_most_difficult_task(self,losses, tasks, n = 10):
        '''
        Select n most difficult tasks form losses and corresponding tasks

        :param losses:  list of losses
        :param tasks: list of tasks
        :param n: number of task
        :return: selected task
        '''
        ind = np.argpartition(losses, -n)[-n:]

        sel_loss=  losses[ind]
        sel_task = tasks[ind]

        return  sel_loss , sel_task


    def add_task(self,losses,tasks):
        '''
        Replace the lowest validation lost task by new tasks
        :return:
        '''
        n = len(tasks)
        if n > self.size:
            raise Exception("Over buffer size")

        # get *n* index of lowest loss
        ind = np.argpartition(self.losses, n)[:n]

        for i, loss , task in zip(ind,losses,tasks):
            self.losses[i] = loss
            self.task_buffer[i] = task


    def sample_max_dif(self,n = 10):
        '''
        Sample the highest difficult tasks in the buffer. The task which was sampled will be removed from the buffer
        :param n:  number of sampled tasks
        :return: list of task
        '''

        if n > self.get_current_size():
            raise Exception("The buffer doesnot have enough tasks to sample")

        # get *n* index of higher loss elements in buffer
        ind = np.argpartition(self.losses, -n)[-n:]
        ret = self.task_buffer[ind]
        self.task_buffer[ind] = dict()
        self.losses[ind] = -1000
        return  ret


    def get_current_size(self):
        return sum(self.task_buffer != 0 )


    def sample_dis(self,n = 10):
        '''
        Sample tasks with probability according to the difficulty. The task which was sampled will be removed from the buffer
        :param size: number of sampled tasks
        :return: list of task
        '''

        if n > self.get_current_size():
            raise Exception("The buffer doesnot have enough tasks to sample")

        def softmax(x):
            """Compute softmax values for each sets of scores in x."""
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()
        ind = np.random.choice(self.size, n, replace=False, p=softmax(self.losses))

        ret = self.task_buffer[ind]
        self.task_buffer[ind] = dict()
        self.losses[ind] = -1000
        return ret

    def reset(self):
        self.losses = np.zeros(self.size)
        self.task_buffer = np.zeros(self.size)


def main(args):

    log_dir = "log"
    save_dir = "save"
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.output_folder is not None:
        log_dir = os.path.join(log_dir,args.output_folder)
        save_dir = os.path.join(save_dir,args.output_folder)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        policy_filename = os.path.join(save_dir, 'policy.th')
        config_filename = os.path.join(save_dir, 'config.json')

        with open(config_filename, 'w') as f:
            config.update(vars(args))
            json.dump(config, f, indent=2)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    writer = SummaryWriter(log_dir)

    env = gym.make(config['env-name'], **config['env-kwargs'])
    env.close()

    # Policy
    policy = get_policy_for_env(env,
                                hidden_sizes=config['hidden-sizes'],
                                nonlinearity=config['nonlinearity'])
    policy.share_memory()

    # Baseline
    baseline = LinearFeatureBaseline(get_input_size(env))

    # Sampler
    sampler = MultiTaskSampler(config['env-name'],
                               env_kwargs=config['env-kwargs'],
                               batch_size=config['fast-batch-size'],
                               policy=policy,
                               baseline=baseline,
                               env=env,
                               seed=args.seed,
                               num_workers=args.num_workers)

    metalearner = MAMLTRPO(policy,
                           fast_lr=config['fast-lr'],
                           first_order=config['first-order'],
                           device=args.device)

    num_iterations = 0


    first_train = True
    num_prior = 10
    buffer = PriorTaskBuffer(size=100)

    for batch in trange(config['num-batches']):
        if first_train:
            tasks = sampler.sample_tasks(num_tasks=config['meta-batch-size'])
            first_train = False
        else:
            tasks =  sampler.sample_tasks(num_tasks=config['meta-batch-size'] - num_prior)
            tasks.extend(buffer.sample_max_dif(num_prior).tolist())

        futures = sampler.sample_async(tasks,
                                       num_steps=config['num-steps'],
                                       fast_lr=config['fast-lr'],
                                       gamma=config['gamma'],
                                       gae_lambda=config['gae-lambda'],
                                       device=args.device)
        logs = metalearner.step(*futures,
                                max_kl=config['max-kl'],
                                cg_iters=config['cg-iters'],
                                cg_damping=config['cg-damping'],
                                ls_max_steps=config['ls-max-steps'],
                                ls_backtrack_ratio=config['ls-backtrack-ratio'])

        train_episodes, valid_episodes = sampler.sample_wait(futures)
        num_iterations += sum(sum(episode.lengths) for episode in train_episodes[0])
        num_iterations += sum(sum(episode.lengths) for episode in valid_episodes)
        # logs.update(tasks=tasks,
        #             num_iterations=num_iterations,
        #             train_returns=get_returns(train_episodes[0]),
        #             valid_returns=get_returns(valid_episodes))

        logs.update(train_returns=get_returns(train_episodes[0]),
                    valid_returns=get_returns(valid_episodes))


        # Update prioritize buffer

        loss_after = np.squeeze(np.array(logs['loss_after']))
        cor_tasks = np.array(tasks)

        sel_losses, sel_tasks = buffer.select_most_difficult_task(loss_after,cor_tasks,num_prior)
        buffer.add_task(sel_losses,sel_tasks)


        # log tensorboard
        for key in logs.keys():
            writer.add_scalar(key,np.mean(logs[key]),batch)

        # Save policy
        if args.output_folder is not None:
            with open(policy_filename, 'wb') as f:
                torch.save(policy.state_dict(), f)


if __name__ == '__main__':
    import argparse
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML) - Train')

    parser.add_argument('--config', type=str, required=True,
        help='path to the configuration file.')

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--output-folder', type=str,
        help='name of the output folder')
    misc.add_argument('--seed', type=int, default=None,
        help='random seed')
    misc.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling (default: '
             '{0})'.format(mp.cpu_count() - 1))
    misc.add_argument('--use-cuda', action='store_true',
        help='use cuda (default: false, use cpu). WARNING: Full upport for cuda '
        'is not guaranteed. Using CPU is encouraged.')

    args = parser.parse_args()
    args.device = ('cuda' if (torch.cuda.is_available()
                   and args.use_cuda) else 'cpu')

    main(args)
