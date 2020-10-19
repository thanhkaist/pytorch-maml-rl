'''Experiment with noise environment. Add noise to the training data to make the training distribution diferent from testing distribution
Half cheetar vel : onservation 20 action 6
'''
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
import torch.nn as nn
import time
import pdb

from maml_rl.episode import BatchEpisodes
### This part is for testing code
from maml_rl.utils.reinforcement_learning import reinforce_loss
from datetime import datetime, timezone


def create_episodes(env, policy,baseline, params = None,batch= 20, gamma = 0.95,gae_lambda = 1.0, device='cpu' ):
  ''' generate the train episode along with updated baseline'''
  episodes = BatchEpisodes(batch_size = batch,gamma=gamma,device=device)
  episodes.log('_createdAt', datetime.now(timezone.utc))
  t0 = time.time()
  for item in sample_trajectories(env,policy,batch,params=params):
    episodes.append(*item)
  episodes.log('duration', time.time() - t0)

  baseline.fit(episodes)
  episodes.compute_advantages(baseline,
                              gae_lambda=gae_lambda,
                              normalize=True)
  return episodes


def sample_trajectories( env, policy, batch,params=None, max_step = 100):
  with torch.no_grad():
    for i in range(batch):
      observations = env.reset()
      for j in range(max_step):
        observations_tensor = torch.from_numpy(np.array([observations])).float()
        pi = policy(observations_tensor, params=params)
        actions_tensor = pi.sample()
        actions = actions_tensor.cpu().numpy()
        new_observations, rewards, done, infos = env.step(actions.squeeze())
        batch_ids = i
        yield ([observations], [actions.squeeze()], [rewards], [batch_ids])
        observations = new_observations
        if done:
          break

### This part is for priority buffer. 

def get_buffer(sample, score,extent= None,max =50):
  ''' sort array in the ascending order '''
  if extent is not None:
    sampl_buf, score_buf  = get_sample_buff(extent,extent.shape[0])
    sample = np.concatenate((sample,sampl_buf),axis=0)
    score = np.concatenate((score,score_buf),axis=0)
    if sample.shape[0] > max:
      sample = sample[:max]
      score = score[:max]
  sort_arr = np.concatenate((sample,score),axis=1)
  sorted_arr = sorted(sort_arr,key=lambda x:x[2])
  return np.array(sorted_arr)
  
def get_sample_buff(buf,num = 5,mode=0):
  "0-easy,1-medium,2-hard"
  if mode == 0:
    sample = buf[:num,:2]
    score = buf[:num,2]
  if mode == 1:
    mean = buf.shape[0]//2
    begin = mean - num//2
    end = begin +num
    sample = buf[begin:end,:2]
    score = buf[begin:end,2]
  if mode == 2:
    sample = buf[-num:,:2]
    score = buf[-num:,2]
  return sample,np.reshape(score,(-1,1))

### Main function

def main(args):
    print(args)
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
    policy.set_learning_rate(config['fast-lr']) #config['fast-lr']
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

    # Prioritize task
    first_train = True
    num_prior = args.num_prior
    # interpolate  buffsize 
    NUM_ITER = config['num-batches']
    SAMPLING_TASKS = config['meta-batch-size']
    buffer = []
    # Mode
    prior_type = args.prior_type
    if prior_type == "EASY":
        PRIOR_TYPE = 0
    elif prior_type == "MEDIUM":
        PRIOR_TYPE = 1
    elif prior_type == "HARD":
        PRIOR_TYPE = 2

    inter = np.uint8(np.sqrt(np.linspace(0,0.7,NUM_ITER))*SAMPLING_TASKS)
    
    # Adaptive learning rate
    policy.adapt_alpha = args.adapt_lr
    
    total_time = 0
    for batch in trange(config['num-batches']):
        tic = time.time()
        if first_train or (num_prior == 0):
            tasks = sampler.sample_tasks(num_tasks=config['meta-batch-size'])
            first_train = False
        else:
            if False: #Enable interpolate num prior 
                num_prior = inter[batch]
            tasks =  sampler.sample_tasks(num_tasks=config['meta-batch-size'] - num_prior)
            sample,score = get_sample_buff(buffer,num_prior,PRIOR_TYPE)
            sample = sample.reshape((-1,))
            tasks.extend(sample.tolist())

        futures = sampler.sample_async(tasks,
                                       num_steps=config['num-steps'],
                                       fast_lr=policy.alpha.item(),                                 ## config['fast-lr']
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
        
        logs.update(num_iterations=num_iterations)
        logs.update(train_returns=get_returns(train_episodes[0]),
                    valid_returns=get_returns(valid_episodes))

        # Update prioritize buffer
        use_loss = False
        if use_loss:
            score = np.squeeze(np.array(logs['loss_after'])) # use loss for prioritize task, higher loss=> more difficult task
        else:
            score = - np.squeeze(np.mean(np.array(logs['valid_returns']), axis=1)) # use return for prioritize task, low return => more difficult task

        cor_tasks = np.array(tasks)
        
        cor_tasks = cor_tasks.reshape((1,-1))
        score = score.reshape((1,-1))
        buffer = get_buffer(cor_tasks,score)
        # log tensorboard
        for key in logs.keys():
            writer.add_scalar(key,np.mean(logs[key]),batch)
        toc = time.time()
        total_time = total_time + toc - tic
        writer.add_scalar("Training time(min)",total_time//60,batch)
    
        # Save policy
        if args.output_folder is not None:
            with open(policy_filename, 'wb') as f:
                torch.save(policy.state_dict(), f)
    
    ## Run quick evaluate
    print("RUN QUICK EVALUATION")
    # Remove noise in env
    config['env-kwargs']['noise_ratio']= 0.0
    env = gym.make(config['env-name'], **config['env-kwargs'])
    NUMS_TASKS = 10
    for batch in trange(10):
        train_episodes_list = [[] for i in range(NUMS_TASKS)]
        valid_episodes_list = []
        tasks = env.sample_tasks(NUMS_TASKS)
        for i,task in enumerate(tasks):
            env.reset_task(task)
            params = None 
            for step in range(config['num-steps']):
                train_episodes = create_episodes(env,policy,baseline,params=params,batch=config['fast-batch-size'],device=args.device)
                loss = reinforce_loss(policy,train_episodes,params = params)
                params = policy.update_params(loss,params=params,step_size=0.01,first_order=True)
            train_episodes_list[i].append(train_episodes)
            valid_episodes = create_episodes(env,policy,baseline,params=params,device=args.device)
            valid_episodes_list.append(valid_episodes)
        rws =  [np.mean(get_returns(task_epis)) for task_epis in train_episodes_list]
        writer.add_scalar('avg_train_returns', np.mean(rws),batch)
        writer.add_scalar('avg_test_returns', np.mean(get_returns(valid_episodes_list)),batch)

    writer.close()
    ## Run quick evaluate
#    print("RUN QUICK EVALUATION")
    # Remove noise in env
#    config['env-kwargs']['noise_ratio']= 0.0
#    # Sampler
#    env = gym.make(config['env-name'], **config['env-kwargs'])  
#    env.close()
#    del sampler
#    sampler = MultiTaskSampler(config['env-name'],
#                               env_kwargs=config['env-kwargs'],
#                               batch_size=config['fast-batch-size'],
#                               policy=policy,
#                               baseline=baseline,
#                               env=env,
#                               seed=args.seed,
#                               num_workers=args.num_workers)
#
#    logs = {'tasks': []}
#    train_returns, valid_returns = [], []
#    for batch in trange(10):
#        tasks = sampler.sample_tasks(num_tasks=10)
#        futures = sampler.sample_async(tasks,
#                                       num_steps=config['num-steps'],
#                                       fast_lr=policy.alpha.item(),                                 ## config['fast-lr']
#                                       gamma=config['gamma'],
#                                       gae_lambda=config['gae-lambda'],
#                                       device=args.device)
#        train_episodes, valid_episodes = sampler.sample_wait(futures)
##        train_episodes, valid_episodes = sampler.sample(tasks,
##                                                        num_steps=config['num-steps'],
##                                                        fast_lr=policy.alpha.item(),
##                                                        gamma=config['gamma'],
##                                                        gae_lambda=config['gae-lambda'],
##                                                        device=args.device)
##
#        logs['tasks'].extend(tasks)
#        train_returns.append(get_returns(train_episodes[0]))
#        valid_returns.append(get_returns(valid_episodes))
#
#        writer.add_scalar('avg_train_returns', np.mean(get_returns(train_episodes[0])),batch)
#        writer.add_scalar('avg_test_returns', np.mean(get_returns(valid_episodes)),batch)
#
#
#
#    logs['train_returns'] = np.concatenate(train_returns, axis=0)
#    logs['valid_returns'] = np.concatenate(valid_returns, axis=0)
#    writer.close()
#    with open(result_file, 'wb') as f:
#         np.savez(f, **logs)

if __name__ == '__main__':
    import argparse
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML) - Train')

    parser.add_argument('--config', type=str, required=True,
        help='path to the configuration file.')

    parser.add_argument('--adapt_lr', action='store_true', help='adapt learning rate')

    parser.add_argument('--num_prior', type=int, default=0,
                      help='number of priority samples')
    parser.add_argument('--prior_type', type=str, choices = ["EASY","MEDIUM","HARD"],default="EASY",
                      help='prior type')
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
