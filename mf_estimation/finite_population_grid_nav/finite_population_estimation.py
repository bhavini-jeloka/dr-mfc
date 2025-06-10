# Use mf-env grid nav with appropriate config file
# Use test file and for mf-mappo, use the network we have instead and restructure code
# Begin with no obstacles and basic nav under LCP and see videos and then add obstacles
# Add estimator as a separate plug-in module while cleaning code - maybe remove num env

import numpy as np
import torch
from buffers.mean_field_buffer import MFBuffer
from buffers.wrappers.multi_population import SharedGlobalObsBufferWrapper
import importlib
import sys
from datetime import datetime
import time
from pathlib import Path
import copy
import matplotlib.pyplot as plt

sys.path.append('../mf-env')
import gym_mf_envs
import algos

class Runner():
    def __init__(self, envs, config, algorithm_config=None, env_config=None):
        
        self.env_name = config['env_name']
        self.num_envs = config["num_parallel_envs"]
        self.seed = config["seed"]
        
        self.max_ep_len = config["episode_length"]
        self.max_training_timesteps = config["training_timesteps"] if "training_timesteps" in config else None
        self.num_test_ep = config["num_test_episodes"] if "num_test_episodes" in config else None
        self._num_test_timesteps = self.num_test_ep*self.max_ep_len
        
        self.grid = config["size"]
        self.gamma = config["gamma"] if "gamma" in config else None
        self.beta = config["beta"] if "beta" in config else None
        self.gamma = config["gamma"] if "gamma" in config else None
        self.num_epoch = config["epochs"] if "epochs" in config else None
        self.eps_clip = config["eps_clip"] if "eps_clip" in config else None
        self.lr_actor = config["lr_actor"] if "lr_actor" in config else None
        self.lr_critic = config["lr_critic"] if "lr_critic" in config else None
        self.num_mini_batches = config["num_mini_batches"] if "num_mini_batches" in config else None

        self.action_dim = config["action_dim"]
        
        self.num_agent_list = list(config["num_agent_map"].values())
        self.team_list = list(config["num_agent_map"].keys())
        self.num_population = config["num_population"]
        self.total_num_agents = sum(self.num_agent_list)
        
        self.update_freq = config["update_frequency"] if "update_frequency" in config else None
        self.print_freq = config["print_frequency"] if "print_frequency" in config else None
        self.log_freq = config["log_frequency"] if "log_frequency" in config else None
        self.save_model_freq = config["save_frequency"] if "save_frequency" in config else None
        self.plot_freq = config["plot_frequency"] if "plot_frequency" in config else None
        
        # Call environment 
        self.envs = envs

        print("Configuration render mode", config["render_mode"])
        if "render_mode" in config:
            self.render = 1
            self.frame_delay = 0
        
        #TODO: change cuz currently module and class name are the same 
        self.buffers = [MFBuffer(self.num_population, self.num_agent_list[i], self.grid, self.num_envs)  for i in range(self.num_population)]
        self.buffers = SharedGlobalObsBufferWrapper(self.buffers)
        self.buffers.prep_algo(config, self.envs)
        self.algo_name = config["algorithm"]
        module = importlib.import_module("algos." + self.algo_name)
        self.algo = []
        for i in range(self.num_population):
            print(config["state_dim_actor"][i])
            print(config["state_dim_critic"][i])
            self.algo.append(getattr(module, self.algo_name)(config["state_dim_actor"][i], config["state_dim_critic"][i], self.action_dim, 
                                                             self.num_agent_list[i], self.lr_actor, self.lr_critic, self.gamma, self.num_epoch, 
                                                             self.eps_clip, self.beta, self.buffers[i], self.num_mini_batches,
                                                             num_envs=self.num_envs))
        self.warmup_training = False
        if "pretrained_dir" in config:
            self.pretrained_dir = config["pretrained_dir"] 
            self.warmup_training = True
        
        # Make additional directories
        self.log_file = config["log_file"]
        self.video_dir = config["video_dir"]
        self.save_dir = config["save_dir"] 
        self.model_dir = config["model_dir"] 
        
        # seed
        if self.seed:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)
    
    def test(self):
        
        # Loading model
        model = PolicyNetwork(state_dim_actor=(2, grid_size, grid_size), state_dim_critic=(1, grid_size, grid_size), action_dim=num_actions, policy_type="lcp_policy_3x3")
        
        # track total training time
        start_time = datetime.now().replace(microsecond=0)
        print("Started testing at (GMT) : ", start_time)
        print("============================================================================================")
        
        test_running_reward = {team:0 for team in self.team_list}

        print("First reset")
        state, _ = self.envs.reset()
        ep_reward = {team:0 for team in self.team_list}
    
        for t in range(self._num_test_timesteps):
            print('Timestep:', t)
           
            all_actions = np.zeros((self.num_envs, self.total_num_agents), dtype=int)
            start_index = 0 
            action_start_time = time.time()
            for i in range(self.num_population):
                obs = self.buffers.get_obs_data(i, state)
                action = self.algo[i].select_action(obs) 
                action = action.reshape((self.num_envs, self.num_agent_list[i]))
                end_index = start_index + self.num_agent_list[i]
                all_actions[: , start_index: end_index] = action
                start_index = end_index
            action_time = time.time()-action_start_time
            step_start_time = time.time()
            state, reward, done, terminated,_ = self.envs.step(all_actions.astype(int)) 

            step_time = time.time()-step_start_time
            
            for team, rew in reward[0].items():
                ep_reward[team] += rew
            
            #time.sleep(2)
            
            if done[0] or terminated[0]:
                for team in self.team_list:
                    test_running_reward[team] += ep_reward[team]
                    print('Reward Team {}: {}'.format(team, round(ep_reward[team], 2)))
                ep_reward = {team:0 for team in self.team_list}
            
        self.envs.close()
        
        print("============================================================================================")
        end_time = datetime.now().replace(microsecond=0)
        print("Finished training at (GMT) : ", end_time)
        for team in self.team_list:
            avg_test_reward = test_running_reward[team] / self.num_test_ep
            print('average test reward team {}: {} '.format(team, str(avg_test_reward)))
        print("============================================================================================")
        self.envs.save_gif(self.video_dir)
    
    def warmup(self, pretrained_dir):
        self.load(pretrained_dir)
    
    def cumulative_average(self, data):
        return np.cumsum(data) / np.arange(1, len(data) + 1)
    
    def moving_average(self, data, window_size=100):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    def save(self):
        """Save policy's actor and critic networks."""
        for i in range(self.num_population):
            save_path = Path(self.save_dir) / str(i)
            save_path.mkdir(parents=True, exist_ok=True)
            self.algo[i].save(Path(self.save_dir)/str(i))

    def load(self, dir=None):
        """Restore policy's networks from a saved model."""
        model_dir = dir if dir is not None else self.model_dir
        for i in range(self.num_population):
            self.algo[i].load(Path(model_dir)/str(i))

    def log_info(self, log_running_reward, log_running_episodes, time_step):
        """
        Log training info.
        """
        for running_reward in log_running_reward.values():
            log_avg_reward = running_reward / log_running_episodes
            log_avg_reward = round(log_avg_reward, 4)
            log_f = open(self.log_file, "w+")
            log_f.write('{},{}\n'.format(time_step, log_avg_reward))
        log_f.flush()

    def print_info(self, print_running_reward, print_running_episodes, time_step):
        """
        Print training info.
        """
        print("============================================================================================")
        print("Timestep : {}".format(time_step))
        for team, running_reward in print_running_reward.items():
            print_avg_reward = round(running_reward / print_running_episodes, 3)
            print("Average Reward {}: {}".format(team, print_avg_reward))
        print("============================================================================================")