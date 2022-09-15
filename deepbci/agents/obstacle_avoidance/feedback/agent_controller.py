import random
import copy
import time
import sys
import os
import yaml

from pdb import set_trace
from queue import Queue
from collections import deque
from os.path import join, dirname, exists

import pygame
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 

import deepbci.utils.logger as logger  
import deepbci.utils.utils as utils
import deepbci.utils.loading as load
import deepbci.networks.dqn.agents.oa.baseline.agent_controller as baseline
from deepbci.data.scripts.npy2img.npy2img_memory import npy2img_memory
from deepbci.utils.utils import rgb2gray
from deepbci.networks.dqn.expert_controller import ExpertDQNController

utils.tf_allow_growth()

class AgentController(baseline.AgentController):
    def __init__(self, state_loading=None, **kwargs):
        super().__init__(workingdir=dirname(__file__), **kwargs)
        self.sl = state_loading
     
        # State loading variables
        if self.sl is not None:
            self.remove_frozen =  self.sl.get('remove_frozen', {})
            self.task = self.sl['task']
            self.subject = self.sl['subject']
            self.trials = utils.parse_trials(self.sl['trials'])
            # self.trial_lens = np.zeros(len(self.trials), dtype=int)
            self.npy_file = join(*self.sl['npy_file'])
            self.npy_clean = self.sl['npy_clean']
            self.state_file = self.sl['state_file']
            self.load_state_names = self.sl['load_state_names']
            self.use_state_names = self.sl['use_state_names']
            self.rtype = self.remove_frozen.get('rtype')

            # Error Checks
            if not all(n in self.load_state_names for n in self.use_state_names):
                raise ValueError("Found columns in use_state_names that are " \
                                 "not in load_state_names")
            if len(self.use_state_names) > 2:
                raise ValueError("Passed more than two state name")
            if self.use_state_names[0] != 'actions':
                msg = "The label 'actions' was not detected in the first column " \
                      "of use_state_names"
                raise ValueError(msg)

        # Overwrite baseline DQN controller with an extpert controller
        self.dqn_controller = ExpertDQNController(file_writer=self.dqn_controller.file_writer, 
                                                  **kwargs['dqn'])

        if self.load_dir is not None:
            load_path = join(self.workingdir, "models", *self.load_dir)
            self._log("Loading model... {}".format(load_path))
            self.dqn_controller.load_checkpoint(load_path)
        if self.train:
            self._load_state_tuples()
            expert_replay_len = len(self.dqn_controller.expert_replay)
            formatted_msg = utils.bordered("Expert length: {}".format(expert_replay_len))
            self._log(formatted_msg)

    @utils.timeme
    def _load_state_tuples(self):

        # Setup data folder paths (i.e. path to task/subject/trial)
        subject_path = load.path_to( 
                root_dir=os.getcwd(), target=join(self.task, self.subject))
        trial_paths = load.get_trial_paths(
                dir_path=subject_path, trial_numbers=self.trials)

        # Load state images
        # TODO: Covert to dqn-state-images if not detected
        image_paths = []
        for i, t in enumerate(trial_paths):
            image_paths.append(join(t, self.npy_file))
        states = npy2img_memory(file_paths=image_paths, clean=self.npy_clean)

        # Load targets and actions
        state_info = load.load_state_info(folders=trial_paths, 
                                          state_file=self.state_file, 
                                          load_headers=self.load_state_names)

        state_targets = []
        for i, info in enumerate(state_info):
            state_targets.append(info['targets'].values)
            state_info[i] = info[self.use_state_names].values

        if self.remove_frozen:
            for trial, (S, Y) in enumerate(zip(states, state_targets)):
                expected_duplicates = np.where(Y==1)[0]
                expected_duplicates = utils.group_consecutive(expected_duplicates)
                delete_states = []
                for i, f in enumerate(expected_duplicates):
                    # Get unique states where freeze occurred 
                    # _, indices = np.unique(S[f], axis=0, return_index=True)
                    detected_duplicates = np.setdiff1d(f, f[0])
                    delete_states.append(detected_duplicates)
                    # Set target to take max target of all expected_duplicates states.
                    if self.rtype == 'max':
                        state_info[trial][f, 1] = np.max(state_info[trial][f, 1])

                # Delete duplicates states
                states[trial] = np.delete(states[trial], np.hstack(delete_states), axis=0)
                state_info[trial] = np.delete(state_info[trial], np.hstack(delete_states), axis=0)
                # state_info[trial][np.hstack(delete_states), 1] = 0
                assert len(np.where(state_info[trial][:, 1]==1)[0]) == len(expected_duplicates)

        # combine all Q-learning state tuple information
        for trial, (S, info) in enumerate(zip(states, state_info)):
            for t, (s, a, r) in enumerate(zip(S, info[:, 0], info[:, 1])):
                r = r*-1
                if self._terminal_based_on_last_state_condition(t, S):
                    continue
                
                terminal, terminal_loc = self._get_terminal_based_on_reward(r)
                if terminal_loc is not None:
                    self.dqn_controller.terminal_states.append(terminal_loc)
                    
                self.dqn_controller.expert_replay.append((s, a, r, terminal, S[t+1]))
                self.dqn_controller.expert_rewards.append(r)
                # if self._terminal_based_on_last_state_condition(t+1, S):
                #     self._print_state([s], frame=1, pre='s')
                #     self._print_state([S[t+1]], frame=1, pre='s_next1')
                #     # self._print_state([S[t+2]], frame=1, pre='s_next2')
                #     # self._print_state([S[t+3]], frame=1, pre='s_next3')
                #     print("action: {} reward: {}".format(a, self.dqn_controller.expert_rewards[-1])) 

    def _terminal_based_on_reward_condition(self, reward):
        if reward == -1:
            return True
        
    def _get_terminal_based_on_reward(self, reward):
        terminal = 0
        terminal_loc = None
        if self._terminal_based_on_reward_condition(reward):
            terminal = 1
            terminal_loc = len(self.dqn_controller.expert_replay)
        return terminal, terminal_loc
            
    def _terminal_based_on_last_state_condition(self, step, trial_states):
         if step == len(trial_states)-1: 
            return True
        
    def _get_terminal_based_on_last_state(self, step, trial_states):
        terminal = 0
        terminal_loc = None
        if self._terminal_based_on_last_state_condition(step+1, trial_states):
            terminal = 1
            terminal_loc = len(self.dqn_controller.expert_replay)
        return terminal, terminal_loc
    
    def q_learning(self):
        """Runs the q-learning algorithm for RL.

        Note:
            Here we capture the state which has not yet been appended to the replay
            memory yet. This means that when we compile the current state with
            n history we take the most recent n-1 frames from the queue (e.g. if
            the queue is full then we concatenate the following: [s] + [t-k,t-1k,t-2k]).

        Args:
            act (bool); Determines if a new action should be selected (corresponds
            with k).
        """
        # Get the current state image and add it to stack.
        s = self.get_state()
        self.dqn_controller.state_stack.append(s)

        # Determine agent actions
        if self.frame >= self.dqn_controller.replay_start:
            s_stack = self.dqn_controller.get_state_stack() 
            a, atype = self.dqn_controller.epsilon_greedy(s=s_stack, frame=self.frame) 
            self.atype = 'off' if atype == 1 else 'random'
            self._track_q_values(s=s_stack)
        else:
            self.atype = 'random' 
            a = self.dqn_controller.random_action()

        # Iterate game by k steps (k frames)
        skipped_r = self.k_steps(action=a)

        # Take max reward during skipped frame
        r = min(skipped_r) if min(skipped_r) == self.collision_reward else self.reward

        # Set terminal state if reward == -1
        t = 1 if r == -1 else 0
                
        # Get next state after action has been set
        s1 = self.get_state() # get next state

        # Only append to replay if you are training.
        if not self.train: 
            self.dqn_controller.replay.append((s,a,r,t,s1)) 

        # Log information every kth state
        self._track_reward(r=r)
        self._log("S: {} S': {} Action: {} Reward: {} Epsilon: {:.5f}".format(
            (self.frame - self.dqn_controller.k), self.frame, 
            a, r, self.dqn_controller.epsilon))
        
    def k_steps(self, action):
        """ Iterates game k frames while logging and tracking any in-game info.

            This method should run any code that is dependent on a frame-by-
            frame basis (called every frame).

            Args:
                a (int): Optional action that can be passed to be used instead
                of no-action.
        """
        reward_per_step = []
        if self.frame == 0:
            self._log_frame_info(skipped=(self.frame % self.dqn_controller.k != 0))

        for _ in range(self.dqn_controller.k):
            self.pygame_dt()

            self.step(action=action, dt=self.dt/1000)
        
            if self.frame >= self.dqn_controller.replay_start and self.train:
                if self.frame % self.dqn_controller.train_interval == 0:
                    self.trained_count += 1
                    loss , _ = self.dqn_controller.train()
                    self._track_loss(loss)

                # Check if update target network interval reached.
                if self.frame % self.dqn_controller.C == 0:
                    self._log('Updating Target Network - Frame: {}'.format(self.frame))
                    self.dqn_controller.update_target_network()
            
            self._log_frame_info(skipped=(self.frame % self.dqn_controller.k) != 0)
            
            # Track each game steps reward. (We will take the smallest reward
            # at the kth frame as this marks a collision occurred during a 
            # skipped frame/state)
            reward_per_step.append(self.reward)

          # If collision occurs during a skipped frame report the negative
            # collision reward instead of the negative base reward.
            if self.collision_frame == self.frame:
                self.track_life_length()
                self.collision_count += 1
            if self.collision_end_frame == self.frame:
                self.life_start = self.get_time()

            if self.save_frame is not None:
                if self.frame % self.save_frame == 0 and self.frame != 0:
                    self._checkpoint()

            # Human level response restriction replaces repeated actions.
            # That is, instead of repeating an action the agent is restricted
            # to responding every 100ms by repeating in the no action instead of 
            # repeating the agent selected action at the kth step.
            if self.dqn_controller.action_to_repeat is not None:
                action = self.dqn_controller.action_to_repeat

        return reward_per_step


        
