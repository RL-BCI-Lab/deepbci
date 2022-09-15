import random
import time
import sys
import os
from pdb import set_trace
from collections import deque
from os.path import join, dirname

import numpy as np
import tensorflow as tf

import utils.logger as logger  
import utils.utils as utils
from deepbci.networks.dqn.controller import DQNController, RLLosses

class ExpertDQNController(DQNController):
    def __init__(self, expert_replay_weight=.5, **kwargs):
        super().__init__(**kwargs)
        self.expert_replay_weight = expert_replay_weight
        self.expert_replay = []
        self.expert_rewards = []
        np.random.seed(0)
    
    def get_batch_indices(self,):
        if len(self.replay) != 0:
            rand = np.random.uniform(low=0, high=1)
            if rand < self.expert_replay_weight:
                return np.random.randint(self.n-1, len(self.expert_replay), self.batch_size)
            else:
                return np.random.randint(self.n-1, len(self.replay), self.batch_size)
        else:
            return np.random.randint(self.n-1, len(self.expert_replay), self.batch_size)
        
    def get_batch(self, batch_indices=None):
        """ Gets minibatch from replay

            s = (t, t+k, t+2k, t+4k) or (t-3k, t-2k, t-k, t) where 
            k is abstracted by appendeding system of the replay buffer.
            This meaning that the replay buffer only appends every kth state.

            Returns:s
                np.array: Returns an array for each piece of the state information
                (s, a, r, t, s1), 5 arrays in total.
        """
        S, A, R, T, S_next = [], [], [], [], []
        if batch_indices is None:
            batch_indices = self.get_batch_indices()
        # Loop through random main states to get supporting states.
        for idx in batch_indices:
            S_tmp, S_next_tmp = [], []
            a_tmp, r_tmp, t_tmp = 0, 0, 0
            
            # Get main state (index - 1) info and supporting state images.           
            # Faster than itertools as long as self.n < 170
            # Test: https://stackoverflow.com/questions/10003143/how-to-slice-a-deque
            for i in reversed(range(self.n)):
                s, a, r, t, s_next = self.expert_replay[idx-i]
                S_tmp.append(s)
                S_next_tmp.append(s_next)
                if (idx - i) == idx: 
                    a_tmp, r_tmp, t_tmp = a, r, t
                    if self.loss_type == 'monte_carlo':
                        r_tmp = self._expected_return(step=idx)
            # self.print_state(tmp_s)
            # Append each piece of state information to a minibatch
            S.append(np.stack(S_tmp, axis=-1)) # (batch_size, 84, 84, n)
            A.append(a_tmp) # (batch_size,)
            R.append(r_tmp) # (batch_size,)
            T.append(t_tmp) # (batch_size,)
            S_next.append(np.stack(S_next_tmp, axis=-1)) # (batch_size, 84, 84, n)

        return (np.stack(S, axis=0), np.hstack(A), np.hstack(R),
            np.hstack(T), np.stack(S_next, axis=0))
        
    def _expected_return(self, step):
        nearest_terminal_state_loc = np.where(self.terminal_states - step >= 0)[0][0]
        nearest_terminal_state = self.terminal_states[nearest_terminal_state_loc]
        shape = (nearest_terminal_state - step) + 1
        gamma = np.full((shape), self.gamma)
        power = np.arange(0, shape)
        discount = np.power(self.gamma, power)
        return np.sum(discount * self.expert_rewards[step:nearest_terminal_state+1])
    
    def get_state_stack(self):
        return np.stack(self.state_stack, axis=-1)[np.newaxis,...]