import random
import time
import sys
import os
from pdb import set_trace
from collections import deque
from os.path import join, dirname

import cv2
import numpy as np
import tensorflow as tf

import deepbci.utils.logger as logger  
import deepbci.utils.utils as utils
from deepbci.models import DQN
               
class DQNController(object):
    def __init__(self, actions, shape,
                 C=10000, 
                 k=6, 
                 n=4, 
                 r=3,  
                 replay_size=1e6,
                 replay_start=None,
                 action_to_repeat=None, 
                 train_interval=None, 
                 batch_size=32,
                 learning_rate=.25e-3, 
                 gamma=.99, 
                 epsilon=1, 
                 decay_epsilon=None,
                 loss_type='temporal_difference', 
                 file_writer=None,
                 init_seed=None,
                 **kwargs):
        
        utils.tf_allow_growth()
        
        self.actions = actions
        self.shape = shape
        self.C = C
        self.k = k
        self.n = n
        self.r = r
        self.batch_size = batch_size
        self.action_to_repeat = action_to_repeat
        self.replay_start = replay_start if replay_start is not None else n * k
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.init_epsilon = epsilon
        self.decay_epsilon = decay_epsilon
        if decay_epsilon is not None:
            # Compensat for random action time for final epsilon frame
            self.decay_epsilon[0] + self.replay_start
        self.file_writer = file_writer
        self.train_interval = k * r

        self.loss_type = loss_type
        self.loss = None
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        
        self.terminal_states = []
        self.replay = deque(maxlen=replay_size)
        self.state_stack = deque(maxlen=self.n)
        self.dqn_model = DQN(actions=actions, name='dqn_model', init_seed=init_seed)
        self.dqn_target = DQN(actions=actions, name='dqn_target', init_seed=init_seed)
        self.dqn_model.init_input(shape=shape)
        self.dqn_target.init_input(shape=shape)
        self.update_target_network()

        self.writer = tf.summary.create_file_writer(file_writer)
        self.capture_graph = False
        
    def train(self, batch_indices=None):

        S, A, R, T, S_next = self.get_batch(batch_indices)
        S = tf.divide(tf.convert_to_tensor(S, dtype=tf.float32), 255)
        A = tf.convert_to_tensor(A, dtype=tf.uint8)
        R = tf.convert_to_tensor(R, dtype=tf.float32)
        T = tf.convert_to_tensor(T, dtype=tf.float32)
        S_next = tf.divide(tf.convert_to_tensor(S_next, dtype=tf.float32), 255)
        
        # Save computational graph for tensorboard
        if not self.capture_graph: 
            tf.summary.trace_on(graph=True, profiler=True)

        # Apply DQN forward pass
        loss, grads = self.forwards(S, A, R, T, S_next)
        
        # Save computational graph for tensorboard one time only
        if not self.capture_graph:
            self.capture_graph = True
            with self.writer.as_default():
                tf.summary.trace_export(name='graph', 
                                        step=0, 
                                        profiler_outdir=self.file_writer) 
        
        return loss, grads
        
    def get_batch_indices(self):
        """ Get randomly selected indices from replay buffer for training

            Returns:
                Returns a ndarray of size equal to passed batch size
        """
        return np.random.randint(self.n-1, len(self.replay), size=self.batch_size)
        
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
                s, a, r, t, s_next = self.replay[idx-i]
                S_tmp.append(s)
                S_next_tmp.append(s_next)
                # Extract main state a, r, t
                if (idx - i) == idx: # i.e. (idx - 0) == idx
                    a_tmp, r_tmp, t_tmp = a, r, t
                    if self.loss_type == 'monte_carlo':
                        r_tmp = self._expected_return(step=idx)
            
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
        nearest_terminal_state = self.terminal_states[nearest_terminal_state_loc] + 1
        shape = nearest_terminal_state - step
        gamma = np.full((shape), self.gamma)
        power = np.arange(0, shape)
        discount = np.power(self.gamma, power)

        return np.sum(discount * self.replay[step:nearest_terminal_state])
    
    @tf.function
    def forwards(self, S, A, R, T, S_next):
        with tf.GradientTape() as tape:
            Qs = self.dqn_model(S)
            Q = tf.reduce_sum(Qs * tf.one_hot(A, self.actions), axis=1)
            Qs_target = self.dqn_target(S_next)
            Q_target = tf.reduce_max(Qs_target, axis=1)
            loss = self._get_loss(R, T, Q, Q_target)

        dqn_grads = tape.gradient(loss, self.dqn_model.trainable_variables)
        self.optimizer.apply_gradients(zip(dqn_grads, self.dqn_model.trainable_variables))

        return loss, dqn_grads
    
    def _get_loss(self, R, T, Q, Q_target):
        if self.loss_type == 'temporal_difference':
            return DQNLosses.temporal_difference(Q=Q, 
                                                Q_target=Q_target, 
                                                reward=R, 
                                                gamma=self.gamma, 
                                                terminal=T)
        elif self.loss_type == 'monte_carlo':
            return  DQNLosses.monte_carlo(Q=Q, G=R)
        else:
            raise ValueError("Invalid loss type")
            
    def epsilon_greedy(self, s, frame):
        """ Decays epsilon and gets a greedy or random action.

        Args:
            s (np.array): Current state image.

        Returns:
            int: Returns an action
        """
        if self.decay_epsilon is not None: 
            self.epsilon_decay(
                frame, self.decay_epsilon[0], self.replay_start,
                self.decay_epsilon[1], self.init_epsilon)

        if np.random.rand() < self.epsilon:
            return self.random_action(), 0
        else:
            return self.get_action(s).numpy()[0], 1
       
    def epsilon_decay(self, t, max_step, min_step=0, final_e=.1, init_e=1.):
        """ Anneals epsilon to a specified value.

        Args:
            t (int): Current frame the game is one.
            max_step (int): The maximum number frame the game will have.
            min_step (int): The minimum number of frames the game will have.
            final_e (float): The final decayed value of epsilon.
            init_e (float): The initial value of epsilon.

        Returns:
            Float64: Returns the max between the current epsilon and final
            epsilon value.
        """
        num = ((max_step - min_step) * init_e 
              + (init_e - final_e) * min_step 
              - (init_e - final_e) * t)
        denom = max_step - min_step
        e_decay= num/denom
        self.epsilon = max(e_decay, final_e)
        
    def random_action(self):
        """ Gets a random action.

            Returns:
                Returns a random action from available_actions.
        """
        return np.random.randint(self.actions)

    def get_action(self, s):
        s = tf.divide(tf.convert_to_tensor(s, tf.float32), 255.0)
        
        return tf.argmax(self.dqn_model(s), axis=1)

    def get_values(self, s, model):
        s = tf.divide(tf.convert_to_tensor(s, tf.float32), 255.0)
        
        return model(s)
    
    def get_state_stack(self):
        return np.stack(self.state_stack, axis=-1)[np.newaxis,...]
    
    def update_target_network(self):
        self.dqn_target.set_weights(self.dqn_model.get_weights())
        
    def load_checkpoint(self, load_path):
        self.dqn_model.load_weights(load_path)
        self.dqn_target.load_weights(load_path)

    def checkpoint(self, save_path):
        self.dqn_model.save_weights(save_path)
        
class DQNLosses(object):
    @staticmethod
    def temporal_difference(Q, Q_target, reward, gamma, terminal):
        memory = tf.multiply(gamma, Q_target, name='decay') 
        terminal = tf.multiply(memory, (1-terminal), name='terminal')
        target = tf.add(reward, terminal, name='target')
        loss = tf.losses.Huber(reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
        
        return loss(target, Q)
    
    @staticmethod
    def monte_carlo(Q, G):
        loss = tf.losses.Huber(reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
        
        return loss(G, Q)
    
def print_state(stack, save_dir, ext="jpg", dim=0):
    for s in range(len(stack)):
        file_name = "frame_{}-s_.{}".format(s, ext) 
        save_loc = join(save_dir, file_name)
        cv2.imwrite(save_loc, np.squeeze(stack[s]))