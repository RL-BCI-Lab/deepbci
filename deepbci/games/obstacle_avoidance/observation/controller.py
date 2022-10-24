import os
from pdb import set_trace

import tensorflow as tf
import numpy as np
import pygame
import yaml
import cv2
from collections import deque

from deepbci.models.networks import DQN
from deepbci.games.obstacle_avoidance.base_controller import BaseController as OABaseController
from deepbci.utils import utils
from deepbci.agents.utils import rgb2gray

def print_state(image, save_loc):
     cv2.imwrite(save_loc, np.squeeze(image))
            
class Controller(OABaseController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ckpt_path = utils.path_to(utils.get_module_root(), 'games/obstacle_avoidance/observation/model')
        self.ckpt_path = os.path.join(ckpt_path, 'cp.ckpt')
        self.actions = [self.down_action, self.up_action]
        self.state_shape = [84, 84, 4]
        self.agent_history = deque(maxlen=4)
        
        self.model = DQN(actions=len(self.actions), name='dqn_model', init_seed=0)
        
        self.model.init_input(self.state_shape)
        self.model.load_weights( self.ckpt_path)
        
    def k_steps(self):
        skipped_actions = []
        action = self.get_action()
        for i in range(self.recording_interval):
            self.pygame_dt()
            self.step(action=action, dt=self.dt/1000)
            skipped_actions.append(self.action)

            if self.collision_start_frame == self.frame:
                timestamp = self.collision_start - self.start_time
                self.track_life_length()
                self.collision_timestamps.append(timestamp)
                self.collision_count += 1
                # print("collision at", i, self.frame)
                self.collision_action = max(skipped_actions)
                self.log("ERN START {} {}".format(timestamp, self.frame))
            if self.collision_end_frame == self.frame:
                timestamp = self.get_time() - self.start_time
                self.life_start = self.get_time() # start new life
                skipped_actions = [self.action]
                self.log("ERN END {} {}".format(timestamp, self.frame))   
            action = self.down_action # skip action

        action = max(skipped_actions) 
        reward = self.reward # Reward of next state k-steps later
        # print(self.state_interval, skipped_actions, max(skipped_actions), reward)
        return action, reward
    
    def get_action(self):
        self.agent_history.append(self.get_state())
        
        if len(self.agent_history) < self.agent_history.maxlen:
            return self.random_action()
        
        state = np.stack(self.agent_history, axis=-1)[np.newaxis,...]
        state = tf.divide(tf.convert_to_tensor(state, tf.float32), 255.0)
        action = tf.argmax(self.model(state), axis=1).numpy()[0]
        return action
    
    def get_state(self):
        """Gets the current state image from the pygames screenself.

        Returns:
            np.array of unit8: grayscale and reduced image of pygames screen
        """
        screen = self.get_screen()
        screen = pygame.transform.scale(screen, (84, 84))
        state = pygame.surfarray.array3d(screen).astype(np.uint8)
        state =  np.transpose(state, (1, 0, 2))
        return rgb2gray(state)
            
    def random_action(self):
        return np.random.choice(self.actions)