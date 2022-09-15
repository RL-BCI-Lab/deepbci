import time
import sys
import os
import math
from os.path import join
from pdb import set_trace

import numpy as np
import pandas as pd
import pygame
import yaml
import cv2

import deepbci.utils.logger as logger
from deepbci.utils.compress import zstd_compress
from deepbci.utils.utils import bordered, get_timestamp, timeme
from deepbci.games.obstacle_avoidance.base_controller import BaseController as OABaseController

def print_state(image, save_loc):
     cv2.imwrite(save_loc, np.squeeze(image))
            
class Controller(OABaseController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def k_steps(self):
        skipped_actions = []

        for i in range(self.recording_interval):
            self.pygame_dt()
            action = self.get_action()
            self.step(action=action, dt=self.dt/1000)
            skipped_actions.append(self.action)

            # Capture information about frozen state
            if self.collision_start_frame == self.frame:
                timestamp = self.collision_start - self.start_time
                self.track_life_length()
                self.collision_timestamps.append(timestamp)
                self.collision_count += 1
                self.collision_action = max(skipped_actions)
                self.log("ERN START {} {}".format(timestamp, self.frame))
            if self.collision_end_frame == self.frame:
                timestamp = self.get_time() - self.start_time
                self.life_start = self.get_time() # start new life
                skipped_actions = [self.action]
                self.log("ERN END {} {}".format(timestamp, self.frame))   

        action = max(skipped_actions) 
        reward = self.reward # reward of next state k-steps later

        return action, reward
    
    def get_action(self):
        for event in pygame.event.get():
            # Allows game to end correctly
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # Listens for mouse button
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                return 1
        return 0
    