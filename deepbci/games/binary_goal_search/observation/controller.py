from pdb import set_trace

import pygame
import yaml
import numpy as np

from deepbci.games.binary_goal_search.base_controller import BaseController as BGSBaseController

class Controller(BGSBaseController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def get_action(self):
        return self.correct_action()
