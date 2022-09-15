import sys
from pdb import set_trace

import pygame
import yaml
import numpy as np

from deepbci.games.binary_goal_search.base_controller import BaseController as BGSBaseController

class Controller(BGSBaseController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_action(self):
        waiting = True
        action_input_msg = "X"
        render_action_msg = self.font.render(action_input_msg, False, (0, 0, 0))
        self.draw_text(text=render_action_msg,
                       position=(self.screen_width/2, self.screen_height/3),
                       update_all=False)
        pygame.event.clear(eventtype=pygame.KEYDOWN)
        while waiting:
            pygame.time.delay(25)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a:
                        return self.left_action
                    elif event.key == pygame.K_d:
                        return self.right_action

