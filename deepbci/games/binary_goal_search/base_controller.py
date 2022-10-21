import sys
import time
import random
import math
import os
from abc import ABC, abstractmethod
from os.path import join
from pdb import set_trace

import pygame
import yaml
import numpy as np
import pandas as pd

import deepbci.utils.utils as utils
import deepbci.utils.logger as logger
from deepbci.games.binary_goal_search.core_mechanics import CoreMechanics as BGSCoreMechanics
from deepbci.utils.compress import zstd_compress
from deepbci.utils.utils import bordered, get_timestamp, timeme

class BaseController(ABC, BGSCoreMechanics):
    def __init__(self, paths, duration, logging=False, before_start=1, **kwargs):
        super().__init__(**kwargs)
        self.paths = paths
        self.duration = duration
        self.before_start = before_start
        self.logging = logging
        self.frame = 1
        self.states = []
        self.state_info = []
        self.state_timestamps = []
        self.game_info = []
        self.font = pygame.font.SysFont('arial', 15*self.scale, bold=True)
        logger.configure(join(paths['subject_trial_dir'], "log"))
        self._log_parameters()
        
    def _log_parameters(self):
        config = {}
        config['Parameters'] = self.__dict__
        utils.log_config(config=config, logger=logger)

    def run(self):
        """ Loop through the game until duration constraint is exceeded."""
        epoch = 1
        record_stats_on_finish = False

        self.wait_to_start()
        self.countdown_to_start()
        self.start_time = self.get_time()
        self.log("EPOCH: ",  epoch)
        self._update_caption()
        # The first initialization is an unlabeled resting state essentially
        self.initialize_game()
 
        while True:
            if self.frame <= self.duration or not self.terminal:
                if self.terminal:
                    self.capture_state(state=self.state, 
                                       action=self.no_action, 
                                       label=self.rest_label, 
                                       terminal=self.terminal, 
                                       timestamp=self.get_time()-self.start_time)
               
                    epoch += 1
                    self.log("EPOCH: ",  epoch)
                    self.frame += 1
                    self._update_caption()
                    self.initialize_game()

                self.frame += 1
                self._update_caption()
                s = self.state
                action = self.get_action() 
                self.step(action=action)
                self.capture_state(state=s, 
                                   action=self.action, 
                                   label=self.label, 
                                   terminal=self.terminal, 
                                   timestamp=self.action_timestamp-self.start_time)

            else:               
                if not record_stats_on_finish:
                    total_duration = self.get_time() - self.start_time
                    ts, ms = get_timestamp()
                    # Captures last step of the task, i.e. terminal state
                    self.capture_state(state=self.state, 
                                       action=self.no_action, 
                                       label=self.rest_label, 
                                       terminal=self.terminal, 
                                       timestamp=self.get_time()-self.start_time)
                    self.game_info.append(ms)
                    self.game_info.insert(0, total_duration)
                    record_stats_on_finish = True
                    self.log("\n"+bordered("End ts: {} / {}".format(ts, ms)))
                
                game_over_msg = "Game Over!"
                render_game_over = self.font.render(game_over_msg, False, (0, 0, 0))
                self.draw_text(text=render_game_over, 
                               position=(self.screen_width/2, self.screen_height/2),
                               update_all=True)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.write_state_info()
                        self.print_game_info(epoch, total_duration)
                        pygame.quit()
                        sys.exit()
                pygame.time.delay(25)
                
    def wait_to_start(self):
        """ Wait for the user to click the screen to start the game."""
        waiting = True
        start_msg = "Click screen to start!"
        render_start_msg = self.font.render(start_msg, False, (0, 0, 0))
        self.draw_text(text=render_start_msg, 
                       position=(self.screen_width/2, self.screen_height/2),
                       update_all=True)
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    waiting = False
            pygame.time.delay(25)

        ts, ms = get_timestamp()
        self.log("\n"+bordered("Before start ts: {} / {}".format(ts, ms)))
        self.game_info.append(ms)

    def countdown_to_start(self):
        """ Begin countdown before game begins."""
        start = self.get_time()
        while (self.get_time()-start) < self.before_start:
            time_msg = str(int(self.get_time() - start)+1)
            render_count_down = self.font.render(time_msg, False, (0, 0, 0),)
            self.draw_text(text=render_count_down,
                           position=(self.screen_width/2, self.screen_height/2),
                           update_all=True)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            pygame.time.delay(25)

        ts, ms = get_timestamp()
        self.log("\n"+bordered("Start ts: {} / {}".format(ts, ms)))
        self.game_info.append(ms)
        
    def draw_text(self, text, position, update_all=False):
        """Draw text in the center of the screen

            Args:
                text (str): Text to be displayed on the screen
        """
        self.screen.fill((255, 255, 255))
        text_rect = text.get_rect(center=position)
        blitted_text = self.screen.blit(text, text_rect)
        
        if update_all:
            pygame.display.update()
        else:
            pygame.display.update(blitted_text) 
            
    def log(self, *args, **kwarg):
        """ Logs information using mpi4py or simply prints to stdout

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        if self.logging: 
            logger.log(*args)
        else: 
            print(*args, *kwarg)
            
    def _update_caption(self):
        time_elapsed = self.get_time() - self.start_time
        pygame.display.set_caption("%.1f  %d" % (time_elapsed, self.frame)) 
        
    @abstractmethod
    def get_action(self):
        pass
    
    def capture_state(self, state, action, label, terminal, timestamp):
        """ Captures state information, i.e. actions, images error or correct action."""
        self.states.append(state)
        self.state_info.append([label, action, terminal])
        self.state_timestamps.append(timestamp)
        
        msg = "\tCapturing - Time:{:.5f} L: {} A: {} T: {}"
        self.log(msg.format(timestamp, label, action, terminal))
        
    def print_game_info(self, epoch, duration):
        labels = np.vstack(self.state_info)[:, 0]
        err = len(labels[labels == self.incorrect_label])
        cor = len(labels[labels == self.correct_label])
        reset = len(labels[labels == self.rest_label])
        
        # Check information was recorded correctly
        if epoch != reset:
            error_msg = "Epoch and resets mismatched: epoch: {} resets: {}"
            self.log(error_msg.format(epoch, reset))
        if len(self.states) != (err+cor+reset):
            error_msg = "Image count mismatch state records: images: {} state: {}"
            self.log(error_msg.format(len(self.states), (err+cor+reset)))
        
        # Log final information
        msg = "Steps:{} Err:{} Cor:{} Reset:{} Imgs:{} Time:{:.5f} E-R:{}"
        state_info = msg.format((err + cor), err, cor, reset, 
                               len(self.states), duration, self.error_rate)
        self.log("\n"+bordered(text=state_info))

    @timeme
    def write_state_info(self):
        """Save all state information recorded during game

            Notes:
                old state format (shape: nxm):
                    error (0 or 1) | correct (0 or 1) | terminal state (0 or 1)
                
                game info format (shape: nx1):
                    game duration in seconds              |
                    start of count down for time_to_start |
                    start of game                         |
                    game duration in milliseconds         |

                CRN/ERN timing format (shape: nx1);
                    time erroneous/correct action was made |
        """
        state_ts_df = pd.DataFrame(self.state_timestamps, columns=['timestamps'])
        state_info = np.vstack(self.state_info)
        state_info_df = pd.DataFrame(state_info, columns=['labels', 'actions', 'terminal'])
        state_info_df = pd.concat([state_ts_df, state_info_df], axis=1)
        state_info_df.to_csv(path_or_buf=self.paths["state_file"], index=False)
        
        # Record game info
        game_info_df = pd.DataFrame(np.vstack(self.game_info))
        game_info_df.to_csv(path_or_buf=self.paths["game_info_file"], 
                            header=None,
                            index=False, 
                            float_format='%10f')

        ern_ts = state_ts_df[state_info[:, 0] == self.incorrect_label] 
        ern_ts_df = pd.DataFrame(ern_ts)
        ern_ts_df.to_csv(path_or_buf=self.paths["ern_timing_file"], 
                         header=None, 
                         index=False, 
                         float_format='%10f')
        
        crn_ts = state_ts_df[state_info[:, 0] == self.correct_label]
        crn_ts_df = pd.DataFrame(crn_ts)
        crn_ts_df.to_csv(path_or_buf=self.paths["crn_timing_file"], 
                         header=None,
                         index=False,
                         float_format='%10f')
        
        rest_ts = state_ts_df[state_info[:, 0] == self.rest_label]
        rest_ts_df = pd.DataFrame(rest_ts)
        rest_ts_df.to_csv(path_or_buf=self.paths["rest_timing_file"], 
                         header=None,
                         index=False,
                         float_format='%10f')
                       
        # Record game state images as .npy
        np.save(self.paths["state_imgs_file"], np.stack(self.states, axis=0))
        try:
            zstd_compress(file_path=self.paths["state_imgs_file"], clean=True)
        except Exception:
            print("\nWARNING: Compression Failed...\n") 
