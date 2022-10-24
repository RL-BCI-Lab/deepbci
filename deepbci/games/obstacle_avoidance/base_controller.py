import time
import sys
import os
import math
from os.path import join
from abc import ABC, abstractmethod
from pdb import set_trace

import numpy as np
import pandas as pd
import pygame
import yaml
import cv2


import deepbci.utils.utils as utils
import deepbci.utils.logger as logger
from deepbci.utils.compress import zstd_compress
from deepbci.utils.utils import bordered, get_timestamp, timeme
from deepbci.games.obstacle_avoidance.core_mechanics import CoreMechanics as OACoreMechanics

def print_state(image, save_loc):
     cv2.imwrite(save_loc, np.squeeze(image))
            
class BaseController(ABC, OACoreMechanics):
    """
        This class inherits from OA and acts as OA's brains. The pygame
        loop, needed to make the game run, is done here in conjunction with help
        code.

        Methods:
            run(): Used to loop step() function in OA which progresses the game
                by a single frame every loop.

            wait(): Waits for the user to click the screen in order to start the game.

            time_to_start(): Waits a set duration of time before starting the 
                game, usually comes after wait().

            print_fps_time(): Prints time realted information debugging and 
                tracking information, mainly for debugging.

            write_state_info(): Writes all collected game data to their
                respective files.
    """
    def __init__(self, paths, duration, 
                 logging=False, 
                 before_start=1,
                 recording_interval=100,
                 interval=1,
                 **kwargs):
        """
            Args:
                paths (dict): Important file paths for saving game data.

                duration (float): The approximate duration of the game. Interval
                    is added to duration to include recording of last state.

                logging (bool): If true then converts all print statements to
                    be logged in a log file. If false, prints to commandline.
                    
                before_start (float): Determines the number of seconds to wait
                    before the game starts.
                    
                s (ndarray): The image (state) at the begining of a recording 
                    interval.
                    
                a (int): Report action player made, no mouse click (0) or 
                    mouse click (1) over a given number of frames. Reset when 
                    state is recorded.
                
                states (list): Stores a list of targets and actions (old game state
                    info).

                event_times (list): Tracks timestamps of all collisions (ms).

                state_times (list): Tracks timestamps at which a state was captured (ms).

                game_info (list): Tracks game info required for syncing for EEG.

                save_images (bool): Determines if game state images should be saved.

                imgs (list): Stores state images captured during the game

                interval (float): Rate at which stepper increases (resolution of 
                    ime for state capturing).
                
                state_interval (int): Iterates discrete version game time according to 
                    interval, used as a target for when to record game states.
                
        """
        super().__init__(**kwargs)

        self.paths = paths
        self.before_start = before_start
        self.logging = logging
        self.clock = pygame.time.Clock()
        self.life_offset = 0
        self.max_life = 0
        self.min_life = float("inf")
        self.avg_life = 0
        self.sum_lives = 0
        self.lives_reported = 0
        self.s = None
        self.state_info = [] # tracks collisions and actions
        self.collision_timestamps = [] # tracks timestamps of all collisions in milliseconds
        self.state_timestamps = [] # tracks timestamps at which each state was captured
        self.game_info = [] # tracks game info required for syncing for EEG
        self.states = [] # game state images captured during the game
        self.recording_interval = (self.fps // (1000 // recording_interval))
        self.state_interval = 0
        self.duration = duration 
        self.font = pygame.font.SysFont('arial', 15*self.scale, bold=True)
        logger.configure(join(paths['subject_trial_dir'], "log"), ['log'])
        self._log_parameters()
        
        self.dt = 0
        self.log("Recording interval: {} Duration: {}".format(self.recording_interval, 
                                                              self.duration))
    @abstractmethod  
    def k_steps(self):
        """ Iterates game k frames while logging and tracking any in-game info.

            This method should run any code that is dependent on a frame-by-
            frame basis (called every frame).

            Returns:
                Action and reward for next state after k frames have passed.
        """
        return action, reward
    
    def _log_parameters(self):
        config = {}
        config['Parameters'] = self.__dict__
        utils.log_config(config=config, logger=logger)
        
    def run(self):
        """ Loop through the game until time constraint is reached."""
        record_stats_on_finish = False
        self.wait_to_start()
        self.countdown_to_start()
        self.start_time = self.get_time()
        self.life_start = self.start_time
        self.dt = self.clock.tick(self.fps)
        self.draw(self.immune)
        
        while True:
            if (self.state_interval < self.duration
                or self.state_interval >= self.duration and self.frozen):
                
                state = self.state
                state_timestamp = self.get_time() - self.start_time
                action, reward = self.k_steps()
                self.record_info(state=state,
                                 state_timestamp=state_timestamp,
                                 action=action,
                                 reward=reward)
                self.state_interval += 1
            else:
                if not record_stats_on_finish:
                    task_duration = self.get_time() - self.start_time
                    ts, ms = get_timestamp()
                    self.game_info.append(ms) 
                    self.game_info.insert(0, task_duration) 
                    # Track ending life length if player didn't freeze 
                    # at the very end of the task (freezing automatically reports
                    # life length so no need to).
                    if not self.immune:
                        self.track_life_length()          
                    record_stats_on_finish = True
                    self.log("\n"+bordered("End ts: {} / {}".format(ts, ms)))
                    self.log("\n"+bordered("Seed: {}".format(self.seed)))
                    
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        logger.add_stdout()
                        self.write_state_info()
                        # Log stats to be manually recorded 
                        lives_msg = "Avg: {:.2f} Max: {} Min: {} Lives: {}"
                        stats_msg = "Duration: {:.3f} Collisions: {}"
                        lives_msg = lives_msg.format(self.avg_life, 
                                           self.max_life,
                                           self.min_life,
                                           self.lives_reported)
                        stats_msg = stats_msg.format(task_duration, self.collision_count)
                        self.log("\n"+bordered("{}\n{}".format(lives_msg, stats_msg)))
                        pygame.quit()
                        sys.exit()
                pygame.time.delay(25)
            pygame.display.update()
            
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
        
    def pygame_dt(self):
        self.dt = self.clock.tick(self.fps)
        time_elapsed = self.get_time() - self.start_time
        pygame.display.set_caption("%d %d %.1f" % (self.clock.get_fps(), 
                                                   self.state_interval,
                                                   time_elapsed))
           
        # Detect artificial pauses which are assumed to be over a given threshold
        if self.dt > 100:
            lag_msg = "Frame Lag - ts:{:.3f} frame:{} dt:{}"
            self.log(lag_msg.format(time_elapsed, self.frame, self.dt))
            self.life_offset += self.dt/1000
            if self.immune and not self.frozen:
                self.immune_start += self.dt/1000
            self.dt = 1000//self.fps
    
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
             
    def record_info(self, state, state_timestamp, action, reward):
        """ Captures state information, i.e. target, images, and actions."""
        self.states.append(state)
        self.state_timestamps.append(state_timestamp)
        self.state_info.append([reward, action])
        
        msg = "{} {:.5f} a={} r={}"
        self.log(msg.format(self.state_interval, 
                            self.get_time() - self.start_time, 
                            action, 
                            reward))

    def track_life_length(self):
        """ Track duration between collisions.
        
            To keep life legnth accurate the artificial offset caused by delaying 
            frames when saving models or when frames drop must be 
            tracked and subtracted.
        """
        life = round(time.monotonic() - self.life_start - self.life_offset, 2)
        if life < self.immune_duration: life = self.immune_duration
        length_msg = "Life length: {} {:.2f}".format(life, life + self.life_offset)
        
        # Reset life offset
        self.life_offset = 0

        # Track min and max life lengths
        if self.max_life < life:
            self.max_life = life
        if self.min_life > life and life > self.immune_duration:
            self.min_life = life

        # Calculate average life span
        self.sum_lives += life
        self.lives_reported += 1 # varies slightly from collision_count
        self.avg_life =  self.sum_lives / self.lives_reported 
        
        # Log life stats
        stats_msg = "Avg: {:.2f} Max: {} Min: {} Lives: {}"
        stats_msg = stats_msg.format(self.avg_life, 
                                     self.max_life, 
                                     self.min_life, 
                                     self.lives_reported)
        self.log("{} {}".format(length_msg, stats_msg))
    
    @timeme
    def write_state_info(self):
        """ Save all state information recorded during game"""
        def extract_initial_collision_reward(labels):
            """
                Create reward by only reporting first collision state label.
                
                Groups labels and only uses first label which is when the player
                first collides.
            """
            new_labels = np.zeros(labels.shape, dtype=int)
            
            label_loc = np.where(labels == 1)[0]
            
            if len(label_loc) != 0:
                group_split_condi = np.where(np.diff(label_loc) != 1)[0]+1
                groups = np.split(label_loc, group_split_condi)
                
                first_locs = [g[0] for g in groups]
                other_locs = np.hstack([g[1:] for g in groups])
                
                new_labels[first_locs] = 1
                
                mask = labels == new_labels
                mask_false_locs = np.where(mask == False)[0]
                assert (mask_false_locs == other_locs).all()
            
            return new_labels 
        
        # Build state info
        state_info = np.vstack(self.state_info)
        state_info_df = pd.DataFrame(state_info, columns=['labels', 'actions'])
        
        state_ts_df = pd.DataFrame(self.state_timestamps, columns=['timestamps'])
  
        rewards = extract_initial_collision_reward(labels=state_info_df.labels.values)
        state_reward_df = pd.DataFrame(rewards, columns=['rewards'])
    
        to_concat = [
            state_ts_df, 
            state_info_df.labels, 
            state_reward_df, 
            state_info_df.actions
        ]
        state_info_df = pd.concat(to_concat, axis=1)
        state_info_df.to_csv(path_or_buf=self.paths["state_file"], index=False)

        # Record game info
        game_info_df = pd.DataFrame(np.vstack(self.game_info))
        game_info_df.to_csv(path_or_buf=self.paths["game_info_file"], 
                            header=None, 
                            index=False, 
                            float_format='%10f')

        # Record new state tracker information which records collision times
        try:
            collision_timestamps_df = pd.DataFrame(np.vstack(self.collision_timestamps))
            collision_timestamps_df.to_csv(path_or_buf=self.paths["timing_file"], 
                                           header=None, 
                                           index=False, 
                                           float_format='%10f')
        except ValueError:
            self.log("No collisions occurred but an empty file has been created")
            open(self.paths["timing_file"], 'a').close()

        # Record game state images as .npy
        states = np.stack(self.states, axis=0)
        np.save(self.paths["state_imgs_file"], states)
        try:
            zstd_compress(file_path=self.paths["state_imgs_file"], clean=True)
        except Exception:
            print("\nWARNING: Compression Failed...\n")
        
        print("States: {}".format(len(states)))
            
    def print_fps_time(self):
        """ Debug game timings (look for frame drops and measure performance)"""
        self.log('-'*10)
        self.log(self.frame/self.fps, self.frame, self.clock.get_fps())
        self.log('-'*10)