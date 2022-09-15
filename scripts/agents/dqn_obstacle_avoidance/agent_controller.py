import time
import sys
import os
import yaml
from pdb import set_trace
from os.path import join, dirname
from datetime import datetime

import pygame
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 

import deepbci.utils.logger as logger
from deepbci.utils.utils import log_config
from deepbci.games.obstacle_avoidance.core_mechanics import CoreMechanics as OACoreMechanics
from deepbci.agents.dqn_controller import DQNController
from deepbci.agents.utils import rgb2gray

class AgentController(OACoreMechanics):
    def __init__(self,
                 duration, 
                 train=True, 
                 exp_name=None, 
                 save_frame=None, 
                 load_dir=None, 
                 logging=False,
                 working_dir=None,
                 graphics=True, 
                 **kwargs):
        # WARNING: Disabling graphics prevents pygames events from being used.
        if not graphics: 
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            
        super().__init__(**kwargs['oa'])
        
        self.duration = duration
        self.exp_name = join(*exp_name)
        self.save_frame = save_frame
        self._logging = logging
        self.train = train
        self.life_offset = 0
        self.max_life = 0
        self.min_life = float("inf")
        self.avg_life = 0
        self.sum_lives = 0
        self.lives_reported = 0
        self.trained_count = 0
        self.atype = None
        self.clock = pygame.time.Clock()
        if working_dir is None:
            self.working_dir = dirname(__file__)
        else:
            self.working_dir = working_dir
        self._generate_directories()

        # Initialize DQN controller
        date = datetime.now().strftime("%Y%m%d-%H%M%S")
        file_writer = join(self.graph_dir, date)
        self.dqn_controller = DQNController(file_writer=file_writer, **kwargs['dqn'])
        
        # Load model
        self.load_dir = load_dir
        if self.load_dir is not None:
            load_path = join(self.working_dir, "models", *load_dir)
            self._log("Loading model... {}".format(load_path))
            self.dqn_controller.load_checkpoint(load_path)

        # Log config
        if self._logging:
            config = {}
            config['AgentController'] = self.__dict__
            config['DQNController'] = self.dqn_controller.__dict__
            log_config(config, logger)
            
    def _generate_directories(self):
        if self.save_frame is not None:
            self.model_dir = join(self.working_dir, "models", self.exp_name)
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
                
        self.state_dir = join(self.working_dir, "states", self.exp_name)
        if not os.path.exists(self.state_dir):
            os.makedirs(self.state_dir)

        self.graph_dir = join(self.working_dir, "graphs", self.exp_name)
        if not os.path.exists(self.graph_dir):
            os.makedirs(self.graph_dir)

        # initialize logger
        if self._logging: 
            logger.configure(join(self.working_dir, 'log', self.exp_name), ['log'])
             
    def run(self):
        """ Ties together DQN and game stepping.

        Note:
            During training the agent will preform random actions until
            the frames reach a given number ( give by replay_start). For testing
            the agent will preform random actions until k*n frames have been
            reached.
        """
        while True:
            # Duration of training is based on frames seen
            if self.frame <= self.duration: 
                # Initialize the game.
                if self.frame == 0:
                    self.dt = self.clock.tick(self.fps)
                    self.start_time = self.get_time()
                    self.life_start = self.start_time
                    
                self.q_learning()

            else:
                # Track end life length
                self.track_life_length()

                # Log final training information
                self._log("Training duration in seconds: {:.2f}".format(
                    time.monotonic() - self.start_time))
                self._log("Collisions: {}".format(self.collision_count))
                
                if self.save_frame is not None:
                    self._checkpoint()
                    
                pygame.quit()
                sys.exit()
            
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
        
        # Take min reward if collision occurred and max if no collision is detected
        # This ensures that when the immune state is turned off the base reward 
        # is reported (assuming based reward is > immune reward).
        r = min(skipped_r) if min(skipped_r) == self.collision_reward else self.reward

        # Set terminal state if reward == -1
        t = 1 if r == self.collision_reward else 0
                
        # Get next state after action has been set
        s1 = self.get_state() # get next state

        # Only append to replay if you are training.
        if self.train: 
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
                    loss, _ = self.dqn_controller.train()
                    self._track_loss(loss)

                # Check if update target network interval reached.
                if self.frame%self.dqn_controller.C == 0:
                    self._log('Updating Target Network - Frame: {}'.format(self.frame))
                    self.dqn_controller.update_target_network()
            
            self._log_frame_info(skipped=(self.frame % self.dqn_controller.k) != 0)
            
            # Track each game steps reward. (We will take the smallest reward
            # at the kth frame as this marks a collision occurred during a 
            # skipped frame/state)
            reward_per_step.append(self.reward)

            # If collision occurs during a skipped frame report the negative
            # collision reward instead of the negative base reward.
            if self.collision_start_frame == self.frame:
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
    
    def pygame_dt(self):
        # Maintain at most a given FPS
        self.dt = self.clock.tick(self.fps)
        # Display the current FPS and the current stepper iteration 
        pygame.display.set_caption("%d" % self.clock.get_fps())

        # Check a threshold for artificial pauses (usually due to DQN). Since RL
        # algorithms are not running in separate thread thus can disrupt game.
        if self.dt > 100: 
            self._log('Frame Lag:{:.5f} {} {}'.format(
                time.monotonic()-self.start_time, self.frame, self.dt))
            self.life_offset += self.dt/1000
            if self.immune and not self.frozen:
                self.immune_start += self.dt/1000
            self.dt = 1000//self.fps
            
    def track_life_length(self):
        """ Track duration between terminal states.
        
            To keep life legnth accurate the artificial offset caused by delaying 
            frames when saving models or when training begins must be 
            tracked and subtracted.
        """
        life = round(time.monotonic() - self.life_start - self.life_offset, 2)
        if life < self.immune_duration: life = self.immune_duration
        length = "Life length: {} {:.2f}".format(life, life + self.life_offset)
        
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
        self._track_life_length(life=life)
        stats = "Avg: {:.2f} Max: {} Min: {} Lives: {}".format(
            self.avg_life, self.max_life, self.min_life, self.lives_reported)
        self._log("X {} {}".format(length, stats))
           
    def get_state(self):
        """Gets the current state image from the pygames screenself.

        Returns:
            np.array of unit8: grayscale and reduced image of pygames screen
        """
        return rgb2gray(self.state)
            
    def _log(self, *args, **kwarg):
        """ Logs information using mpi4py or simply prints to stdout

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        if self._logging: 
            logger.log(*args)
        else: 
            print(*args, *kwarg)  
                
    def _log_frame_info(self, skipped):
            skip = "-" if skipped else ">"

            # Log information every frame
            self._log("{} {:.3f} {} {} {} {} {} {} {} {}".format(
                skip,
                (self.get_time()-self.start_time),
                self.frame,
                self.dt,
                self.atype,
                self.action,
                self.reward,
                self.immune,
                len(self.dqn_controller.replay),
                self.trained_count))
                
    def _checkpoint(self):
        frame = "{:d}".format(self.frame)
        save_dir = join(self.model_dir, frame)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        save_path = join(save_dir, 'cp.ckpt')
        self.dqn_controller.checkpoint(save_path)

        # Log model save point information
        self._log("Saving Model - Frame: {} Collisions: {}: Path: {}  ".format(
            self.frame, self.collision_count,  save_path))     
        
    def _track_q_values(self, s):
        with self.dqn_controller.writer.as_default():
            q_values = self.dqn_controller.get_values(
                s=s, model=self.dqn_controller.dqn_model)
            q_target_values = self.dqn_controller.get_values(
                s=s, model=self.dqn_controller.dqn_target)
            
            tf.summary.scalar(name="Model: Q1 - Down", 
                              data=q_values[0, 0], 
                              step=self.frame)
            tf.summary.scalar(name="Model: Q2 - Up", 
                              data=q_values[0, 1], 
                              step=self.frame)
            tf.summary.scalar(name="Target: Q1 - Down", 
                              data=q_target_values[0, 0], 
                              step=self.frame)
            tf.summary.scalar(name="Target: Q2 - Up", 
                              data=q_target_values[0, 1], 
                              step=self.frame)
            
    def _track_life_length(self, life):
        with self.dqn_controller.writer.as_default():
            tf.summary.scalar('Life Length', life, step=self.frame)

    def _track_reward(self, r):
        with self.dqn_controller.writer.as_default():
            tf.summary.scalar(name='Reward', data=r, step=self.frame)

    def _track_loss(self, loss):
        with self.dqn_controller.writer.as_default():
            tf.summary.scalar(name='Loss', data=loss, step=self.frame)
            
    def _print_state(self, stack, frame, pre="state", ext="jpg"):
        for s in range(len(stack)):
            file_name = "{}-{}-{}.{}".format(pre, frame, s, ext) 
            save_loc = join(self.state_dir, file_name)
            cv2.imwrite(save_loc, np.squeeze(stack[s]))