import sys
import time
import datetime
import random
import os
from pdb import set_trace
from math import floor

import pygame
import numpy as np
import cv2 # pip install opencv-python

import deepbci.utils.logger as logger
from deepbci.games.pygame_objects import GameScreen, Block

"""
About: This game is used to detect the ErrP signal related to errors.
The blue agent moves towards any of 2 possible goal nodes that are randomly generated
on the left or right side of the agent. The agent will move towards which ever goal spawns
with a chance to move in the wrong direction.

Mechanics: At time step 0 the agent will wait the minimum delay time before moving
making a correct or incorrect step (towards or away from the goal). All steps
are offset by a random whole interval between min_delay and max_delay. The correct
or incorrect step is probabilistic and determined by the erro_rate.
When the agent reaches the goal there will be another random whole interval
delay before the game resets and begins again. Once a max number of trials
has been reached the game will end.

Mapping: Target 0 (timestep 0) maps to the first 1s of EEG drop_data,
target 1 maps to next 1s of EEG data and so on.
The goal is to capture the signal AFTER the agent makes a move.
"""


class Agent(Block):
    """Agent block that requires moving abilities."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def move(self, step):
        """
            Moves the block to a new x coordinate.
            
            Args:
                step (int): Step size to move block
        """
        self.rect.centerx += step

class CoreMechanics(GameScreen):
    def __init__(self, 
                 screen_width=420, 
                 screen_height=420, 
                 fps=1, 
                 agent_width=50, 
                 agent_height=50, 
                 goal_width=50, 
                 goal_height=50, 
                 goal_distance=150, 
                 no_action=0, 
                 left_action=0, 
                 right_action=1, 
                 base_reward=0, 
                 incorrect_reward=-1, 
                 correct_reward=1, 
                 step_size=50, 
                 error_rate=.2, 
                 delay=[1.3, 1.3], 
                 jitter=[.01, .1], 
                 state_width=84, 
                 state_height=84, 
                 scale=1):

        self.screen_height = screen_height * scale
        self.screen_width = screen_width * scale
        self.agent_width = agent_width * scale
        self.agent_height = agent_height * scale
        self.goal_width = goal_width * scale
        self.goal_height = goal_height * scale
        self.goal_distance = goal_distance * scale
        self.step_size = step_size * scale
        self.error_rate = error_rate
        self.delay = delay
        self.jitter = jitter
        self.no_action = no_action
        self.left_action = left_action
        self.right_action = right_action
        self.base_reward = base_reward
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward
        self.state_width = state_width
        self.state_height = state_height
        self.scale = scale
        
        self.__action = no_action
        self.__reward = base_reward
        self.action_timestamp = 0
        
        super().__init__(self.screen_height, self.screen_width)
        # self.start = self.screen_width // 2
    def get_time(self):
        """ Gets current time stamp.
            
            Add any time stamp modifications here, which will cause the rest of
            the base game use them.
        """
        #ts = datetime.datetime.now().time()
        #print(ts, ts.hour, ts.minute , ts.second ,ts.microsecond/1000000)
        #return float(ts.hour*3600 + ts.minute*60 + ts.second +ts.microsecond/1000000)
        return time.monotonic()
        
    def draw(self):
        # Draw background
        self.screen.fill((255,255,255))

        # Draw goal
        self.goal.draw((0,255,0))

        # Draw Agent
        self.agent.draw((0,0,255))
        
        # Update pygames screen
        pygame.display.update()

        
    def get_delay(self):
        delay = self.delay[random.randint(0,1)]
        jitter = np.round(random.uniform(self.jitter[0], self.jitter[1]), 2)
        
        msg = "\tDelay:{:.5f} Jitter:{:.5f} Total:{:.5f}"
        logger.log(msg.format(delay, jitter, delay+jitter))
        
        return delay + jitter
    
    def initialize_game(self):
        """ Initializes the game objects."""
        # Calculate goal position
        random_chance = np.random.uniform()
        dist = self.goal_distance if random_chance < .5 else -self.goal_distance
        
        self.agent = Agent(screen=self.screen, width=self.agent_width,
                           height=self.agent_height, x_offset=0)
        self.goal = Block(screen=self.screen, width=self.goal_width,
                          height=self.goal_height, x_offset=dist)
        self.draw()

        delay = self.get_delay()
        pygame.time.delay(int(delay*1000))
        
    def event_handling(self):
        """ Handles all events for pygames and manually constructed events."""
        for event in pygame.event.get():
            # Allows game to end correctly
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def correct_action(self):
        """ Returns which action to take if you want to move towards the goal"""
        if self.goal.coord()[0] < self.agent.coord()[0]:
            return self.left_action
        elif self.goal.coord()[0] > self.agent.coord()[0]:
            return self.right_action

    def incorrect_action(self):
        """ Returns opposite of the input action which is assumed to be incorrect """
        if self.goal.coord()[0] < self.agent.coord()[0]:
            return self.right_action
        elif self.goal.coord()[0] > self.agent.coord()[0]:
            return self.left_action
        
    def move_agent(self, action):
        random_chance = np.random.uniform()
        correct_action = self.correct_action()
        incorrect_action = self.incorrect_action()
        
        if abs(self.goal.coord()[0] - self.agent.coord()[0]) >= self.goal_distance*2:
            action = correct_action
        elif random_chance < self.error_rate and action != incorrect_action:
            action = incorrect_action
            
        if action == self.left_action:
            self.agent.move(-self.step_size)
        elif action == self.right_action:
            self.agent.move(self.step_size)
            
        if action == correct_action:
            reward = self.correct_reward
        elif action == incorrect_action:
            reward = self.incorrect_reward
        
        return reward, action

    def step(self, action, dt=0):
        """ Main loop of OA game.

            Acts as the main loop of the game that drives all game mechanics 
            and functions. Any changes to how base game mechanics are executed 
            should be made here.

            Note:
                DeltaT (dt) is artificial in this game since there are frequent
                sleeps after every move. This means FPS is also artificial.

            Args:
                dt (float): This variable represents the time between frames in 
                seconds and is used for scaling continuous movement with FPS.

            Returns:
                Return true when the agent has reached the goal.
        """            
        # Move agent based on passed or validated action
        self.reward, self.action = self.move_agent(action)

        # Draw screen
        self.draw()
        self.action_timestamp = self.get_time()
        
        # Select delay time in milliseconds
        delay = self.get_delay()
        pygame.time.delay(int(delay*1000))
        
        # Handle events
        self.event_handling()

    def sleep(self, delay):
        """ Custom sleep function (simulates time.sleep())"""
        start = self.get_time()
        #print(delay, start)
        while self.get_time()-start < delay:
            #print(self.get_time(), self.get_time()-start, delay)
            self.event_handling()
        #print("Slept for {}".format(self.get_time()-start))

    @property
    def state(self):
        screen = self.get_screen()
        screen = pygame.transform.scale(
            screen, (self.state_width, self.state_height))
        state = pygame.surfarray.array3d(screen).astype(np.uint8)

        return np.transpose(state, (1, 0, 2))

    @property
    def reward(self):
        return self.__reward
    
    @reward.setter
    def reward(self, value):
        self.__reward = value
        
    @property
    def action(self):
        return self.__action
    
    @action.setter
    def action(self, value):
        self.__action = value
        
    @property
    def terminal(self):
        if self.agent.centers_aligned(self.goal.rect):
            return 1
        return 0