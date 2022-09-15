import sys
import time
import random
import math
import os
from pdb import set_trace
from collections import deque

#from PIL import Image
import cv2
import numpy as np
import pandas as pd
import pygame

import deepbci.utils.logger as logger
from deepbci.games.pygame_objects import GameScreen, Circle

def euclidean_distance(p1, p2):
    dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return int(dist)

class Player(Circle):
    def __init__(self, gravity, fly, screen, radius, screen_width, screen_height):
        super().__init__(screen=screen,
                         radius=radius, 
                         screen_width=screen_width, 
                         screen_height=screen_height)
        self.gravity = gravity
        self.fly = fly

    def down(self, dt):
        self.y += math.floor(self.gravity * dt)

    def up(self, dt):
        self.y -= int(self.fly)

class Ball(Circle):
    def __init__(self, speed, screen, radius, screen_width, screen_height):
        super().__init__(screen=screen,
                         radius=radius, 
                         screen_width=screen_width, 
                         screen_height=screen_height)
        self.speed = speed
        self.spawn()

    def inside_screen(self):
        return 0 < self.x < self.screen_width and 0 < self.y < self.screen_height

    def spawn(self):
        self.y =  np.random.randint(low=0, high=self.screen_height-1)
        # self.y = random.randrange(0, self.screen_height-1)
        self.x = self.screen_width-1
        self.draw(color=(0,0,0))

    def move(self, dt):
        self.x -= int(self.speed * dt)

class CoreMechanics(GameScreen):
    """ Base game mechanics for the obstacle avoidance task.
    
        Game used in conjunction with EEG equipment to detect errors (collisions).
        The goal of the game is avoid obstacles by either clicking the mouse 
        (move up) or not clicking the mouse (move down). When a collision occurs
        the game is typically frozen for a short period of time (helpful for 
        allowing a cleaner ERN EEG signal to be captured). The goal of the EEG
        recording is to capture Error-related negativity which is a brain signal
        associated with erroneous actions.

        Attributes:
            screen_width (int): Width to set the pygame screen.

            screen_height (int): Height to set the pygame screen.
            
            player_size (int): Radius of the the player shape.
            
            player_down (int): Number of pixels the player moves down by.
            
            player_up (int): Number of pixels player moves up by.
            
            balls (list): Tracks all activate ball objects.
            
            ball_size (int): Radius of the ball shape.
            
            ball_speed (int): Number of pixels ball moves per frame.
            
            ball_limit (int): Maximum number of balls on screen.
            
            collide_top (int): Number of pixels from the top of the screen the 
            player needs to be to collide with the top boundary.
            
            collide_bottom (int): Number of pixels from the bottom of the screen
            the player needs to be to collide with the bottom boundary.
            
            collide_ball (int): Number pixels away from a ball the player needs
            to be to collide with the ball.

            collision_count (int): Number of times the player has collided with
            an obstacle.    
            
            immune (bool): Determines if player is currently immune to collisions.
            
            turn_on_immune (bool): Determines if immune state should be activated
            next frame.
            
            immune_duration (int): Length of the immune state.
            
            spawn_rate (int): Chance a ball has to spawn (higher the number the
            greater the chance).
            
            freeze (int): Duration the player freezes for upon collision (seconds).
    """
    def __init__(self, screen_width=420, screen_height=420, fps=60,player_size=7, 
                 player_down=70, player_up=19, ball_size=11, ball_speed=150, 
                 ball_limit=40, collide_top=10, collide_bottom=5, 
                 collide_ball=28, immune_duration=1, spawn_rate=80, freeze_duration=1, 
                 base_reward=.1, collision_reward=-1, state_width=84, 
                 state_height=84, scale=1, seed=None):
        super().__init__(width=screen_width*scale, 
                         height=screen_height*scale)
        self.fps = fps
        self.screen_width = screen_width * scale
        self.screen_height = screen_height * scale
        self.player_size = player_size * scale
        self.player_down = player_down * scale
        self.player_up = player_up * scale
        self.ball_size = ball_size * scale
        self.ball_speed = ball_speed * scale
        self.ball_limit = ball_limit
        self.collide_top = collide_top * scale
        self.collide_bottom = collide_bottom * scale
        self.collide_ball = collide_ball * scale
        self.immune_duration = immune_duration
        self.spawn_rate = spawn_rate
        self.freeze_duration = freeze_duration
        self.base_reward = base_reward
        self.collision_reward = collision_reward
        self.state_width = state_width
        self.state_height = state_height
        self.scale = scale
        self.seed = seed
        self.frame = 0
        self.immune = False
        self.frozen = False
        self.up_action = 1
        self.down_action = 0
        self.collision_action = 0
        self.collision_count = 0
        self.collision_start_frame = 0
        self.collision_end_frame = 0
        self.__action = 0
        self.balls = deque(maxlen=self.ball_limit)

        self.initialize_task()
        
    def _set_seed(self):
        if self.seed is None:
            self._genereate_random_seed()
        np.random.seed(self.seed)
        
    def _genereate_random_seed(self, high=99999):
        self.seed = np.random.randint(high)
    
    def initialize_task(self):
        self._set_seed()
        self.player = Player(screen=self.screen,
                           radius=self.player_size,
                           gravity=self.player_down,
                           fly=self.player_up,
                           screen_height=self.screen_height,
                           screen_width=self.screen_width)
        self.draw(self.immune)
        self.t_bound = (self.player.coord()[0], 0)
        self.b_bound = (self.player.coord()[0], self.screen_height)
        
    def get_time(self):
        """ Gets current time stamp.
            
            Add any time stamp modifications here, which will cause the rest of
            the base game use them.
        """
        #ts = datetime.datetime.now().time()
        #print(ts, ts.hour, ts.minute , ts.second ,ts.microsecond/1000000)
        #return float(ts.hour*3600 + ts.minute*60 + ts.second +ts.microsecond/1000000)
        return time.monotonic()

    def draw(self, immune):
        """ Draw all base game related objects.

            All the drawing of pygames graphics are done here. This function must be
            called after graphic values are changed (position, color, ect).

            Args:
                alive (bool): Determines the player's color depending on if they
                are in the immune state or not. 
        """
        # Clear screen
        self.screen.fill((255, 255, 255))

        # Draw Player
        self.player.draw((255,0,0)) if immune else self.player.draw((0,0,255))

        # Draw balls
        for ball in self.balls:
            ball.draw(color=(0,0,0))
            
        pygame.display.update()

    def disable_event(self, event):
        """ Disbales a given pygames event."""
        pygame.event.set_blocked(event)

    def enable_event(self, event):
        """ Enables a given pygames event."""
        pygame.event.set_allowed(event)
        
    def collision_handling(self):
        """ Handles collision events and player freezing.

            Handles collision events and initializes a freeze/freeze followed by 
            the immune state (this function also fills in missing state 
            realted information during the freeze).
        """
        self.immune = True # turn on immunity
        self.frozen = True
        self.draw(immune=self.immune) # redraw player
        self.collision_action = self.action
        self.collision_start_frame = self.frame # store current collision frame
        self.collision_start = self.get_time()
        
    def event_handling(self, dt):
        """ Handles all events for pygames and manually constructed events.

            Args:
                dt (float): This variable represents the time between frames in 
                seconds and is used for scaling continuous movement with FPS.
        """

        # Collision Checks
        if euclidean_distance(self.t_bound, self.player.coord()) <= self.collide_top:
            if self.player.y <= self.t_bound[-1]:
                self.player.y = self.t_bound[-1] + self.collide_top
            if not self.immune: 
                self.collision_handling()
                return
        if euclidean_distance(self.b_bound, self.player.coord()) <= self.collide_bottom:
            if self.player.y >= self.b_bound[-1]:
                self.player.y = self.b_bound[-1] - self.collide_bottom
            if not self.immune: 
                self.collision_handling()
                return
        for ball in self.balls:
            if euclidean_distance(ball.coord(), self.player.coord()) <= self.collide_ball:
                if not self.immune: 
                    self.collision_handling()
                    return
                
        self.draw(immune=self.immune)
        
    def step(self, action, dt):
        """ Main loop of OA game.

            Acts as the main loop of the game that drives all game mechanics 
            and functions. Any changes to how base game mechanics are executed 
            should be made here.

            Args:
                dt (float): This variable represents the time between frames in 
                seconds and is used for scaling continuous movement with FPS.
        """
        self.frame += 1 # iterate the current step
        self.action = action # Set action for current step unless frozen
        
        if self.frozen:
            if self.get_time() - self.collision_start >= self.freeze_duration:
                self.collision_end_frame = self.frame
                self.frozen = False
                self.immune_start = self.get_time()
            else:
                self.action = self.collision_action
                return

        # Check immunity duration
        if self.immune:
            if (self.get_time() - self.immune_start) >= self.immune_duration:
                self.immune = False
                self.draw(self.immune)

        # Move balls
        for ball in self.balls:
            ball.move(dt)

        # Remove balls that are off-screen
        while len(self.balls) > 0 and not self.balls[0].inside_screen():
            self.balls.popleft()

        # Spawn balls
        if np.random.rand() <= (self.spawn_rate*dt)/10 and len(self.balls) < self.ball_limit:
            ball = Ball(screen=self.screen,
                        radius=self.ball_size,
                        speed=self.ball_speed,
                        screen_height=self.screen_height,
                        screen_width=self.screen_width)
            self.balls.append(ball)

        # Move player
        if not self.frozen:
            if self.action == self.up_action:
                self.player.up(dt)
            else:
                self.player.down(dt)
                
        # Update player and ball movement on screen for more visual information
        # for player or agent. This allows visual information to represent collision
        # check which comes next.
        self.draw(self.immune)
        
        # Handle collision events. Check player after moving.
        self.event_handling(dt)
        
    
    @property
    def state(self):
        screen = self.get_screen()
        screen = pygame.transform.scale(
            screen, (self.state_width, self.state_height))
        state = pygame.surfarray.array3d(screen).astype(np.uint8)

        return np.transpose(state, (1, 0, 2))

    @property
    def reward(self):
        if self.immune:
            if self.frozen:
                return self.collision_reward
            return self.base_reward * -1
        return self.base_reward

    @property
    def action(self):
        return self.__action
    
    @action.setter
    def action(self, value):
        self.__action = value