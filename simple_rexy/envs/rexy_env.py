#needs heavy mods

import os
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import math
import pybullet as p
import matplotlib.pyplot as plt
from simple_rexy.resources.REXY import Rexy
from simple_rexy.resources.plane import Plane #called in RESET FUNCT
from simple_rexy.resources.goal import Goal #called in RESET FUNCT
import random

#print(Rexy.get_observation)

class SimpleRexyEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}  
    
    def __init__(self, self_collision_enabled = True):
        self._self_collision_enabled = self_collision_enabled
        self.servoindices = [2, 4, 6, 10, 12, 14] #hardcoded

        self.action_space = gym.spaces.box.Box(
            # NOTE : set values to 0?
            #low=np.array([30,30,25,20,20,100], dtype=np.float32), # LOW VALUES FOR EACH SERVO
            #high=np.array([210,150,160,220,178,230], dtype=np.float32)) #HIGH VALUES FOR EACH SERVO 
            low = np.float32(-.4*np.pi*np.ones_like(self.servoindices)), # < -.4 * 180 degrees for all- prob overkill still
            high = np.float32(.4*np.pi*np.ones_like(self.servoindices)))
        self.observation_space = gym.spaces.box.Box(
            low=np.float32(np.array([-10, -10, -1, -1, -5, -5, 0, 0])),
            high=np.float32(np.array([10, 10, 1, 1, 5, 5, 10, 10]))) #NOTE: Fixed x bounds
        # consisting of xy position of the rexy, unit xy orientation of the rexy, xy velocity of the rexy, and xy position of a target we want to reach.
        self.np_random, _ = gym.utils.seeding.np_random()

        self.client = p.connect(p.GUI) # NOTE : GUI OR DIRECT
        # Reduce length of episodes for RL algorithms
        p.setTimeStep(1/30, self.client)

        self.rexy = None
        self.goal = None
        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.reset()

    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

        # Feed action to the rexy and get observation of rexy's state
        self.rexy.apply_action(action)
        p.stepSimulation()
        
        rexy_ob = self.rexy.get_observation()

        # Compute reward as L2 change in distance to goal
        dist_to_goal = math.sqrt(((rexy_ob[0] - self.goal[0]) ** 2 +
                                  (rexy_ob[1] - self.goal[1]) ** 2))
        reward = max(self.prev_dist_to_goal - dist_to_goal, 0)
        self.prev_dist_to_goal = dist_to_goal

        # Done by running off boundaries
        if (rexy_ob[0] >= 10 or rexy_ob[0] <= -10 or
                rexy_ob[1] >= 10 or rexy_ob[1] <= -10):
            self.done = True
        # Done by reaching goal
        elif dist_to_goal < 1:
            self.done = True
            reward = 50

        ob = np.array(rexy_ob + self.goal, dtype=np.float32)
        return ob, reward, self.done, dict()

    def seed(self, seed: int) -> None:
        """Set the seeds for random and numpy
        Args: seed (int): The seed to set
        """
        np.random.seed(seed)
        random.seed(seed)
        
    def reset(self):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.81) #m/s^2
        # Reload the plane and rexy
        Plane(self.client)
        self.rexy = Rexy(self.client)

        # Set the goal to a random target
        x = self.np_random.uniform(6, 8) # if self.np_random.integers(0, 2) else self.np_random.uniform(-5, -9) #NOTE: I guess that I can just vary the one distance
        y = 0 #NOTE: CHANGED HERE
        self.goal = (x, y)
        self.done = False

        # Visual element of the goal
        Goal(self.client, self.goal)

        # Get observation to return
        rexy_ob = self.rexy.get_observation()

        self.prev_dist_to_goal = math.sqrt(((rexy_ob[0] - self.goal[0]) ** 2 +
                                           (rexy_ob[1] - self.goal[1]) ** 2))
        return np.array(rexy_ob + self.goal, dtype=np.float32)

    
    
    def render(self): #, mode='human'): #NOTE: didn't change anything but car-> rexy
        #p.connect(p.GUI)
        if self.rendered_img is None:
            self.rendered_img = plt.imshow(np.zeros((100, 100, 4)))

        # Base information
        rexy_id, client_id = self.rexy.get_ids()
        proj_matrix = p.computeProjectionMatrixFOV(fov=1200, aspect=1,
                                                   nearVal=0.01, farVal=100)
        pos, ori = [list(l) for l in
                    p.getBasePositionAndOrientation(rexy_id, client_id)]
        #pos[2] = 0.2 #NOTE: What the hell is this

        # Rotate camera direction
        rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
        view_matrix = p.computeViewMatrix(pos, pos + camera_vec, up_vec)

        # Display image
        frame = p.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
        frame = np.reshape(frame, (100, 100, 4))
        self.rendered_img.set_data(frame)
        plt.draw()
        plt.pause(.00001)
    
    """
    def render(self, mode='human'):
        mode = {'render.modes': [mode]}
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7,0,0.05],
                                                            distance=.7,
                                                            yaw=90,
                                                            pitch=-70,
                                                            roll=0,
                                                            upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(960) /720,
                                                     nearVal=0.1,
                                                     farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                              height=720,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720,960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array"""
    
    def close(self):
        p.disconnect(self.client)