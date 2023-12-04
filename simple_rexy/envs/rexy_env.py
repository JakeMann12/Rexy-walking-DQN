import torch
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from numpy import pi
import math
import pybullet as p
import matplotlib.pyplot as plt
from simple_rexy.resources.REXY import Rexy
from simple_rexy.resources.plane import Plane #called in RESET FUNCT
from simple_rexy.resources.goal import Goal #called in RESET FUNCT
import random

class SimpleRexyEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}  

    #Reward Function Tuning Vals
    FORWARD_YAW_THRESHOLD = pi / 3
    ALIGNMENT_PENALTY = -10
    PITCH_DEGREES = 45
    MAX_PITCH_THRESHOLD = np.radians(PITCH_DEGREES)
    TIPPING_PENALTY = -75
    SURVIVAL_REWARD = 1  # Small positive reward for staying alive
    TIME_PENALTY_SCALE = 0.01  # Adjust as needed
    MAX_STEPS_EPISODE = 800 #how long it can go
    
    def __init__(self, client = p.connect(p.DIRECT), self_collision_enabled = True): # NOTE : GUI OR DIRECT
        self._self_collision_enabled = self_collision_enabled
        self.servoindices = [2, 4, 6, 10, 12, 14] #hardcoded

        self.action_space = gym.spaces.box.Box(
            low = np.float32(-.4*pi*np.ones_like(self.servoindices)), # < -.4 * 180 degrees for all - prob overkill still
            high = np.float32(.4*pi*np.ones_like(self.servoindices)))
        self.observation_space = gym.spaces.box.Box(
            # x, y, z positions, roll, pitch, yaw angles, x and y velocity components. NOTE: REMOVED x and y position of goal
            low =np.float32(np.array([-10, -5,  0, -pi, -pi/2, -pi, -1, -1])),
            high=np.float32(np.array([ 10,  5,  1,  pi,  pi/2,  pi,  5,  5]))) 
        
        self.np_random, _ = gym.utils.seeding.np_random()
        self.client = client 
        # Reduce length of episodes for RL algorithms
        p.setTimeStep(1/30, self.client)

        self.DEBUG_MODE = False
        self.rexy = None
        self.goal = None
        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.reset()
        self.current_step = 0  # Initialize the current step
        self.first_obs = self.rexy.get_observation()

    def step(self, action):
        """
        Takes 6-D array input, calls reward function and returns
        values for observation, reward, and 
        """
        print("=====  STEPTIME  =====\n") if self.DEBUG_MODE else None
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

        # Feed action to the rexy and get observation of rexy's state
        print("Before apply_action") if self.DEBUG_MODE else None
        self.rexy.apply_action(action)
        print("After apply_action") if self.DEBUG_MODE else None
        p.stepSimulation()
        print("After stepSimulation") if self.DEBUG_MODE else None
        rexy_ob = self.rexy.get_observation()

        ob = np.array(rexy_ob, dtype=np.float32)
        reward = self.compute_reward(rexy_ob) 
        done = self.done
        
        self.current_step += 1  # Increment the current step at each call to step()
        return ob, reward, done, dict()

    def compute_reward(self, rexy_ob):
        """
        Takes 8-part rexy obs (x y z r p y v_x v_y) and rewards for:
        1. Closeness to goal
        2. Being pointed towards the goal (yaw)
            - penalty if not
        3. Punishes for tilting downwards too far (pitch)
            - breaks if outside of range (fell over)
        4. Survival reward for staying alive
        5. Time-dependent penalty for taking too long
        """
        
        print("------COMPUTEREWARD TIME--------\n") if self.DEBUG_MODE else None

        # Compute reward as L2 change in distance to goal
        dist_to_goal = math.sqrt(((rexy_ob[0] - self.goal[0]) ** 2 +
                                (rexy_ob[1] - self.goal[1]) ** 2))
        
        reward = max(self.prev_dist_to_goal - dist_to_goal, 0) * 100

        # Penalize if the robot is not facing forwards (based on yaw angle)
        yaw_angle = rexy_ob[5]

        if abs(yaw_angle) > self.FORWARD_YAW_THRESHOLD:
            reward += self.ALIGNMENT_PENALTY
            print("Alignment penalty applied") if self.DEBUG_MODE else None

        # Penalize if the robot is not upright (based on pitch angle)
        pitch_angle = rexy_ob[4]
        pitch_difference = abs(pitch_angle - self.first_obs[4])

        if pitch_difference > self.MAX_PITCH_THRESHOLD:
            reward += self.TIPPING_PENALTY
            print("Tipping penalty applied") if self.DEBUG_MODE else None
            self.done = True  # End the episode if tipping occurs

        # Survival reward for staying alive
        reward += self.SURVIVAL_REWARD

        # Time-dependent penalty for taking too long
        time_penalty = -self.TIME_PENALTY_SCALE * self.current_step
        reward += time_penalty

        # Timeout
        if self.current_step >= self.MAX_STEPS_EPISODE:
            self.done = True

        # Update prev_dist_to_goal for the next step
        self.prev_dist_to_goal = dist_to_goal

        print(f"DONE COMPUTING REWARD: {reward:.3f}") if self.DEBUG_MODE else None
        return reward
        
    def reset(self, seed = None, options = {}):
        print("=== Resetting Environment ===") if self.DEBUG_MODE else None
        options, seed = options, seed #Worthless- avoids an error

        # Reset the simulation and gravity
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.81) #-9.81)  # m/s^2

        # Reload the plane and rexy
        print("Loading Plane...") if self.DEBUG_MODE else None
        Plane(self.client)

        print("Creating Rexy...") if self.DEBUG_MODE else None
        self.rexy = Rexy(self.client)

        # Set the goal to a target
        x = 3  # Hardcoded value for now
        y = 0
        self.goal = (x, y)
        self.done = False

        # Visual element of the goal
        print(f"Creating Goal at {self.goal}...") if self.DEBUG_MODE else None
        Goal(self.client, self.goal)

        # Get observation to return
        rexy_ob = self.rexy.get_observation()

        self.prev_dist_to_goal = math.sqrt(((rexy_ob[0] - self.goal[0]) ** 2 +
                                        (rexy_ob[1] - self.goal[1]) ** 2))

        print(f"Prev dist to goal: {self.prev_dist_to_goal:.3f}") if self.DEBUG_MODE else None

        observation = np.array(rexy_ob, dtype=np.float32)
        info = {}

        print(f"Observation (get_obs): {observation}") if self.DEBUG_MODE else None
        print(f"Info: {info}") if self.DEBUG_MODE else None
        print("=== Reset Complete ===") if self.DEBUG_MODE else None

        return observation, info
    
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
    
    def close(self):
        p.disconnect(self.client)