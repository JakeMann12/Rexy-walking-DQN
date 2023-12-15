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
    FORWARD_YAW_THRESHOLD = 2 * pi / 5  # Adjusted threshold for alignment
    ALIGNMENT_PENALTY = -50  # Penalty for misalignment
    PITCH_DEGREES = 25  # Maximum pitch threshold
    MAX_PITCH_THRESHOLD = np.radians(PITCH_DEGREES)
    TIPPING_PENALTY = -150  # Penalty for tipping
    HEIGHT_REWARD = 100  # Reward for being at the correct height
    HEIGHT_PENALTY = -100  # Penalty for being outside the height range
    SURVIVAL_REWARD = 150  # Reward for staying alive
    TIMEOUT_PENALTY = -50  # Penalty for reaching the timeout
    X_DIST_REWARD_COEF = 0  # Coefficient for distance reward (de-emphasized)
    X_VEL_REWARD_COEFF = 0  # Coefficient for X-velocity reward (de-emphasized)
    REACH_GOAL_REWARD = 0  # Reward for reaching the goal (de-emphasized)
    
    def __init__(self, client = p.connect(p.GUI), self_collision_enabled = True): # NOTE : GUI OR DIRECT
        self._self_collision_enabled = self_collision_enabled
        self.servoindices = [2, 4, 6, 10, 12, 14] #hardcoded

        self.action_space = gym.spaces.box.Box(
            low = np.float32(-.25*pi*np.ones_like(self.servoindices)), # < -.4 * 180 degrees for all - prob overkill still
            high = np.float32(.25*pi*np.ones_like(self.servoindices)))
        self.observation_space = gym.spaces.box.Box(
            # x, y, z positions, roll, pitch, yaw angles, x and y velocity components. NOTE: REMOVED x and y position of goal
            low =np.float32(np.array([-5, -5,  0, -pi, -pi/2, -pi, -1, -1])),
            high=np.float32(np.array([ 5,  5,  .4,  pi,  pi/2,  pi,  5,  5]))) 
        
        self.client = client 
        self.rexy = Rexy(self.client)
        Plane(self.client)
        self.goal = (1.5, 0) #hardcoded for now- moved up from reset
        Goal(self.client, self.goal)
        
        # Save the initial state
        self.initial_state = p.saveState(physicsClientId=self.client)
        # Reduce length of episodes for RL algorithms
        p.setTimeStep(1/30, self.client)

        self.DEBUG_MODE = False

        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.reset()
        self.current_step = 0  # Initialize the current step
        self.first_obs = self.rexy.get_observation()

        self.best_reward_yet = 0
        self.new_best = False

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
        reward, newbest = self.compute_reward(rexy_ob) 
        done = self.done
        
        self.current_step += 1  # Increment the current step at each call to step()
        return ob, reward, done, newbest

    def compute_reward(self, rexy_ob):
        """
        Takes 8-part rexy obs (x y z r p y v_x v_y) and rewards for:
        1. Closeness to goal
        2. X-velocity
        2. Being pointed towards the goal (yaw)
            - penalty if not
        3. Punishes for tilting downwards too far (pitch)
            - breaks if outside of range (fell over)
        4. Being on the ground
            - punishes if goes higher than .35 m (starts at .25 m)
        5. Survival reward for staying alive
        6. Time-dependent penalty for taking too long
        7. Exit condition for reaching the goal
        """
        
        print("------COMPUTE-REWARD TIME--------\n") if self.DEBUG_MODE else None

        # Initialize reward
        reward = 0

        # Penalize if the robot is not facing forwards (based on yaw angle)
        yaw_angle = rexy_ob[5] - self.first_obs[5]
        if abs(yaw_angle) > self.FORWARD_YAW_THRESHOLD:
            reward += self.ALIGNMENT_PENALTY

        # Punishes for tilting downwards too far (pitch)
        pitch_angle = rexy_ob[4]
        pitch_difference = abs(pitch_angle - self.first_obs[4])
        if pitch_difference > self.MAX_PITCH_THRESHOLD:
            reward += self.TIPPING_PENALTY
            self.done = True

        # Rewards/punishes based on Z height
        z_height = rexy_ob[2]
        if .145 < z_height <= .325:
            reward += self.HEIGHT_REWARD
        else:
            reward += self.HEIGHT_PENALTY
            self.done = True

        # Survival reward for staying alive
        reward += self.SURVIVAL_REWARD

        # Timeout penalty
        if self.current_step >= 600:
            reward += 1000
            print('got to the end!')
            self.done = True

        #save if we broke a record (exclude weird fringe at start)
        if reward > self.best_reward_yet and self.current_step > 100:
            self.new_best = True
            print('New High Score!')
            self.best_reward_yet = reward

        return reward, self.new_best
        
    def reset(self, seed = None, options = {}):
        print("=== Resetting Environment ===") if self.DEBUG_MODE else None
        options, seed = options, seed #Worthless- avoids an error
        
        self.new_best = False
        self.done = False
        self.current_step = 0

        print("Resetting Rexy Pos...") if self.DEBUG_MODE else None
        p.restoreState(self.initial_state, physicsClientId=self.client)
        
        # Get observation to return
        rexy_ob = self.rexy.get_observation()

        self.prev_dist_to_goal = math.sqrt(((rexy_ob[0] - self.goal[0]) ** 2 +
                                        (rexy_ob[1] - self.goal[1]) ** 2))

        print(f"Prev dist to goal: {self.prev_dist_to_goal:.3f}") if self.DEBUG_MODE else None

        info = {}
        

        print(f"Observation (get_obs): {rexy_ob}") if self.DEBUG_MODE else None
        print(f"Info: {info}") if self.DEBUG_MODE else None
        print("=== Reset Complete ===") if self.DEBUG_MODE else None

        return rexy_ob, info
    
    def render(self): #, mode='human'): #NOTE: didn't change anything but car-> rexy
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