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

        self.DEBUG_MODE = False
        self.rexy = None
        self.goal = None
        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.reset()

    def step(self, action):
        print("=====  STEPTIME  =====\n") if self.DEBUG_MODE else None
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

        # Feed action to the rexy and get observation of rexy's state
        print("Before apply_action") if self.DEBUG_MODE else None
        self.rexy.apply_action(action)
        print("After apply_action") if self.DEBUG_MODE else None
        p.stepSimulation()
        print("After stepSimulation") if self.DEBUG_MODE else None
        rexy_ob = self.rexy.get_observation()
        print("After get_observation") if self.DEBUG_MODE else None
        reward = self.compute_reward(rexy_ob)
        print("After compute_reward") if self.DEBUG_MODE else None
        ob = np.array(rexy_ob + self.goal, dtype=np.float32)
        print("After array conversion") if self.DEBUG_MODE else None
        return ob, reward, self.done, dict()

    def compute_reward(self, rexy_ob):
        print("------COMPUTEREWARD TIME--------\n") if self.DEBUG_MODE else None
        # Compute reward as L2 change in distance to goal
        dist_to_goal = math.sqrt(((rexy_ob[0] - self.goal[0]) ** 2 +
                                (rexy_ob[1] - self.goal[1]) ** 2))
        
        reward = max(self.prev_dist_to_goal - dist_to_goal, 0)

        """# Additional reward modifications or calculations can be done here

        # Penalize if the robot is not facing forwards
        forward_direction = np.array([1, 0])  # Assuming your forward direction is along the x-axis
        current_direction = np.array([math.cos(rexy_ob[2]), math.sin(rexy_ob[2])])
        alignment_penalty = -10  # Adjust as needed

        alignment_reward = np.dot(forward_direction, current_direction)
        if alignment_reward < 0.8:  # Example threshold, adjust as needed
            reward += alignment_penalty
            print("Alignment penalty applied") if self.DEBUG_MODE else None

        # Penalize if the robot is not upright
        orientation_quaternion = rexy_ob[1]  # Assuming orientation is the second element
        orientation_euler = p.getEulerFromQuaternion(orientation_quaternion)
        tilt_penalty = -10  # Adjust as needed

        # Check the pitch angle (rotation around the y-axis)
        pitch_angle = orientation_euler[1]
        if abs(pitch_angle) > 0.2:  # Example threshold, adjust as needed
            reward += tilt_penalty
            print("Tilt penalty applied") if self.DEBUG_MODE else None"""

        # Update prev_dist_to_goal for the next step
        self.prev_dist_to_goal = dist_to_goal

        print(f"DONE COMPUTING REWARD: {reward:.3f}") if self.DEBUG_MODE else None
        return reward

    def seed(self, seed: int) -> None:
        """Set the seeds for random and numpy
        Args: seed (int): The seed to set
        """
        np.random.seed(seed)
        random.seed(seed)
        
    def reset(self):
        print("=== Resetting Environment ===") if self.DEBUG_MODE else None

        # Reset the simulation and gravity
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.81)  # m/s^2

        # Reload the plane and rexy
        print("Loading Plane...") if self.DEBUG_MODE else None
        Plane(self.client)

        print("Creating Rexy...") if self.DEBUG_MODE else None
        self.rexy = Rexy(self.client)

        # Set the goal to a random target
        x = 7  # Hardcoded value for now
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

        observation = np.array(rexy_ob + self.goal, dtype=np.float32)
        info = {}

        print(f"Observation: {observation}") if self.DEBUG_MODE else None
        print(f"Observation: {info}") if self.DEBUG_MODE else None
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