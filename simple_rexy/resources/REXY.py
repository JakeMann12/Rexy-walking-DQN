import pybullet as p
import numpy as np
import gym
from numpy import pi

class Rexy:
    def __init__(self, client):
        p.setGravity(0,0,-9.81)
        self.client = client
        f_name = "simple_rexy/resources/RexyURDF/jake.urdf"
        self.rexy = p.loadURDF(
            fileName=f_name,
            basePosition=[0, 0, 0.18],  # hardcoded
            physicsClientId=client,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
        )  # NOTE: added flags in extra from hello_bullet
        self.servo_joints = [2, 4, 6, 10, 12, 14] #hardcoded  
        self.max_force = 1.6671305  # NOTE: Pretty sure is Nm
        self.max_vel = 2 * np.pi/1.14 # rad / sec. 
        self.joint_min = -.2 * pi
        self.joint_max = .2 * pi

    def get_ids(self):
        return self.client, self.rexy

    def apply_action(
        self, jvals, action_index
    ):
        """
        Takes previous joint values and an index number within [0,63]. 
        Converts index to binary and multiplies it by XXX, adds or sub the current jval
        """
        action_movements = self.index_to_action(action_index)

        for i, joint_index in enumerate(self.servo_joints):
            # Calculate the new target position for each joint
            # Ensure that the target position is within the joint limits
            target_position = max(min(jvals[i] + action_movements[i], self.joint_max), self.joint_min)

            # Apply control to each joint individually
            p.setJointMotorControl2(
                bodyUniqueId=self.rexy,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,  # Use POSITION_CONTROL for position control
                targetPosition=target_position,
                force=self.max_force,
                maxVelocity=self.max_vel
            )

    def index_to_action(self, index):
        # Convert the index to a 6-bit binary string
        binary_str = format(index, '06b')
        # Translate the binary string to actions for each joint
        actions = [-self.max_vel/30 if bit == '0' else self.max_vel/30 for bit in binary_str]
        return actions

    def min_max_scale(self, observation):
        num_joints = len(self.servo_joints)
        low=np.float32(np.array([-1, -2, 0, -pi, -pi/2, -pi, -4, -4] + [self.joint_min]*num_joints + [-self.max_vel] * num_joints))
        high=np.float32(np.array([5, 2, .4, pi, pi/2, pi, 4, 4] + [self.joint_max]*num_joints + [self.max_vel] * num_joints))
        # Compute the scale for each feature and compute mix-max
        scales = 1 / (high - low)
        scaled_observation = (observation - low) * scales
        return scaled_observation
    

    def get_observation(self):
        # Get position, orientation, and velocity in fewer calls
        pos, orientation = p.getBasePositionAndOrientation(self.rexy, self.client)
        rpy = p.getEulerFromQuaternion(orientation)
        vel = p.getBaseVelocity(self.rexy, self.client)[0][:2]

        # Batch query for joint states if possible
        joint_states = p.getJointStates(self.rexy, self.servo_joints, self.client)
        joint_angles, joint_speeds = zip(*[(state[0], state[1]) for state in joint_states])

        # Create observation array efficiently
        observation = np.array(pos + rpy + vel + joint_angles + joint_speeds, dtype=np.float32)
        observation = self.min_max_scale(observation)
        return observation
        

    def get_ids(self):
            return self.client, self.rexy