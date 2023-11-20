import pybullet as p
import os
import math
from numpy import pi

#%% NOTE: REQUIRES HEAVY EDITING

class Rexy:
    def __init__(self, client):
        self.client = client
        f_name = r"C:\Users\jmann\Box\Dook Work\Robot Learning\robot_code\URDFs\feetasrigidjts\jake.urdf"
        self.rexy = p.loadURDF(fileName=f_name,
                              basePosition=[0, 0, 0.18], #hardcoded
                              physicsClientId=client,
                              flags=p.URDF_USE_INERTIA_FROM_FILE) #NOTE: added this in extra from hello_bullet
        self.servo_joints = [2, 4, 6, 10, 12, 14] #steering joints, I believe, with reference to the other shit

    def get_ids(self):
        return self.client, self.rexy

    def apply_action(self, action): #NOTE: THIS IS WHERE WE ARE GOING TO SEE ISSUES I BET
        # Expects action to be SIX DIMENSIONAL

        # Clip throttle and steering angle to reasonable values - we don't need throttle
        #throttle = min(max(throttle, 0), 1)

        #servo_angles = max(min(servo_angles, .4*pi), -.4*pi) #NOTE: should these be clipped to the bounds in the action space?

        # Define parameters for joint control
        control_mode = p.POSITION_CONTROL  # Use POSITION_CONTROL for position control
        max_force = 100.0  # You can adjust the maximum force as needed

        for i, joint_index in enumerate(self.servo_joints):
            # Apply control to each joint individually
            target_position = action[i]  # Set the desired position for the joint
            p.setJointMotorControl2(
                bodyUniqueId=self.rexy,
                jointIndex=joint_index,
                controlMode=control_mode,
                targetPosition=target_position,
                force=max_force
            )
    
    def get_observation(self):
        # Get the position and orientation of the rexy in the simulation
        pos, ang = p.getBasePositionAndOrientation(self.rexy, self.client)
        ang = p.getEulerFromQuaternion(ang)
        ori = (math.cos(ang[2]), math.sin(ang[2]))
        pos = pos[:2]
        # Get the velocity of the rexy
        vel = p.getBaseVelocity(self.rexy, self.client)[0][0:2]

        # Concatenate position, orientation, velocity
        observation = (pos + ori + vel)

        return observation

    









