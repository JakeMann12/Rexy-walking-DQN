import pybullet as p
import numpy as np

class Rexy:
    def __init__(self, client):
        p.setGravity(0,0,-9.81)
        self.client = client
        f_name = r"simple_rexy\resources\RexyURDF\jake.urdf"
        self.rexy = p.loadURDF(
            fileName=f_name,
            basePosition=[0, 0, 0.18],  # hardcoded
            physicsClientId=client,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
        )  # NOTE: added flags in extra from hello_bullet
        self.servo_joints = [2, 4, 6, 10, 12, 14] #hardcoded  
        self.max_force = 1.6671305  # NOTE: Pretty sure is Nm
        self.max_vel = 2*np.pi/1.14 #NOTE: somehwat assuming rad / sec

    def get_ids(self):
        return self.client, self.rexy

    def apply_action(
        self, action
    ):
        """
        Takes SIX-DIMENSIONAL ACTION INPUT and applies it to the servo joints via Joint Control
        """
        for i, joint_index in enumerate(self.servo_joints):
            # Apply control to each joint individually
            target_position = action[i]  # Set the desired position for the joint
            p.setJointMotorControl2(
                bodyUniqueId=self.rexy,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,  # Use POSITION_CONTROL for position control
                targetPosition=target_position,
                force=self.max_force,
                maxVelocity = self.max_vel
            )

    def get_observation(self):
        """
        Returns the x, y, z positions, and the roll, pitch, yaw angles, along with the x and y velocity components,
        and the joint angles for each of the servo joints.
        """
        # Get the position and orientation of the rexy in the simulation
        pos, orientation = p.getBasePositionAndOrientation(self.rexy, self.client)
        rpy = p.getEulerFromQuaternion(orientation)
        # Get the velocity of the rexy
        vel = p.getBaseVelocity(self.rexy, self.client)[0][0:2]
        # Get the joint angles
        joint_angles = [p.getJointState(self.rexy, joint_index, self.client)[0] for joint_index in self.servo_joints]
        joint_speeds = [p.getJointState(self.rexy, joint_index, self.client)[1] for joint_index in self.servo_joints]
        # Concatenate position, orientation, velocity, and joint angles
        observation = (pos[0], pos[1], pos[2], rpy[0], rpy[1], rpy[2], vel[0], vel[1]) + tuple(joint_angles) + tuple(joint_speeds)
        return np.array(observation, dtype=np.float32)
