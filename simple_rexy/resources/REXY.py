import pybullet as p
import numpy as np


# %% NOTE: REQUIRES HEAVY EDITING


class Rexy:
    def __init__(self, client):
        self.client = client
        f_name = r"simple_rexy\resources\RexyURDF\jake.urdf"
        self.rexy = p.loadURDF(
            fileName=f_name,
            basePosition=[0, 0, 0.18],  # hardcoded
            physicsClientId=client,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
        )  # NOTE: added this in extra from hello_bullet
        self.servo_joints = [2, 4, 6, 10, 12, 14,
        ]  # steering joints, I believe, with reference to the other example
        self.max_force = 1.6671305  # NOTE: Pretty sure is Nm

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
            )

    def get_observation(self):
        """
        returns the x, y, z positions, and the roll, pitch, yaw angles, along with the x and y velocity components
        """
        # Get the position and orientation of the rexy in the simulation
        pos, orientation = p.getBasePositionAndOrientation(self.rexy, self.client)
        rpy = p.getEulerFromQuaternion(orientation)
        # Get the velocity of the rexy
        vel = p.getBaseVelocity(self.rexy, self.client)[0][0:2]
        # Concatenate position, orientation, velocity
        observation = (pos[0], pos[1], pos[2], rpy[0], rpy[1], rpy[2], vel[0], vel[1])
        return np.array(observation, dtype=np.float32)
