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
        self.max_vel = 3 * 2*np.pi/1.14 #NOTE: somehwat assuming rad / sec. Tripled to make sure.

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
        # Get position, orientation, and velocity in fewer calls
        pos, orientation = p.getBasePositionAndOrientation(self.rexy, self.client)
        rpy = p.getEulerFromQuaternion(orientation)
        vel = p.getBaseVelocity(self.rexy, self.client)[0][:2]

        # Batch query for joint states if possible
        joint_states = p.getJointStates(self.rexy, self.servo_joints, self.client)
        joint_angles, joint_speeds = zip(*[(state[0], state[1]) for state in joint_states])

        # Create observation array efficiently
        observation = np.array(pos + rpy + vel + joint_angles + joint_speeds, dtype=np.float32)
        return observation
