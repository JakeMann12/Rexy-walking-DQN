import pybullet as p
import time
import pybullet_data
import numpy as np

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0,0,0.18]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
robotId = p.loadURDF(r"C:\Users\jmann\Box\Dook Work\Robot Learning\simple-rexy\simple_rexy\resources\RexyURDF\jake.urdf",cubeStartPos, cubeStartOrientation, 
                   # useMaximalCoordinates=1, ## New feature in Pybullet
                   flags=p.URDF_USE_INERTIA_FROM_FILE)
good_motors = [2,4,6,10,12, 14]
mode = p.POSITION_CONTROL


#from mediocre hill climber results:
sinparams = [[0.70312131, 0.04431859, 0.005915  ],
            [0.26623661, 0.57702719, 0.003185  ],
            [0.07756777, 0.91296144, 0.0041405 ],
            [0.34295338, 1.31905742, 0.00455   ],
            [0.87986487, 2.08979675, 0.0076895 ],
            [0.28227091, 1.16768994, 0.00455   ]]
#a+b*np.sin(i*c)

startpos = (32.88, 94.32, 50, 147, 98.16, 136)
#zeroes = [0]*len(good_motors) #stupid - just tell it to go to 0

for i in range(10000):
    p.stepSimulation()
    for servo in good_motors:   
        p.setJointMotorControl2(robotId, servo, controlMode=mode, targetPosition= 45 * (-1) **i, maxVelocity = 2*np.pi/1.14)


cubePos, cubeOrn = p.getBasePositionAndOrientation(robotId)
print(cubePos,cubeOrn)
p.disconnect()