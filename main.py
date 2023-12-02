import gym
import torch
#from agent import TRPOAgent
from simple_rexy.envs.rexy_env import SimpleRexyEnv
import time
import pybullet as p

#zswang666 and electricelephant

def main(): #VeRSION 2 PANDA/Foo
    env = gym.make('rexy-v0')
    env.reset()      #observation = env.reset()
    steps = 500
    for _ in range(steps):
        action = env.action_space.sample()
        #action = [0]*6 #test
        #print(action)
        observation, reward, done, info = env.step(action)
        
        if _ == 1: #comparison
            print(f"startobs:\n {observation}\n")
            print('\n'+str(env.rexy.get_observation())+'\n')
        if not env.observation_space.contains(observation):
            print("Invalid observation:", observation)
        if _ == steps-1:
            print(f'got to the end of the for loop! \n{observation}\n')
        if done:
            break

    print(f"got to the end point!\n {observation}")
    env.close()

if __name__ == '__main__':
    main()
