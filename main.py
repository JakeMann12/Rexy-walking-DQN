import gym
import torch
#from agent import TRPOAgent
from simple_rexy.envs.rexy_env import SimpleRexyEnv
import time
import pybullet as p

#zswang666 and electricelephant

"""
def main(): # VERSION 1 (from medium post)
    #nn = torch.nn.Sequential(torch.nn.Linear(8, 64), torch.nn.Tanh(),
                             #torch.nn.Linear(64, 2))
    #agent = TRPOAgent(policy=nn)

    #agent.load_model("agent.pth")
    #agent.train("rexy-v0", seed=0, batch_size=5000, iterations=100,
                #max_episode_length=250, verbose=True)
    #agent.save_model("agent.pth")

    env = gym.make('rexy-v0')
    ob = env.reset()
    while True:
        # action = agent(ob) #OG
        action = env.action_space.sample()
        ob, _, done, _ = env.step(action)
        env.render()
        if done:
            ob = env.reset()
            time.sleep(1/30)"""

#physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version

def main(): #VeRSION 2 PANDA/Foo
    env = gym.make('rexy-v0')
    #print("\n\n\n"+str(env.metadata['render.modes'])+"\n\n\n") #does in fact return ['human']
    env.reset()      #observation = env.reset()
    #env.render(mode = "human")
    for _ in range(1000):
        # Render the environment in 'human' mode
        
        #env.render_mode('human') #definitely not right
        
        #p.connect(p.GUI)
        action = env.action_space.sample()
        #action = [0]*6 #test
        #print(action)
        observation, reward, done, info = env.step(action)
        
        if _ == 1: #comparison
            print(f"startobs:\n {observation}\n")
            print('\n'+str(env.rexy.get_observation())+'\n')
        if not env.observation_space.contains(observation):
            print("Invalid observation:", observation)
        if done:
            break
    print(f"got to the end point!\n {observation}")
    env.close()

if __name__ == '__main__':
    main()
