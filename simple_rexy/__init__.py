from gym.envs.registration import register

register(
    id='rexy-v0',
    entry_point='simple_rexy.envs:SimpleRexyEnv',
)

