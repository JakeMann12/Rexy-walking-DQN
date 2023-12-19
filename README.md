# Jake Mann REXY environment
Basic OpenAI gym environment for Rexy the T-REX. 

This project explored using Deep Q Networks to get a Bipedal Robot named Rexy to Walk. Check out some pictures at this link! This code uses gym (NOT GYMNASIUM), pybullet, and torch, namely, along with tensorflow and cProfile for analyzing the code and its outputs. This project was... not a great success. But it taught me an insane amount and I'm so glad that I did it. If you're reading this thinking about doing a similar locomotion project, DO NOT USE DQN!!!! DQN requires a discreet action space, which for a robot with six servos, is $6^\inf$ combinations!

Resource for [Part Two: Getting Rexy to Walk with Deep Learning](https://jmann6702.wixsite.com/jake/rexy-the-t-rex). 
