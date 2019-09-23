import gym
from Agents import QAgent
import numpy as np
from crawler_env import CrawlingRobotEnv
import time



env = CrawlingRobotEnv(render=False)

agent = QAgent(env,gamma=0.9)


env = CrawlingRobotEnv(render=True)

time.sleep(10)
