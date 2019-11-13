import gym
from Agents import QAgent
import numpy as np
from crawler_env import CrawlingRobotEnv
import time
import getch




env = CrawlingRobotEnv(render=False)

agent = QAgent(env,gamma=0.9)



all_rewards=0
current_state=env.reset()
total_reward = 0
i = 0
while i < 50000:
    i=i+1
    action = agent.choose_action(current_state)
    next_state,reward,done,info = env.step(action)
    agent.learn(current_state,action,reward,next_state)
    current_state = next_state
    total_reward += reward

    if i % 1000 == 0: # evaluation
        print(i, "average_reward in last 1000 steps", total_reward / i)
        if (total_reward / i) > 1.3:
            break
        average_reward = 0
        env.render = True


env = CrawlingRobotEnv(render=True)
current_state=env.reset()
total_reward = 0
agent.eps = 0

print()

i = 0
while i < 5000:
    i=i+1
    action = agent.choose_action(current_state)
    next_state,reward,done,info = env.step(action)
    agent.learn(current_state,action,reward,next_state)
    current_state = next_state
    total_reward += reward

    if i % 1000 == 0: # evaluation
        print("total reward till now", total_reward)
        average_reward = 0
        env.render = True
       

#time.sleep(2)
msvcrt.getch()
