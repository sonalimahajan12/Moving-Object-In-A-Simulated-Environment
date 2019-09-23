import random
from collections import defaultdict
import numpy as np
class QAgent():
	def __init__(self,env,gamma):
		self.gamma = 0.9
		self.env = env
		self.q_vals = defaultdict(lambda: np.array([0. for _ in range(env.action_space.n)]))

	def choose_action(self,state):
		i=1
		#todo

	def learn(self, cur_state,action,reward,next_state):
		i=1
		#todo
