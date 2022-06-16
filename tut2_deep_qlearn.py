import gym
import numpy as np 
from collections import deque
import random

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

# ignore sklearn warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

env = gym.make('MountainCar-v0')

action_list=[0, 1, 2]
gamma = 0.99  	# discount factor
lr = 0.001		# learning rate
num_episodes = 2000		# 
epsilon = 1		# epsilon for greedy search
epsilon_decay = 0.995
memory_size = 1000
batch_size = 100
show=True
action_size = env.action_space.n
state_size = env.observation_space.shape[0]
#print(env.observation_space, env.observation_space.shape)
#print(action_size, type(action_size))

factor=[1, 100]

memory = deque(maxlen=memory_size)


# get initial state
s = env.reset()
#print(s.reshape((1,-1)).reshape(-1,s.reshape((1,-1)).shape[0],s.reshape((1,-1)).shape[1]))
s = s.reshape((1,-1))
s = s*factor
cum_r = 0

# fill memory... why random?
for i in range(memory_size):
	a = env.action_space.sample()
	new_s, r, d, info = env.step(a)
	new_s = new_s.reshape((1,-1))		# this reshape is to fit the format req by network
	new_s = new_s*factor
	cum_r = cum_r + r

	if show:
		env.render()
	if d:	# finished
		experience = (s, r, a, new_s, d)
		s = env.reset()
		s = s.reshape((1,-1))
		if cum_r > -199:
			print("Step {}: Success by random actions!".format(i))
	else:
		experience = (s, r, a, new_s, d)
	memory.append(experience)
	s = new_s
env.close()

# create neural network
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(1,state_size)))
model.add(Dense(100, activation='relu'))
model.add(Dense(action_size, activation = 'linear'))

# compile
model.compile(loss='mse', optimizer = Adam(lr=lr))

# print
model.summary()


ep_list =[]
reward_list =[] 
index=0 
oh = OneHotEncoder(n_values=3)
for ep in range(num_episodes):
    s= env.reset()
    s=s.reshape((1,-1))
    s = s*factor
    total_rewards =0
    d = False
    j = 0
    for j in range(200):
        if np.random.random()< epsilon:
            a = np.random.randint(0,len(action_list))
        else:
            Q = model.predict(s.reshape(-1,s.shape[0],s.shape[1]))
            a =np.argmax(Q)
        new_s,r,d,_ = env.step(a)
        new_s = new_s.reshape((1,-1))
        new_s = new_s*factor
        total_rewards=total_rewards+r
        if show:
            env.render()
        if d:
            if total_rewards<-199:
                r =-100
                experience = (s,r,a,new_s,d)
                memory.append(experience)
                print("Episode %d, Failed! Reward %d"%(ep,total_rewards))
            elif total_rewards<-110 and total_rewards>-199:
                r=10
                d=True
                experience = (s,r,a,new_s,d)
                memory.append(experience)
                print("Episode %d, Better! Reward %d"%(ep,total_rewards))
            elif total_rewards>=-110:
                r=100
                experience = (s,r,a,new_s,d)
                memory.append(experience)

                print("Episode %d, Passed! Reward %d"%(ep,total_rewards))
            ep_list.append(ep)
            reward_list.append(total_rewards)
            break
        
        experience = (s,r,a,new_s,d)
        memory.append(experience)
        if j==199:
            print("Reward %d after full episode"%(total_rewards))
            
        s = new_s
    batches=random.sample(memory,batch_size)
    states= np.array([batch[0] for batch in batches])
    rewards= np.array([batch[1] for batch in batches])
    actions= np.array([batch[2] for batch in batches])
    actions=oh.fit_transform(actions.reshape(-1,1)).toarray()
    actions = actions.reshape(-1,1,action_size)
    new_states= np.array([batch[3] for batch in batches])
    dones= np.array([batch[4] for batch in batches])
    Qs =model.predict(states)
    new_Qs = model.predict(new_states)
    target_Qs=rewards.reshape(-1,1)+gamma*(np.max(new_Qs,axis=2)*(~dones.reshape(-1,1)))
    Qs[actions==1]=target_Qs.reshape(-1,)
    model.fit(states,Qs,verbose=0)
    epsilon=epsilon*epsilon_decay
env.close()

"""
ep_list =[]
reward_list =[] 
oh = OneHotEncoder(n_values=3)
for ep in range(num_episodes):

	s = env.reset().reshape((1,-1)) * factor
	sum_rewards = 0
	d = False
	
	for j in range(200):

		# do epsilon-greedy sampling
		if (np.random.random() < epsilon):	# sample from uniform distribution
			a = np.random.randint(0, action_size)
		else:
			Q = model.predict(s.reshape(-1,s.shape[0],s.shape[1]))   # need this for batch processing
			a = np.argmax(Q)


		new_s, r, d, info = env.step(a)
		new_s = new_s.reshape((1, -1)) * factor
		sum_rewards = sum_rewards + r

		if show:
			env.render()
		if d:
			if sum_rewards < -199 :
				# big punishment
				r = -100
				experience = (s, r , a, new_s, d)
				memory.append(experience)
				print("Failed! Episode {} Sum of reward: {}".format(ep, sum_rewards))

			elif sum_rewards < -110 and sum_rewards > -199:
				# small reward
				r = 10
				experience = (s, r, a, new_s, d)
				memory.append(experience)
				print("Passed! Episode {} Sum of rewards: {}".format(ep, sum_rewards))

			elif sum_rewards >= -110:
				# big reward
				r = 100
				experience = (s, r, a, new_s, d)
				memory.append(experience)
				print("Passed! Episode {} Sum of rewards: {}".format(ep, sum_rewards))

			else:
				# This never happens
				pass

			# ep_list.append(ep)
			reward_list.append(sum_rewards)
			break

		experience = (s, r, a, new_s, d)
		memory.append(experience)
		

		if j == 199:
			print("Reward after episode {}: {}".format(ep, sum_rewards))
		else:
			pass

		s = new_s

	# create training batch
	batches = random.sample(memory,batch_size)		# get random experience steps
	states = np.array([experience[0] for experience in batches])
	rewards = np.array([experience[1] for experience in batches])
	actions = np.array([experience[2] for experience in batches])
	new_states = np.array([experience[3] for experience in batches])
	dones = np.array([experience[4] for experience in batches])

	actions = oh.fit_transform(actions.reshape(-1,1)).toarray()
	actions = actions.reshape(-1,1,action_size) # to batch

	Qs =model.predict(states)
	new_Qs = model.predict(states)
	target_Qs=rewards.reshape(-1,1)+gamma*(np.max(new_Qs,axis=2)*(~dones.reshape(-1,1)))

	model.fit(states, Qs, verbose = 0)
	epsilon = epsilon*epsilon_decay

env.close()
"""


fig1 = plt.figure(figsize=(20,10))
ax1 = fig1.add_subplot(1,1,1)

ax1.plot(range(num_episodes), reward_list)
ax1.set_title("Rewards vs Episode")
ax1.set_xlabel("Episodes")
ax1.set_ylabel("Rewards")

plt.show()






