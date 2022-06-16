# Q Learning tutorial
# From https://github.com/sezan92/RLTutorialKeras/blob/master/English/RL%20Tutorial%201/RL_Tutorial_1.ipynb

import gym
import numpy as np 

#https://github.com/openai/gym/wiki/MountainCar-v0
env = gym.make('MountainCar-v0')
s = env.reset()

#print(env, type(env), env.action_space.n)
#print(s, type(s), s.shape)

legal_actions=env.action_space.n
actions = [0,1,2]
gamma =0.99
lr =0.5
num_episodes =30000
epsilon =0.5
epsilon_decay =0.99


N_BINS = [10,10]

MIN_VALUES = [0.6,0.07]   # max pos, max vel
MAX_VALUES = [-1.2,-.07]  # min pos, min vel
BINS = [np.linspace(MIN_VALUES[i], MAX_VALUES[i], N_BINS[i]) for i in range(len(N_BINS))]
#print(BINS)
rList =[]


def discretize(obs):
       return tuple([int(np.digitize(obs[i], BINS[i])) for i in range(len(N_BINS))])


class QL:
    def __init__(self,Q,policy, legal_actions, actions, gamma, lr):
        self.Q = Q #Q matrix
        self.policy =policy
        self.legal_actions=legal_actions
        self.actions = actions
        self.gamma =gamma
        self.lr =lr

    def q_value(self,s,a):
        """Gets the Q value for a certain state and action"""
        if (s,a) in self.Q:
            self.Q[(s,a)]
        else:
            self.Q[s,a]=0
        return self.Q[s,a]

    def action(self,s):
        """Gets the action for cetain state"""
        if s in self.policy:
            return self.policy[s]
        else:
            self.policy[s] = np.random.randint(0,self.legal_actions)
            return self.policy[s]

    def learn(self,s,a,s1,r,done):
        """Updates the Q matrix"""
        if done== False:
            self.Q[(s,a)] =self.q_value(s,a)+ self.lr*(r+self.gamma*max([self.q_value(s1,a1) for a1 in self.actions]) - self.q_value(s,a))
        else:
            self.Q[(s,a)] =self.q_value(s,a)+ self.lr*(r - self.q_value(s,a))
        self.q_values = [self.q_value(s,a1) for a1 in self.actions]
        self.policy[s] = self.actions[self.q_values.index(max(self.q_values))]


Q = {}
policy = {}
legal_actions = 3  # number of legal actions
QL = QL(Q, policy, legal_actions, actions, gamma, lr)


for i in range(num_episodes):
    s_raw= env.reset() #initialize
    s = discretize(s_raw) #discretize the state
    rAll =0 #total reward
    d = False
    j = 0
    for j in range(200):
        
        #epsilon greedy. to choose random actions initially when Q is all zeros
        if np.random.random()< epsilon:
            a = np.random.randint(0,legal_actions)
            epsilon = epsilon*epsilon_decay
        else:
            a =QL.action(s)
        s1_raw,r,d, info = env.step(a)
        rAll=rAll+r
        s1 = discretize(s1_raw)
        env.render()
        if d:
            if rAll<-199:
                r =-100 #punishment, if the game finishes before reaching the goal , we can give punishment
                QL.learn(s,a,s1,r,d)
                print("Episode {}: Failed! Reward {}".format(i,rAll))
            elif rAll>-199:
                print("Episode {}: Passed! Reward {}".format(i,rAll))
                print(QL.Q)
                print(QL.policy)
            break
        QL.learn(s,a,s1,r,d)
        if j==199:
            print("Reward %d after full episode"%(rAll))
            
        s = s1

env.close()

