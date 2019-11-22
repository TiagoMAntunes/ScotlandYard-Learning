import random

import numpy as np
import math
# LearningAgent to implement
# no knowledeg about the environment can be used
# the code should work even with another environment


class LearningAgent:

    # init
    # nS maximum number of states
    # nA maximum number of action per state
    def __init__(self, nS, nA):

        # define this function
        self.nS = nS + 1
        self.nA = nA + 1
        self.table = np.zeros((nS+1, nA+1)) # matrix for Q-learning algorithm (ignore index zero)
        # define this function

    # Select one action, used when learning
    # st - is the current state
    # aa - is the set of possible actions
    # for a given state they are always given in the same order
    # returns
    # a - the index to the action in aa
    def selectactiontolearn(self, st, aa):
        # define this function
        # print("select one action to learn better")

        stateline = np.copy(self.table[st,])
        for i in range(self.nA):
                if i not in aa:
                        stateline[i] = -math.inf # will never select not given states
        max_val = max(stateline) # maximum value in this line (action to be taken)

        #Get distribution of this line (can be multiple best movements)
        validation = list(map(lambda x: 1 if x == max_val else 0, stateline))
        amount = list(validation).count(1)
        distribution = np.array(validation)
        distribution = np.true_divide(distribution, amount) #Distribution to randomly select
        
        #Randomly select the best line to pick
        a = np.random.choice(np.arange(self.nA), p=distribution)
        
        # define this function
        return a

    # Select one action, used when evaluating
    # st - is the current state
    # aa - is the set of possible actions
    # for a given state they are always given in the same order
    # returns
    # a - the index to the action in aa
    def selectactiontoexecute(self, st, aa):
        # define this function
        a = 0
        # print("select one action to see if I learned")
        return a

    # this function is called after every action
    # st - original state
    # nst - next state
    # a - the index to the action taken
    # r - reward obtained
    def learn(self, ost, nst, a, r):
        # define this function
        #print("learn something from this data")

        return

a = LearningAgent(5,10)
a.table[2,4] = 5
a.table[2,7] = 5

for _ in range(100):
        print(a.selectactiontolearn(2,[2,3,4,5,6,7,10]))