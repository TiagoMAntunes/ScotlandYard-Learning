import random

import numpy as np
import math
import sys
from random import choice
# LearningAgent to implement
# no knowledeg about the environment can be used
# the code should work even with another environment


class LearningAgent:

    # init
    # nS maximum number of states
    # nA maximum number of action per state
    def __init__(self, nS, nA):

        # define this function
        self.nS = nS
        self.nA = nA
        self.table = np.zeros((nS, nA)) # matrix for Q-learning algorithm (ignore index zero)
        self.frequencies = np.zeros((nS,nA)) #holds the frequencies of each state and action
        self.visited = [0 for _ in range(nS)]
        self.learning_rate = 0.7
        self.discount_rate = 0.9
        # define this function

    # Select one action, used when learning
    # st - is the current state
    # aa - is the set of possible actions
    # for a given state they are always given in the same order
    # returns
    # a - the index to the action in aa
    def selectactiontolearn(self, st, aa):
        """ Selects an action to take, while exploring unexplored ones """
        a = np.argmin(self.frequencies[st,:len(aa)])
        self.frequencies[st,a] += 1
        self.visited[st] = len(aa)
        return a

    # Select one action, used when evaluating
    # st - is the current state
    # aa - is the set of possible actions
    # for a given state they are always given in the same order
    # returns
    # a - the index to the action in aa
    def selectactiontoexecute(self, st, aa):
        #Selects best action out of what it has learned
        stateline = np.copy(self.table[st,:len(aa)]) #get actions values
        max_val = max(stateline) # maximum value in this line (action to be taken)
        
        #Randomly select the best ones
        validation = list(map(lambda x: 1 if x == max_val else 0, stateline))
        amount = list(validation).count(1)
        distribution = np.array(validation)
        distribution = np.true_divide(distribution, amount) #Distribution to randomly select
        
        #Randomly select the best action to take, out of the best
        a = np.random.choice(np.arange(len(aa)), p=distribution)
        
        a = np.argmax(self.table[st,:len(aa)])
        return a

    # this function is called after every action
    # st - original state
    # nst - next state
    # a - the index to the action taken
    # r - reward obtained
    def learn(self, ost, nst, a, r):
        # Q-learning formula
        self.table[ost, a] = self.table[ost,a] + self.learning_rate * (r + self.discount_rate * (max(self.table[nst,:self.visited[nst]]) if self.visited[nst] != 0 else 0) - self.table[ost,a])
        return
