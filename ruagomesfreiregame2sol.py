import random

import numpy as np
import math
from scipy.special import softmax

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
        self.time = 1
        self.learning_rate = 0.15
        self.discount_rate = 0.85
        self.number_tries = 7 #arbitrary value
        # define this function

    # Select one action, used when learning
    # st - is the current state
    # aa - is the set of possible actions
    # for a given state they are always given in the same order
    # returns
    # a - the index to the action in aa
    def selectactiontolearn(self, st, aa):
        """ Selects an action to take, while exploring unexplored ones """
        probs = softmax(np.copy(self.table[st,:len(aa)]))
        a = np.random.choice(np.arange(len(aa)), p=probs)

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
        
        return a

    # this function is called after every action
    # st - original state
    # nst - next state
    # a - the index to the action taken
    # r - reward obtained
    def learn(self, ost, nst, a, r):
        # Q-learning formula
        self.table[ost, a] = self.table[ost,a] + self.learning_rate * (r + self.discount_rate * max(self.table[nst,]) - self.table[ost,a])
        return
