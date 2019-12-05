#grupo 2 Tiago Antunes 89545 Mariana Oliveira 89504

import random

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
        self.table = [[0 for _ in range(nA)] for _ in range(nS)]
        self.frequencies = [[0 for _ in range(nA)] for _ in range(nS)]
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
        a = self.frequencies[st].index(min(self.frequencies[st][:len(aa)]))
        self.visited[st] = max(self.visited[st], len(aa)) # Number of maximum states that were decided to or not to be learn at this moment
        return a

    # Select one action, used when evaluating
    # st - is the current state
    # aa - is the set of possible actions
    # for a given state they are always given in the same order
    # returns
    # a - the index to the action in aa
    def selectactiontoexecute(self, st, aa):
        #Selects best action out of what it has learned
        stateline = [] + self.table[st][:min(len(aa), self.visited[st])] #get actions values from the learning performed
        max_val = max(stateline) # maximum value in this line (action to be taken)
        
        #Randomly select the best ones
        validation = list(map(lambda x: 1 if x == max_val else 0, stateline))
        vals = []
        for i, val in enumerate(stateline):
            if val == max_val:
                vals.append(i)
        
        #Randomly select the best action to take, out of the best
        a = random.randint(0, len(vals) - 1)
        return vals[a]

    # this function is called after every action
    # st - original state
    # nst - next state
    # a - the index to the action taken
    # r - reward obtained
    def learn(self, ost, nst, a, r):
        # Q-learning formula
        self.table[ost][a] = self.table[ost][a] + self.learning_rate * (r + self.discount_rate * (max(self.table[nst][:self.visited[nst]]) if self.visited[nst] != 0 else 0) - self.table[ost][a])
        self.frequencies[ost][a] += 1 #Visited action, increment frequency
        return
