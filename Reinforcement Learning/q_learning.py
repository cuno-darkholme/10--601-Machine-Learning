from environment import MountainCar
import sys
import numpy as np
from random import random
import copy


def maxq(s, w, b):
    maxvalue = -220
    index = 0
    for a in range(3):
        value = b
        wa = w[:, a]
        for x in s.keys():
            value += float(s[x]) * wa[x]
        if value > maxvalue:
            maxvalue = value
            index = a
    return maxvalue, index


def eg(qmax, epsi):
     p = random()
     if p > epsi:
         maxpick = qmax
     else:
         maxpick = np.random.choice([0, 1, 2])
     return maxpick


max_iterations = int(sys.argv[5])  # sys5
mode = sys.argv[1]  # sys1
episode = int(sys.argv[4])  # sys4
epsilon = float(sys.argv[6])  # sys6
learning_rate = float(sys.argv[8])  # sys8
gamma = float(sys.argv[7])  # sys 7
weight_out = sys.argv[2]
returns_out = sys.argv[3]

wo = open(sys.argv[2], "w+")
ro = open(sys.argv[3], "w+")


if mode == "raw":
    w = np.zeros((2, 3))
    b = 0
    env = MountainCar("raw")
else:
    w = np.zeros((2048, 3))
    b = 0
    env = MountainCar("tile")

# main loop
for episodeNum in range(episode):
    # reset state
    state = env.reset()  # state is a dictionary
    # find the q-value an the max q-value
    reward = 0
    for iter_time in range(max_iterations):
        (qMax, aMax) = maxq(state, w, b)
        #print (qMax)
        # epsilon-greedy action
        a_taken = eg(aMax, epsilon)
        wa_taken = w[:, a_taken]
        # calculate q
        q = b
        for key in state.keys():
            q += float(state[key]) * wa_taken[key]
        # next state
        (state2, r, flag) = env.step(a_taken)
        reward += r
        (qMax2, aMax2) = maxq(state2, w, b)
        # td
        td = learning_rate * (q - (r + (gamma * qMax2)))
        # update w and b
        b = b - td
        if mode == "raw":
            dw = np.zeros((2, 3))
        else:
            dw = np.zeros((2048, 3))
        for keys in state.keys():
            dw[keys][a_taken] = state[keys]
        #print(dw)
        w = w - (dw * td)
        state = copy.deepcopy(state2)
        if flag == True:
            break
    #print(reward)
    ro.write(str(reward))
    ro.write("\n")
#print(w)
#print(b)
#output
wo.write(str(b))
wo.write("\n")
for i in range(len(w)):
    for j in range(len(w[i])):
        wo.write(str(w[i][j]))
        wo.write("\n")











