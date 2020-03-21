import numpy as np
import sys
import math

# read in the dic

def readIndex(file):
    dic = {}
    index = 0
    read = open(file)
    for lines in read:
        dic[lines.strip('\n')] = index
        index += 1
    return dic


index_to_word = sys.argv[2]
wordIndex = readIndex(index_to_word)
index_to_tag = sys.argv[3]
tagIndex = readIndex(index_to_tag)

N = len(tagIndex)
V = len(wordIndex)
train_input = sys.argv[1]


# read in file
openfile = open(train_input)
words = []
tags = []
for lines in openfile:
    word = []
    tag = []
    line = lines.strip('\n')
    lin = line.split(' ')
    for wt in lin:
        w, t = wt.split('_')
        word.append(w)
        tag.append(t)
    words.append(word)
    tags.append(tag)

A = np.ones((N,N))
B = np.ones((N,V))
Pi = np.ones((N,1))

for line1 in tags:
    Pi[tagIndex[line1[0]]] += 1
Pi = Pi/sum(Pi)

#print(Pi)

for linea in tags:
    for state in range(len(linea)-1):
        curstate = linea[state]
        nextstate = linea[state+1]
        A[tagIndex[curstate], tagIndex[nextstate]] += 1
A = A / np.sum(A, axis=1).reshape((-1, 1))

#print(A)

for lineb in range(len(tags)):
    for tagb in range(len(tags[lineb])):
        B[tagIndex[tags[lineb][tagb]],wordIndex[words[lineb][tagb]]] += 1
B = B / np.sum(B, axis=1).reshape((-1, 1))

#print(B)

#output

hmmprior = sys.argv[4]
hmmemit = sys.argv[5]
hmmtrans = sys.argv[6]

np.savetxt(hmmprior, Pi, delimiter=' ')
np.savetxt(hmmemit, B, delimiter=' ')
np.savetxt(hmmtrans, A, delimiter=' ')