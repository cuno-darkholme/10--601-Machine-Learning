import numpy as np
import sys
import math


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
ti = dict((v,k) for k,v in tagIndex.items())

N = len(tagIndex)
V = len(wordIndex)
test_input = sys.argv[1]
predicted_file = sys.argv[7]
metric_file = sys.argv[8]
ow = open(predicted_file, "w")
om = open(metric_file, "w")


# read in file
openfile = open(test_input)
words = []
truetags = []
for lines in openfile:
    word = []
    truetag = []
    line = lines.strip('\n')
    lin = line.split(' ')
    for wt in lin:
        w, t = wt.split('_')
        word.append(w)
        truetag.append(t)
    words.append(word)
    truetags.append(truetag)
S = len(words)
#print(S)

hmmprior = sys.argv[4]
hmmemit = sys.argv[5]
hmmtrans = sys.argv[6]

Pi = np.genfromtxt(hmmprior, delimiter=' ')
A = np.genfromtxt(hmmtrans, delimiter=' ')
B = np.genfromtxt(hmmemit, delimiter=' ')
#print(Pi)

#predict
T = 0
alld = 0
trued = 0

for i in range(S):
    wordline = words[i]
    ttagline = truetags[i]
    T = len(wordline)
    viterbi = np.zeros((N,T))
    pointer = np.zeros((N,T))
    # init
    for s in range(N):
        pointer[s,0] = s
        viterbi[s,0] = np.log(Pi[s]) + np.log(B[s,wordIndex[wordline[0]]])
    for t in range(1,T):
        wd = wordIndex[wordline[t]]
        for s in range(N):
            prob = []
            for s1 in range(N):
                prob.append(viterbi[s1,t-1] + np.log(A[s1,s]) + np.log(B[s,wd]))
            viterbi[s,t] = np.max(prob)
            pointer[s,t] = np.argmax(prob)
    lp = np.argmax(viterbi[:,-1])
    path = [lp]
    for t2 in reversed(range(1,T)):
        lp = int(pointer[lp,t2])
        path.append(lp)
    path.reverse()
    # output
    for wo in range(len(path)):
        ow.write(wordline[wo]+ '_' + ti[path[wo]])
        if wo != (len(path)-1):
            ow.write(' ')
        alld += 1
        if ti[path[wo]] == ttagline[wo]:
            trued += 1
    ow.write('\n')
ow.close()
mer = trued/alld
om.write('Accuracy: %f'%mer)
om.write('\n')






