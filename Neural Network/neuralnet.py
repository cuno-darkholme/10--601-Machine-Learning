import numpy as np
import math
import sys
import csv

# helper functions
# forward


def linerarForward(x, alpha):
    a = np.dot(alpha, x)
    return a


def sigmoidForward(a):
    p = []
    for i in a:
        p.append(1 / (1 + math.exp(-i)))
    q = np.array(p)
    q = q[:, None]
    return q


def softmaxForward(b):
    total = 0
    for i in b:
        total += math.exp(i)
    index = 0
    maxindex = 0
    result = 0
    max = 0
    yhat = []
    for i in b:
        result = math.exp(i) / total
        yhat.append(result)
        if result > max:
            maxindex = index
            max = result
        index += 1
    yHat = np.array(yhat)
    #print(maxindex)
    return yHat, maxindex


def crossEntropy(y,yhat):
    sum = 0
    for i in range(len(y)):
        sum -= y[i] * math.log(yhat[i])
    return sum

# backprop


def gbfff(y,yhat):
    dldb = []
    for i in range(len(y)):
        dldb.append(yhat[i]-y[i])
    dl_db = np.array(dldb)
    dl_db = dl_db[:,None]
    return dl_db


def gbetagz(gb, z, beta):
    dldbeta = np.dot(gb, np.transpose(z))
    bstar = np.delete(beta, 0, axis=1)
    #print(bstar.shape)
    dldz = np.dot(np.transpose(bstar), gb)
    return dldbeta, dldz


def gattt(gz, zstar):
    # note: this z does not have z0
    dlda = gz * zstar * (1-zstar)
    return dlda


def galphaeee(dlda, x):
    dldalpha = np.dot(dlda, np.transpose(x))
    return dldalpha


def entropy(data, alpha, beta, label):
    row = 0
    entropy = 0
    for lines in data:
        xf = list(map(float, lines))
        xf = np.array(xf)
        xf = xf[:, None]
        af = linerarForward(xf, alpha)
        zstarf = sigmoidForward(af)
        zf = np.insert(zstarf, 0 , [1.0])
        bf = linerarForward(zf, beta)
        (yhatf, maxindexf) = softmaxForward(bf)
        yf = [0] * 10
        truelabelf = int(label[row])
        yf[truelabelf] = 1
        yf = np.array(yf)
        row += 1
        entropy += crossEntropy(yf,yhatf)
    meanentropy = entropy / len(data)
    return meanentropy

def error(datapqa, alpha, beta, label):
    row = 0
    numerror = 0
    predict = []
    for lines in datapqa:
        xf = list(map(float,lines))
        xf = np.array(xf)
        xf = xf[:, None]
        af = linerarForward(xf, alpha)
        zstarf = sigmoidForward(af)
        zf = np.insert(zstarf, 0 , [1.0])
        bf = linerarForward(zf, beta)
        (yhatf, maxindexf) = softmaxForward(bf)
        #print(maxindexf)
        if int(label[row]) != maxindexf:
            numerror += 1
        predict.append(maxindexf)
        row += 1
    errrate = numerror/ len(datapqa)
    #print(numerror)
    #print(len(data))
    return predict, errrate

# main loop:


train_input = sys.argv[1]
test_input = sys.argv[2]
#train_input = "smallTrain.csv"
#test_input = "smallValidation.csv"

with open(train_input) as csvtrain:
    lines = csv.reader(csvtrain, delimiter = ',')
    trainlabels = []
    data = []
    attributes = 0
    for rows in lines:
        attributes = len(rows)
        label = rows.pop(0)
        trainlabels.append(label)
        rows.insert(0,1)
        data.append(rows)
    # print(labels)
    # print(data)

with open(test_input) as csvtest:
    line = csv.reader(csvtest, delimiter = ',')
    testlabels = []
    testdata = []
    for row in line:
        labelt = row.pop(0)
        testlabels.append(labelt)
        row.insert(0,1)
        testdata.append(row)

init_flag = int(sys.argv[8])
hidden_units= int(sys.argv[7])
#init_flag = 1
#hidden_units = 4
if init_flag == 2:
    alpha = np.zeros((hidden_units,attributes))
    beta = np.zeros((10, hidden_units+1))
else:
    alpha = np.random.uniform(-0.1,0.1,(hidden_units,attributes))
    beta = np.random.uniform(-0.1,0.1,(10,hidden_units+1))
    for i in alpha:
        i[0] = 0
    for j in beta:
        j[0] = 0

num_epoch = int(sys.argv[6])
learning_rate = float(sys.argv[9])
metrics = sys.argv[5]
mo = open(sys.argv[5],'w+')

#num_epoch = 2
#learning_rate = 0.1

trainen = []
testen = []

for times in range(num_epoch):
    k = 0
    for lines in data:
        x = list(map(float,lines))
        #print(x)
        x = np.array(x)
        # forward
        x = x[:, None]
        a = linerarForward(x,alpha)
        #print(x.shape)
        #print(a.shape)
        zstar = sigmoidForward(a)
        z = np.insert(zstar, 0 , [1.0])
        z = z[:, None]
        b = linerarForward(z,beta)
        #print(z.shape)
        #print(b.shape)
        (yhat,maxindex) = softmaxForward(b)
        y = [0] * 10
        truelabel = int(trainlabels[k])
        y[truelabel] = 1
        y = np.array(y)
        k += 1
        # backprop
        gba = gbfff(y,yhat)
        #print(gb.shape)
        (gbeta, gz) = gbetagz(gba, z, beta)
        #print(gbeta.shape)
        #print(beta.shape)
        #print(gz.shape)
        gak = gattt(gz, zstar)
        galphaq = galphaeee(gak,x)
        # change
        beta = beta - learning_rate * gbeta
        alpha = alpha - learning_rate * galphaq
    # mean entropy
    # given alpha and beta
    trainentro = entropy(data, alpha, beta, trainlabels)
    trainen.append(trainentro)
    #mo.write("epoch=" + str(times) + " crossentropy(train): " + str(trainentro) + "\n" )
    testentro = entropy(testdata, alpha, beta, testlabels)
    testen.append(testentro)
    #mo.write("epoch=" + str(times) + " crossentropy(test): " + str(testentro) + "\n")
    #print(trainen)
    #print(testen)
# error
ertrain, outtrain = error(data, alpha, beta, trainlabels)
ertest, outtest = error(testdata, alpha, beta, trainlabels)

################################
train_out = sys.argv[3]
to = open(sys.argv[3],'w+')
for i in ertrain:
    to.write(str(i)+'\n')

test_out = sys.argv[4]
tso = open(sys.argv[4],'w+')
for i in ertest:
    tso.write(str(i) + '\n')

for i in range(num_epoch):
    mo.write('epoch=' + str(i+1) + ' crossentropy(train): ' + str(trainen[i]) + '\n')
    mo.write('epoch=' + str(i+1) + ' crossentropy(test): ' + str(testen[i]) + '\n')


mo.write('error(train): ' + str(outtrain) + '\n')
mo.write('error(test): ' + str(outtest) + '\n')





















