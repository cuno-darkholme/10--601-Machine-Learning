import sys
import math

def dotProduct(x,w):
    product = 0.0
    for i, xi in x.items():
        product += float(xi) * float(w[int(i)])
    return product

def lendict(dict):
    lines = dict.read().split('\n')
    i = 0
    for word in lines[:-1]:
        i += 1
    return i

def sgd(theta, featureDict, label):
    step = 0.1
    dict = {}
    for words in featureDict:
        (key,value) = words.split(":")
        dict[key] = value
    dotProd = dotProduct(dict, theta)
    #print(dotProd)
    gradient = int(label) - (math.exp(dotProd)/(1+math.exp(dotProd)))
    for i in dict:
        theta[int(i)] += step * int(dict[i]) * gradient
    return theta

def train(list,traindata, dictionary, time):    
    rawdata = traindata.read().split('\n')
    for i in range (time):
        for lines in rawdata[:-1]:
            words = lines.split("\t")
            label = words.pop(0)
            words.insert(-1,'39176:1')
            #print(words)
            list = sgd(list, words, label)
    #print(list)
    return list
    
def test(theta, featureDict, label):
    dict = {}
    for words in featureDict:
        (key,value) = words.split(":")
        dict[key] = value
    total = 0
    for i in dict:
        total += theta[int(i)] * int(dict[i]) 
    #print(total)
    result = math.exp(total) / (math.exp(total) + 1)
    #print(result)
    return result

def output(theta, testdata):
    rawdata = testdata.read().split('\n')
    errcount = 0
    count = 0
    output = []
    for lines in rawdata[:-1]:
        words = lines.split("\t")
        label = int(words.pop(0))
        words.insert(-1,'39176:1')
        result = test(theta, words, label)
        #print(result)
        if result > 0.5:
            output.append("1")
            if int(label) == 0:
                errcount += 1
            else:
                count += 1
        else:
            output.append("0")
            if int(label) == 1:
                errcount += 1
            else:
                count += 1
    errrate = 0.000000
    errrate = errcount/(errcount+count)
    return ('%.6f'%errrate), output
    

#sys.argv
if __name__ == '__main__':
    ftraininput = sys.argv[1]
    fvalidationinput = sys.argv[2]
    ftestinput = sys.argv[3]
    dictinput = sys.argv[4]
    trainout = sys.argv[5]
    testout = sys.argv[6]
    merticsout = sys.argv[7]
    epoch = sys.argv[8]

dic = open(dictinput)
trainData = open(ftraininput,"r")
list = [0] * (lendict(dic)+1)
out1 = train(list,trainData,dic,int(epoch))
trainData.close()
trainData = open(ftraininput)
(e1,o1) = output(out1,trainData)
create = open(trainout, "w+")
for i in o1:
    create.write(i)
    create.write("\n")
testData = open(ftestinput,"r")
trainData = open(ftraininput,"r")
dic = open(dictinput)
out2 = train(list,trainData,dic,int(epoch))
(e2,o2) = output(out2,testData)
create2 = open(testout,"w")
for j in o2:
    create2.write(j)
    create2.write("\n")
metrics = open(merticsout, 'w+')
metrics.write('error(train): ' + e1 + '\n')
metrics.write('error(test): ' + e2)
metrics.close()



#print(o1)

"""

dic = open("dict.txt")
data = open("train.tsv")
list = [0] * (lendict(dic)+1)
a = train(list,data,dic,30)
out = open("train.tsv")
(e,o) = output(a,out)
print(e)
create = open("aaaa.tsv","w")
print(o)
print(type(o))
for i in o:
    create.write(o)

#print(a)
#print(o)

"""
