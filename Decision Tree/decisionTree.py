from __future__ import print_function
import sys
import csv
import math
import copy
    
def entro(data):
    num = len(data)
    pos = data[0]
    pos1 = 0
    for i in range (num):
        if data[i] == pos:
            pos1 += 1
    pos2 = num - pos1
    pos1Pes = pos1 / num
    pos2Pes = pos2 / num
    if pos1Pes > 0 and pos2Pes > 0:
        entro = -(pos1Pes * math.log(pos1Pes,2) + pos2Pes * math.log(pos2Pes,2))
    elif pos1Pes > 0 and pos2Pes <= 0:
        entro = -(pos1Pes * math.log(pos1Pes,2))
    elif pos1Pes <= 0 and pos2Pes > 0:
        entro = -(pos2Pes * math.log(pos2Pes,2))
    else:
        entro = 1
    return entro
    
def infGain(data1, data2):
    infGain = entro(data1) - entro(data2)
    return infGain
    
def attInfGain(dataset , attributeNum):
    attributeName = dataset[0][attributeNum]
    attributeData = []
    result = []
    for i in range (len(dataset)-1):
        attributeData.append(dataset[i+1][attributeNum])
        result.append(dataset[i+1][-1])
    if len(attributeData) == 0:
        return 0
    else:
        attPos = attributeData[0]
        data1 = []
        data2 = []
        for j in range (len(dataset)-1):
            if attributeData[j] == attPos:
                data1.append(result[j])
            else:
                data2.append(result[j])
        entropy1 = entro(data1)
        if len(data2) != 0:
            entropy2 = entro(data2)
        else:
            entropy2 = 0
        pes1 = len(data1)/(len(data1)+len(data2))
        pes2 = len(data2)/(len(data1)+len(data2))
        totalEntropy = (entropy1 * pes1) + (entropy2 * pes2)
        rawEntropy = entro(result)
        infGainAtt = rawEntropy - totalEntropy
        if infGainAtt >= 0:
            return infGainAtt
        else:
            return None


def stump(trainData):
    maxGain = 0
    maxGainIndex = 0
    for i in range (len(trainData[0])-1):
        gain = attInfGain(trainData , i)
        if gain > maxGain:
            maxGain = gain
            maxGainIndex = i
    if len(trainData)>0 and len(trainData[0]) > maxGainIndex:
        name = trainData[0][maxGainIndex]
    else:
        name = "?"
    if maxGain > 0:
        return maxGain, maxGainIndex, name
    else:
        return 0 , None, "?"
 
def classCounts(rows):
    counts = {}  
    for i in range(len(rows)-1):
        label = rows[i+1][-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


def majorityClass (data):
    list = []
    for i in range (len(data)-1):
        list.append(data[i+1][-1])
    if len(list) == 0:
        return list[0]
    else:
        result1 = list[0]
        result2 = ""
        for i in range (len(list)):
            if list[i] != list[0]:
                result2 = list[i]
        class1 = list.count(result1)
        class2 = list.count(result2)
        if class1 >= class2:
            return result1
        else:
            return result2 

def subSet(trainData, attIndex):
    newData = copy.deepcopy(trainData)
    leftValue = trainData[1][attIndex]
    for i in range (len(trainData)-1):
        if trainData[1+i][attIndex] != leftValue:
            rightValue = trainData[1+i][attIndex]
    leftDataSet = []
    leftDataSet.append(newData[0])
    rightDataSet = []
    rightDataSet.append(newData[0])
    #print(leftDataSet)
    for i in range(len(newData)-1):
        if newData[i+1][attIndex] == leftValue:
            leftDataSet.append(newData[i+1])
        else:
            rightDataSet.append(newData[i+1])
    #print(leftDataSet)
    lSet = copy.deepcopy(leftDataSet)
    rSet = copy.deepcopy(rightDataSet)
    for row in lSet:
        row.pop(attIndex)
    for row in rSet:
        row.pop(attIndex)
    #print(lSet)
    #print(rSet)
    #print(lSet , rSet , leftValue, rightValue)
    return lSet , rSet , leftValue, rightValue
    
class Leaf:
    def __init__(self,data):
        self.predictions = majorityClass(data)
        self.counts = classCounts (data)
        
class Node:
    def __init__(self, feature, depth, leftchild, rightchild, leftvalue, rightvalue, name, classcount):
        self.feature = feature
        self.depth = depth
        self.leftchild = leftchild
        self.rightchild = rightchild
        self.leftvalue = leftvalue
        self.rightvalue = rightvalue
        self.name = name
        self.classcount = classcount


def createTree(trainData, maxDepth, depth = 0):
    a = maxDepth
    if depth >= maxDepth:
        return Leaf(trainData)
    if len(trainData) <= 1:
        return Leaf(trainData)
    cc = classCounts(trainData)
    mostGain, attribute, attributeName = stump(trainData)
    if mostGain <= 0:
        return Leaf(trainData)
    leftData , rightData , leftvalue , rightvalue = subSet(trainData , attribute)
    #print(leftData)
    #print(rightData)
    leftchild = createTree(leftData, a , depth+1)
    rightchild = createTree(rightData, a , depth+1)
    return Node(attribute, depth, leftchild, rightchild, leftvalue, rightvalue, attributeName, cc)
    
def rowPrediction(row, tree, dit):
    if isinstance(tree,Leaf):
        return tree.predictions
    #print (row[tree.feature],tree.leftvalue)
    if row[dit[tree.name]] == tree.leftvalue:
        return rowPrediction(row, tree.leftchild , dit)
    if row[dit[tree.name]] == tree.rightvalue:
        return rowPrediction(row, tree.rightchild , dit)

def prediction(data, tree, dit):
    lenth = len(data)
    list = []
    for i in range(lenth-1):
        #print("\n")
        #print (rowPrediction(data[i+1],tree))
        list.append(rowPrediction(data[i+1],tree,dit))
    #print (list)
    return list
    
def mertics(trainData , testData , tree , dit):
    numData = len(trainData) - 1
    trainError = 0
    testError = 0
    result = prediction(trainData , tree , dit)
    result2 = prediction(testData , tree , dit)
    for i in range(numData - 1):
        if result[i] != trainData[i+1][-1]:
            trainError += 1
    for j in range(len(testData) - 1):
        if result2[j] != testData[j+1][-1]:
            testError += 1
    trainP = trainError / numData
    testP = testError / (len(testData)-1)
    #print (trainP)
    #print (testP)
    out3 = open(merticsout,"w+")
    out3.write("error(train): " + str(trainP))
    out3.write("\n")
    out3.write("error(test): " + str(testP))
    return trainP , testP
    

def printTree(tree , dash="|"):
    if isinstance(tree, Leaf):
        print (tree.counts)
        return
    print (dash + str(tree.name) + " = " + str(tree.leftvalue) + " :", end = '' )
    if not isinstance(tree.leftchild, Leaf):
        print (tree.leftchild.classcount)
    printTree(tree.leftchild, dash + "|")
    print (dash + str(tree.name) + " = " + str(tree.rightvalue) + " :", end = '')
    if not isinstance(tree.rightchild, Leaf):
        print (tree.rightchild.classcount)
    printTree(tree.rightchild, dash + "|")


def finalPrint(rows , tree):
    counts = {} 
    for i in range(len(rows)-1):
        label = rows[i+1][-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    print (counts)
    printTree(tree)
    
def writeLabel(trainData , testData , tree , dit):
    out1 = open(trainout,"w+")
    out2 = open(testout,"w+")
    list1 = prediction(trainData , tree , dit)
    list2 = prediction(testData , tree , dit)
    for e in list1:
        out1.write(e)
        out1.write("\n")
    for k in list2:
        out2.write(k)
        out2.write("\n")
    return

#sys.argv
if __name__ == '__main__':
    traininput = sys.argv[1]
    testinput = sys.argv[2]
    maxdepth = sys.argv[3]
    trainout = sys.argv[4]
    testout = sys.argv[5]
    merticsout = sys.argv[6]
    print("The train input file is: %s" % (traininput)) 
    print("The test input file is: %s" % (testinput))
    print("The max depth is: %s" % (maxdepth)) 
    print("The train out file is: %s" % (trainout)) 
    print("The test out file is: %s" % (testout)) 
    print("The mertics out file is: %s" % (merticsout)) 

data = []
#read the data from the tsv file to a list
with open(traininput) as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    dit = {}
    for row in reader:
        data.append(row)
    for i in range(len(data[0])):
        dit[data[0][i]] = i
       
test = []
#read the data from the tsv file to a list
with open(testinput) as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    for row in reader:
        test.append(row)


k = int(maxdepth)


tree1 = createTree(data,k,)
prediction (test,tree1,dit)
mertics (data , test , tree1 , dit)
finalPrint (data , tree1)
writeLabel(data , test , tree1 , dit)




    
