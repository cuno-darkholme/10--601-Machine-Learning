from __future__ import print_function
import sys

def featureEng(data , dictionary , flag):
    lines = dictionary.read().split('\n')
    dict = {}
    for line in lines[:-1]:
        split = line.split()
        dict[split[0]] = split[1]
    rawdata = data.read().split('\n')
    labels = []
    result = ""
    for datai in rawdata[:-1]:
        #print(datai.split("\t"))
        (label,words) = datai.split("\t")
        word = words.split()
        count = {}
        result += label 
        for attribute in word:
            if attribute in dict.keys():
                value = dict[attribute]
                if value not in count.keys():
                    count[value] = 1
                else:
                    count[value] += 1
        if int(flag) == 1:
            for x in count:
                result += "\t" + x + ":1"
        else:
            for x in count:
                if count[x] < 4:
                    result += "\t" + x + ":1" 
        result += "\n"
    return result


#sys.argv
if __name__ == '__main__':
    traininput = sys.argv[1]
    validationinput = sys.argv[2]
    testinput = sys.argv[3]
    dictinput = sys.argv[4]
    formattedtrainout = sys.argv[5]
    formattedvaliout = sys.argv[6]
    formattedtestout = sys.argv[7]
    flag = sys.argv[8]


dic = open(dictinput)
trainData = open(traininput)
valiData = open(validationinput)
testData = open(testinput)
train = featureEng(trainData , dic , flag)
dic = open(dictinput)
vali = featureEng(valiData,dic,flag)
dic = open(dictinput)
test = featureEng(testData,dic,flag)
trainformat = open(formattedtrainout,'w+')
trainformat.write(train)
trainformat.close()
validformat = open(formattedvaliout, 'w+')
validformat.write(vali)
validformat.close()
testformat = open(formattedtestout, 'w+')
testformat.write(test)
testformat.close()




