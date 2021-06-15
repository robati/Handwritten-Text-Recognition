from PIL import Image ,ImageFilter
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import numpy as np
import csv
import random
import math
import copy
import operator

def incidenceCount(array):
    zeroFlag = False
    iCount = 0
    for i in range(0, len(array)):
        if (zeroFlag):
            if (array[i] == 1):
                zeroFlag = False
        elif (array[i] == 0):
            zeroFlag = True
            iCount += 1
    if (iCount > 2):
        iCount = 2
    return (iCount)


def get4tuple(np_img, x, y):
    newTuple = [0, 0, 0, 0]
    # print x,y #,np_img[y,x]
    right_cells = np_img[y, x:]
    # print right_cells
    # print incidenceCount(right_cells)
    newTuple[0] = incidenceCount(right_cells)
    left_cells = np_img[y, :x]
    # print left_cells
    # print incidenceCount(left_cells)
    newTuple[2] = incidenceCount(left_cells)
    top_cells = np_img[:y, x]
    # print top_cells
    # print incidenceCount(top_cells)
    newTuple[1] = incidenceCount(top_cells)
    bottom_cells = np_img[y:, x]
    # print bottom_cells
    # print incidenceCount(bottom_cells)
    newTuple[3] = incidenceCount(bottom_cells)
    # print "\n"
    return (newTuple)


def convert4TupleToDec(tuple):
    sum = 0
    for i in range(0, 4):
        sum += pow(3, 3 - i) * tuple[i]
    return sum



def mainTest(file):
    img = Image.open(file).convert('L')

    np_img = np.array(img)
    np_img = ~np_img  
    np_img[np_img > 128] = 1 #black
    # np.savetxt('test'+file+'.txt', np_img,fmt='%1.1d', delimiter=' ')
    ft_img = 81 * [0]
    for x in range(0, np_img[0].size):
        for y in range(0, len(np_img)):
            if (np_img[y, x]):
                tuple4 = get4tuple(np_img, x, y)
                code = convert4TupleToDec(tuple4)
                ft_img[code] += 1
   # print(ft_img)
    return (ft_img)


def readAllFileInFolder():
    import pathlib
    import csv

    currentDirectory = pathlib.Path('.')
    ds = []
    i = 0
    currentPattern = "*.jpg"
    for currentFile in currentDirectory.glob(currentPattern):
        # print(currentFile)
        file = str(currentFile).split(".")[0]
        # print(file)
        a = []

        a += mainTest(str(currentFile))
        a += [file]
        ds.append(a)
        # ds[i].append(file)
        i += 1
    csvData = ds

    with open('person.csv', 'w') as csvFile:
        writer = csv.writer(csvFile, lineterminator='\n')
        for row in csvData:
            writer.writerow(row)
    csvFile.close()


class MLClassifier(object):
    def __init__(self, x_train):
       # print(x_train)
        self.x_train = x_train
        self.n_samples = x_train.shape[1]

        self.n_features = x_train.shape[0]
        cov_m = np.cov(x_train)

        #(sign, logdet) = np.linalg.slogdet(cov_m)
        #self.det_cov= sign * np.exp(logdet)
        self.det = np.linalg.det(cov_m)

        self.mu = x_train.mean(axis=1)

        self.label = ""

        if(self.det==0):
            print("singularMatrix")
        else:
            self.i_cov_m = np.linalg.inv(cov_m)
        if(self.det>0):
            self.constant_term = -0.5 * np.log(self.det)  # - \
            #   self.n_features * 0.5 * np.log(2. * np.pi)
        else:
            self.constant_term=0


    def setLabel(self,label):
        #print(label)
        self.label=label
    def getLabel(self):
        return  self.label
    def classify(self, x_test):

        if (self.det == 0):
            return -1
        s = x_test - self.mu
        #log_prob = -  0.5 * \
        #           (np.dot(s, self.i_cov_m) * s).sum(1)
        a = np.dot(s, self.i_cov_m)
        b=(a * s)
        c=b.sum(0)

        c=-  0.5 * np.dot(np.dot(np.transpose(s), self.i_cov_m), s)


        prob = c +self.constant_term
        #print(np.sum(np.dot(s,self.i_cov_m) *s, axis = 0))
        return prob



    #clf = MLClassifier(np.array([a1,a2, a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16]).T)
    # probs = clf.classify ( test)

def loadDataset(filename, split, trainingSet=[], testSet=[],):
    with open(filename) as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines) # [['5.1', '3.5', '1.4', '0.2', 'Iris-setosa'], ['4.9', '3.0', '1.4', '0.2', 'Iris-setosa'],...]
        for x in range(len(dataset)):

            for y in range(len(dataset[0])-1):#excluding the last item(label)
                dataset[x][y] = float(dataset[x][y])

            if random.random() < split:
                trainingSet.append(dataset[x]) #[[5.1, 3.5, 1.4, 0.2, 'Iris-setosa'], [4.9, 3.0, 1.4, 0.2, 'Iris-setosa'],
            else:
                testSet.append(dataset[x])


def normalized(ds,featureCount):
    data=[i[:-1] for i in ds]
    X = data#StandardScaler().fit_transform(data)
    n=np.array(X)
    normalds=[]
    svd = TruncatedSVD(n_components=featureCount)
    a=[]
    Y=svd.fit_transform(X)
    for i in range(Y.shape[0]):
        normalds.append([Y[i],ds[i][-1]])
        np.append(normalds[i],ds[i][-1])#unused

    return normalds #[[[0, 1, 2, 3], 'Iris-setosa'],...]


def getPrimeAttrSet(item,k): # [[0.2, 3], [1.4, 2], [3.5, 1], [5.1, 0], ['Iris-setosa', 4]]

    list=set()
    # returns k biggest members of the list indexes into a set
    for i in range(len(item)-2,len(item)-2-k,-1):
        list.add(item[i][1])
    return list #{0, 1, 2, 3}

def createPrimeAttrSet(trainingSet,n): ##[[5.1, 3.5, 1.4, 0.2, 'Iris-setosa'],...]
    ds = sortAttributesBySize(trainingSet) #[[[0.2, 3], [1.4, 2], [3.5, 1], [5.1, 0], ['Iris-setosa', 4]],...]
    trainingSet2=[]
    for item in ds:
        a = [set(), "label"]
        a[0] = list(getPrimeAttrSet(item,n))     #{0, 1, 2, 3}
        a[1] = item[-1][0]                  #'Iris-setosa'
        trainingSet2.append(a)
    return trainingSet2 #[[{0, 1, 2, 3}, 'Iris-setosa'], [{0, 1, 2, 3}, 'Iris-setosa'],..]


def createPrimeAttrList(trainingSet, n): #[[5.1, 3.5, 1.4, 0.2, 'Iris-setosa'],..]
    ds = sortAttributesBySize(trainingSet) ##[[[0.2, 3], [1.4, 2], [3.5, 1], [5.1, 0], ['Iris-setosa', 4]],...]
    trainingSet2 = []
    for item in ds:
        a = [set(), "label"]
        a[0] = getPrimeAttrList(item, n);
        a[1] = item[-1][0]
        trainingSet2.append(a)
    return trainingSet2 #[[[0, 1, 2, 3], 'Iris-setosa'],...]

def getPrimeAttrList(item, k): # [[0.2, 3], [1.4, 2], [3.5, 1], [5.1, 0], ['Iris-setosa', 4]]
    list = []
    # returns biggest members of the list indexes into a list (ordered)
    for i in range(len(item) - 2, len(item) - 2 - k, -1):
        list.append(item[i][1])
    return list #[0, 2, 1, 3]

def sortAttributesBySize(data):
    ds=copy.deepcopy(data)                  #[[4.6, 3.1, 1.5, 0.2, 'Iris-setosa'],...]
    for item in ds:
        #set index for each attribute
        for i in range(len(item)):
            item[i]=[item[i],i]             #item=[[0.2, 0], [1.4, 1], [3.5, 2], [5.1, 3], ['Iris-setosa', 4]]
        #sorting attributes in each item
        for j in range(len(item)-1):        #exclude label
            for i in range(len(item)-2):    #bubble sort goes until len-1
                if(item[i][0]>item[i+1][0]):
                    temp=item[i+1]
                    item[i + 1]=item[i]
                    item[i]=temp
    return ds #[[[0.2, 3], [1.4, 2], [3.5, 1], [5.1, 0], ['Iris-setosa', 4]], [[0.2, 3], [1.4, 2], [3.0, 1], [4.9, 0], ['Iris-setosa', 4]],...]


def testNtrain(trainSet,testSet2,displayCM=False):
# sort by class
    classVotes={}
    values=[]
    i=0

    for item in trainSet:
        response=item[-1]#label
        if response not in classVotes:
            classVotes[response]=i
            values.append([item[0]])
            i += 1
        else:
            values[classVotes[response]].append(item[0])
#training
    clf_al=[]
    for item in classVotes:
        a = values[classVotes[item]]
       # print(len(a), len(a[0]))
        array1 = np.array([np.array(xi) for xi in a])
        #print('j2', array1)
        clf=MLClassifier(array1.T)
        clf.setLabel(item)
        clf_al.append( clf)

#test

    prediction=[]
    for item in testSet2:
        x = np.array(item[0])#[list([4, 13, 22, 3, 28, 5, 7, 45]) 'Ain']
        i = 0
        results = list()
        for clf in clf_al:
            results.append((clf.classify(x), clf.getLabel()))
        results.sort(key=operator.itemgetter(0), reverse=True)
        prediction.append(results[0][-1])

    true=0
    confMx=[]
    for i in range(len(classVotes)):
        confMx.append([0]*len(classVotes))
        
    for i in range(len(testSet2)):
        if(prediction[i]==testSet2[i][-1]):
            true+=1
        x=classVotes[prediction[i]]
        y=classVotes[testSet2[i][-1]]
        confMx[x][y]+=1

    if(displayCM):
        showMatrix(confMx)

    print("presicion ",true/float(len(testSet2)))
    deghat=true/float(len(testSet2))

    return deghat
def showMatrix(confMx):
    for line in confMx:
        print(line)

def drawSet(fCount,iterCount,split,showCM=False):
    trainingSet = []
    testSet = []
    deghat_Array_f6=[]
    #split = 0.67

    for i in range(iterCount):

        loadDataset('./letter/person.csv', split, trainingSet, testSet)

        trainSet = normalized(trainingSet, fCount)
        testSet2 = normalized(trainingSet, fCount)
        deghat_Array_f6.append(testNtrain(trainSet, testSet2,showCM))
    lines = plt.plot(deghat_Array_f6,label=fCount)
    print(fCount,deghat_Array_f6,sum(deghat_Array_f6)/float(iterCount))
    plt.legend(loc='upper left')

def main():
    print("starting")
    trainingSet = []
    testSet = []

    split = 0.67

    featureCount=11



    loadDataset('./out.csv', split, trainingSet, testSet)

    trainSet=normalized(trainingSet, featureCount)
    testSet2=normalized(trainingSet, featureCount)
    testNtrain(trainSet, testSet2,True)



    #trainSet=createPrimeAttrSet(trainingSet,featureCount)
    #testSet2 = createPrimeAttrSet(testSet, featureCount)
    #testNtrain(trainSet, testSet2);

    # trainSet = createPrimeAttrList(trainingSet, featureCount)
    # testSet2 = createPrimeAttrList(testSet, featureCount)
    # testNtrain(trainSet, testSet2);

    

#readAllFileInFolder()
main()
