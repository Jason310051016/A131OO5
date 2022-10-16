from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from math import log
import operator
import numpy as np
import pandas as pd
from sklearn import tree


def createDataSet(dataframe):
    # Extract attribute name
    colsname = list(dataframe.columns)
    # Construct the dataset
    dataSet = np.array(dataframe)
    dataSet = dataSet.tolist()
    return dataSet, colsname

# Calculate information entropy
def InfoEnt(dataSet):
    DataNum = len(dataSet)
    # Create a dictionary for classification and count the number of data in each category
    LabelCounts = {}
    for data in dataSet:
        data_label = data[-1]
        LabelCounts[data_label] = LabelCounts.get(data_label, 0) + 1
    # Calculate information entropy
    info_ent = 0.0
    for label in LabelCounts:
        p_label = float(LabelCounts[label]) / DataNum
        info_ent += p_label * log(1 / p_label, 2)
    return info_ent

# Returns the data set with the number th attribute value as value
def SplitDataSet(dataSet, number, value):
    NewDataSet = []
    for data in dataSet:
        if data[number] == value:
            newdata = data[:number]
            newdata.extend(data[number + 1:])
            NewDataSet.append(newdata)
    return NewDataSet

# According to the criterion of max information gain, select the optimal attribute
def ChooseBestFeature(dataSet, colsname):
    # Statistical properties
    FeatureNum = len(colsname)
    # The information entropy of the initial situation, that is, the information entropy of all data when it is not divided
    BaseEnt = InfoEnt(dataSet)
    # Record the maximum information gain and its corresponding properties
    BestInfoGain = 0
    BestFeature = -1
    # Traverse each attribute, calculate the information gain, and find the maximum value
    for i in range(FeatureNum):
        # get all values of this property
        Values = [data[i] for data in dataSet]
        # Get all possible values of this property
        UniqualValues = set(Values)
        NewEnt = 0
        for value in UniqualValues:
            NewDataSet = SplitDataSet(dataSet, i, value)
            p_value = len(NewDataSet) / float(len(dataSet))
            NewEnt += p_value * InfoEnt(NewDataSet)
            # Calculate Information Gain
        InfoGain = BaseEnt - NewEnt
        # update maximum information gain
        if InfoGain > BestInfoGain:
            BestInfoGain = InfoGain
            BestFeature = colsname[i]
        # Returns the maximum information gain, the optimal attribute
    return BestInfoGain, BestFeature

# Select the class with the most labels as the leaf node decision result
def MaxNumLabel(LabelList):
    LabelCount = {}
    for label in LabelList:
        LabelCount[label] = LabelCount.get(label, 0) + 1
    # Sort according to the value of (key, value)
    SortedLabelCount = sorted(LabelCount.items(), key=operator.itemgetter(1), reverse=True)
    # Returns the key with the most occurrences
    return SortedLabelCount[0][0]

def CreateTree(dataSet, colsname):
    LabelList = [data[-1] for data in dataSet]
    colsname = colsname.copy()
    # When the LabelList is the same label
    if LabelList.count(LabelList[0]) == len(LabelList):
        return LabelList[0]
    # After traversing all eigenvalues, return the one with the largest number
    if (len(dataSet[0]) == 1):
        return MaxNumLabel(LabelList)
    # Obtain optimal partitioning properties and maximum information gain
    BestInfoGain, BestFeature = ChooseBestFeature(dataSet, colsname)
    # Index BestFeature_id to locate the best split attribute
    BestFeature_id = 0
    for i in range(len(colsname)):
        if colsname[i] == BestFeature:
            BestFeature_id = i
            break
    # Construct a decision tree

    myTree = {BestFeature: {}}

    # Eliminate the best partition attribute BestFeature from the attribute set
    del (colsname[BestFeature_id])
    # The value of the optimal partition attribute
    Values = [data[BestFeature_id] for data in dataSet]

    # All possible values of the optimal partitioning attribute
    UniqueValues = set(Values)
    for value in UniqueValues:
        NewLabels = colsname[:]
        # Recursive call to create decision tree
        myTree[BestFeature][value] = CreateTree(SplitDataSet(dataSet, BestFeature_id, value), NewLabels)
    return myTree


def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    print(firstStr)
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel


data1 = pd.read_csv('Iris.csv')
data=data1.iloc[:145]
data_test=data1.iloc[145:]
print(data_test.iloc[0:1])
dataSet, colsname = createDataSet(data)
trees = CreateTree(dataSet, colsname)
print('decision tree:', trees)


def test(tree,colsname,test_data,score):
    precisionDict = {'Iris-setosa': 0, 'Iris-versicolor': 0, 'Iris-virginica': 0}

    for i in range(test_data.shape[0]-1):
        value1 = np.array(data_test[i:i+1])
        value = [int(value1[0][0]), int(value1[0][1]), int(value1[0][2]), int(value1[0][3])]
        test_pre = classify(tree, colsname, value)
        print(label(test_pre))
        print(label(value1[0][4]))

        score = score+accuracy_score(label(test_pre), label(value1[0][4]))
    #     test_pre=classify(tree,colsname,test_data[i:i+1])
    #     test_pre=np.array(test_pre).reshape(1)
    #     test=np.array(test_data[4]).reshape(1)
    #     score = accuracy_score(test, test_pre)
    return score/test_data.shape[0]
def label(data):
    if data=='Iris-setosa':
        ans=[1,0,0]
    if data == 'Iris-versicolor':
        ans = [0, 1, 0]
    if data == 'Iris-virginica':
        ans = [0, 0, 1]
    return ans

print("accuracy:",test(trees,colsname,data_test,0))
