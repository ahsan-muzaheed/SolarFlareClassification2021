import json
import pandas as pd
from datetime import datetime
import pickle

import numpy as np
from numpy import array


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, adjusted_rand_score,classification_report
from sklearn.preprocessing import StandardScaler


import random
import matplotlib.pyplot as plt

torch.manual_seed(1)


# input_dim = 33, hidden_dim = 128, num_TS = 60, num_classes = 4
class LSTM_MVTS_LRN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(LSTM_MVTS_LRN, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.hidden2class = nn.Linear(hidden_dim, num_classes)

    def forward(self, mvts):
        lstm_out, _ = self.lstm(mvts.view(len(mvts), 1, -1))  
        last_lstm_out = lstm_out[len(lstm_out) - 1]  
        class_space = self.hidden2class(last_lstm_out)  
        class_scores = F.log_softmax(class_space, dim=1)
        return class_scores



def loadInputs(file_name):
        with open(file_name, 'rb') as fp:
            obj = pickle.load(fp)
        return obj

def doClassSpecificCalulcation(Accuracy,trainLebel,classification_report_dict):
  print('\np.mean(Accuracy) :',np.mean(Accuracy))
  print('\np.std(Accuracy) :',np.std(Accuracy))
  print('\n33333333 p.mean np.std(Accuracy) :     ',np.round(np.mean(Accuracy),2),"+-",np.round(np.std(Accuracy),2) )
  for j in range( len(np.unique(trainLebel)) ):
    print('\n\n\n\nclass :',j)
    precision=[]
    recall=[]
    f1_score=[]
    for i in range(len(classification_report_dict)):
      report=classification_report_dict[i]
      temp=report[str(j)]['precision']
      precision.append(temp)

      temp=report[str(j)]['recall']
      recall.append(temp)

      temp=report[str(j)]['f1-score']
      f1_score.append(temp)

    print('\np.mean(precision) \t p.mean(recall) \t p.mean(f1_score) :')


    print(np.mean(precision))
    print(np.mean(recall))
    print(np.mean(f1_score))

    print('\np.mean p.std(precision) \tp.mean  p.std(recall) \tp.mean  p.std(f1_score) :')

    print(np.round(np.mean(precision),2),"+-",np.round(np.std(precision),2) )
    print(np.round(np.mean(recall),2),"+-",np.round(np.std(recall),2) )
    print(np.round(np.mean(f1_score),2),"+-",np.round(np.std(f1_score),2) )


def startCalculations( X_train, X_test, y_train, y_test,HIDDEN_DIM,num_masterIteration,numEpochs):

    classification_report_dict=[]
    Accuracy=[]
    for masterIteration in range(num_masterIteration):
        print("\n masterIteration HIDDEN_DIM : ",masterIteration, HIDDEN_DIM)

        model = LSTM_MVTS_LRN(INPUT_DIM,HIDDEN_DIM,NUM_CLASSES)
        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        numTrain = X_train.shape[0]

        #train
        for epoch in range(numEpochs):
          print("\n nmasterIteration, epoch: ",masterIteration,epoch)
          loss_values = []
          running_loss = 0.0

          for i in range(numTrain):
            model.zero_grad()
            mvts = X_train[i,:,:]
            mvts = torch.from_numpy(mvts).float()
            target = y_train[i]
            target = [target]
            target=np.array(target)
            target = torch.Tensor(target)
            target = target.type(torch.LongTensor)
            mvts = mvts.to(device)
            target = target.to(device)
            mvts = mvts.view(mvts.size(0), -1)
            model.to(device)
            class_scores = model(mvts)
            loss = loss_function(class_scores, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loss_values.append(running_loss / len(X_train))

          maxAcc=0
          max_classification_report_dict=0

          #test
          numTest = X_test.shape[0]
          with torch.no_grad():
              numCorrect = 0
              testLabel=[]
              predictaedLabel=[]
              for i in range(numTest):
                test_mvts = X_test[i,:,:]
                test_label = y_test[i] 
                test_mvts = torch.from_numpy(test_mvts).float()
                test_mvts = test_mvts.to(device)
                test_class_scores = model(test_mvts) 
                class_prediction = torch.argmax(test_class_scores, dim=-1) 
                current_seq = np.argmax(test_class_scores.cpu().numpy())
                testLabel.append(test_label)
                predictaedLabel.append(current_seq)
                if(class_prediction == test_label): #(2,3 ) match
                  numCorrect = numCorrect+1
              acc = numCorrect/numTest
              fgdg=round(acc, 2)
              if fgdg  > maxAcc:
                maxAcc=acc
                print( "maxAcc:" ,maxAcc)
                max_classification_report_dict=metrics.classification_report(testLabel, predictaedLabel, digits=3,output_dict=True)

        plt.plot(np.array(loss_values), 'r')
        classification_report_dict.append(max_classification_report_dict)
        Accuracy.append(maxAcc)
    doClassSpecificCalulcation(Accuracy,trainLebel,classification_report_dict)


def start(test_sizes,HIDDEN_DIMs,num_masterIteration,numEpochs):

    for temp4 in range(len(test_sizes)):
        test_size = test_sizes[temp4]
        print("\n\n\n *************** test_size: ", test_size)
        random_state = 0  
        print("random_state: ", random_state)

        X_train, X_test, y_train, y_test = train_test_split(trainData, trainLebel, test_size=test_size,
                                                            random_state=random_state)

        print("X_train.shape X_test.shape y_train.shape y_test.shape ",
              X_train.shape, X_test.shape, y_train.shape, y_test.shape)


        for temp5 in range(len(HIDDEN_DIMs)):
            hds = HIDDEN_DIMs[temp5]
            startCalculations(X_train, X_test, y_train, y_test, hds,num_masterIteration,numEpochs)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Processing in :",device)

    Sampled_inputs = loadInputs("Sampled_inputs.pck")
    Sampled_labels = loadInputs("Sampled_labels.pck")
    trainData = Sampled_inputs
    trainLebel = Sampled_labels
    print("trainData.shape: ", trainData.shape)
    print("trainLebel.shape: ", trainLebel.shape)

    print("Classes/labels : ",np.unique(trainLebel))


    temptrainData=np.empty([1540,60, 33])
    n=len(trainData)
    for l in range(0, n):
      temp=trainData[l]
      temp=temp.T
      temptrainData[l,:,:]=temp
      n=n+1

    trainData=temptrainData
    print("after transposing trainData.shape: ",trainData.shape)

    sc = StandardScaler()

    npArrays=[]
    for l in range(0, len(trainData)):
      trainData_std = sc.fit_transform(trainData[l])
      npArrays.append(trainData_std)
    arr = np.asarray(npArrays)
    trainData=arr

    INPUT_DIM = 33
    NUM_TS = 60
    NUM_CLASSES = len(np.unique(trainLebel))

    test_sizes = [0.3]
    HIDDEN_DIMs = [128]
    num_masterIteration = 1
    #numEpochs = 500
    numEpochs = 1
    start(test_sizes,HIDDEN_DIMs,num_masterIteration,numEpochs)