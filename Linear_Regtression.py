import os
import numpy as np
from sklearn import datasets
from Regression import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

irisdata = datasets.load_iris()
X_train, X_test, Target_train,  Target_test=train_test_split(irisdata.data,irisdata.target,test_size=.4)
Batch_size=10
epoch_num = int(len(X_train)/Batch_size)
MeanSquareError=np.zeros((3,epoch_num))
R2Score=np.zeros((3,epoch_num))
for epoch in range(epoch_num):
    X_batch= X_train[epoch:epoch+Batch_size,:]
    Y_batch= Target_train[epoch:epoch+Batch_size]
    Reg= Regression(X_batch,Y_batch)
    LinReg,_= Reg.LinearRegression(X_batch)
    RigReg,_= Reg.RidgeRegression(X_batch, alpha=0.1)
    LasReg,_= Reg.LassoRegression(X_batch,alpha=0.1)
    LinReg_Eval=Evaluation(LinReg,Y_batch)
    RigReg_Eval=Evaluation(RigReg, Y_batch)
    LasReg_Eval=Evaluation(LasReg, Y_batch)

    MeanSquareError[0,epoch]=LinReg_Eval.MeanSquarErr()
    MeanSquareError[1,epoch] = RigReg_Eval.MeanSquarErr()
    MeanSquareError[2,epoch] = LasReg_Eval.MeanSquarErr()
    R2Score[0,epoch]=LinReg_Eval.R2Square()
    R2Score[1, epoch] = RigReg_Eval.R2Square()
    R2Score[2, epoch] = LasReg_Eval.R2Square()

fig = plt.figure()
fig1=fig.add_subplot(1,1,1)
# fig2=fig.add_subplot(3,1,2)
# fig3=fig.add_subplot(3,1,3)

fig1.plot(np.arange(1,epoch_num+1),MeanSquareError[0,:],'--r',
          np.arange(1, epoch_num + 1), MeanSquareError[1, :], '-*b',
          np.arange(1, epoch_num + 1), MeanSquareError[2, :], '-^g')

fig1.set_xlabel('Number of epochs')
fig1.set_ylabel('Mean Square Error for three different Regressors')
plt.show()
plt.close()









# Answer,_ = Regression(X_train, Target_train).LinearRegression(X_test)


print("The desired number of {} returns {} marketing expenditure".format(desired_unit, Answer))
