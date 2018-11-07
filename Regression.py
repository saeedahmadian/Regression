
from sklearn import linear_model
import sklearn.metrics as mr

class Regression:
    def __init__(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def checkData(self):
        check = True
        if len(self.X_train) != len(self.Y_train):
            check = False
        return check

    def LinearRegression(self, X_in):
        if self.checkData():
            LinearReg = linear_model.LinearRegression()
            LinearReg.fit(self.X_train, self.Y_train)
            return LinearReg.predict(X_in), LinearReg.coef_
        else:
            print("The Length of the X is {} and length of Y is {}.\nThey have to have the same dimension ".
                  format(len(self.X_train), len(self.Y_train)))

    def RidgeRegression(self,X_in,alpha):
        if self.checkData():
            RidgeReg = linear_model.Ridge(alpha=alpha).fit(self.X_train,self.Y_train)
            return RidgeReg.predict(X_in), RidgeReg.coef_
        else:
            print("The Length of the X is {} and length of Y is {}.\nThey have to have the same dimension ".
                  format(len(self.X_train), len(self.Y_train)))

    def LassoRegression(self,X_in,alpha):
        if self.checkData():
            LassoReg = linear_model.Lasso(alpha=alpha)
            LassoReg.fit(self.X_train,self.Y_train)
            return LassoReg.predict(X_in), LassoReg.coef_
        else:
            print("The Length of the X is {} and length of Y is {}.\nThey have to have the same dimension ".
                  format(len(self.X_train), len(self.Y_train)))


class Evaluation():
    def __init__(self,Y_pred, Y_true):
        self.Y_pred= Y_pred
        self.Y_true= Y_true

    def R2Square(self):
        return mr.r2_score(self.Y_true,self.Y_pred)

    def MeanSquarErr(self):
        return mr.mean_squared_error(self.Y_true,self.Y_pred)


