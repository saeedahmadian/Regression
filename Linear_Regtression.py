import numpy as np
from sklearn import linear_model

X = np.array([[60000], [50000], [90000], [80000], [30000]])
Target = np.array([[300000], [200000], [400000], [300000], [10000]])
Target2 = np.array([[300000], [200000], [400000], [300000]])


#
class MarketingCosts:
    def __init__(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def checkData(self):
        check = True
        if len(self.X_train) != len(self.Y_train):
            check = False
        return check

    def desired_marketing_expenditure(self, desired_units):
        if self.checkData():
            LinearReg = linear_model.LinearRegression()
            LinearReg.fit(self.X_train, self.Y_train)
            return LinearReg.predict(desired_units)
        else:
            print("The Length of the X is {} and length of Y is {}.\n They have to have the same dimension ".
                  format(len(self.X_train), len(self.Y_train)))


desired_unit = [[60000]]
Answer = MarketingCosts(X, Target2).desired_marketing_expenditure(desired_units=desired_unit)

print("The desired number of {} returns {} marketing expenditure".format(desired_unit, Answer))
