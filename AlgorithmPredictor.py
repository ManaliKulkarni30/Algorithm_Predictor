from Header import *

class Algorithm:
    def __init__(self,data_train,data_test,targe_train,target_test):
         self.data_train = data_train
         self.data_test = data_test
         self.targe_train = targe_train
         self.target_test = target_test
         self.AlgoDict = {}

    def DecisionTree(self):
        cobj = tree.DecisionTreeClassifier()
        cobj.fit(self.data_train,self.targe_train)
        output = cobj.predict(self.data_test)
        Accuracy = accuracy_score(self.target_test,output)#IMP
        self.AlgoDict["Decision Tree"] = Accuracy*100

    def KNN(self):
        cobj = KNeighborsClassifier()
        cobj.fit(self.data_train,np.ravel(self.targe_train))
        output = cobj.predict(self.data_test)
        Accuracy = accuracy_score(self.target_test,output)#IMP
        self.AlgoDict["K Nearest Neighbour"] = Accuracy * 100
        
     def NaiveBayes(self):
        cobj = GaussianNB()
        cobj.fit(self.data_train,self.target_train)
        output = cobj.predict(self.data_test)
        Accuracy = accuracy_score(self.target_test,output)#IMP
        self.AlgoDict["Naive Bayes"] = Accuracy * 100

    def predictAlgorithm(self):

        self.DecisionTree()
        self.KNN()

        max_key = max(self.AlgoDict, key=self.AlgoDict.get)

        print("Preferable Algorithm for your Data Set is: ",max_key)
