from Header import * #import header file

#class which contain classification algorithm function and will predict the suitable Algorithm
class AlgorithmPredictor:
    def __init__(self,data_train,data_test,target_train,target_test):#init method accepting testing and training data
         self.data_train = data_train#instance variable
         self.data_test = data_test
         self.target_train = target_train
         self.target_test = target_test
         self.AlgoDict = {}

    #Training and Testing of Data for Decision Tree
    def DecisionTree(self):
        cobj = tree.DecisionTreeClassifier()
        cobj.fit(self.data_train,self.target_train)
        output = cobj.predict(self.data_test)
        Accuracy = accuracy_score(self.target_test,output)#IMP
        #syntax : acuracy_score(expected kay hota,kay o/p ala)
        self.AlgoDict["Decision Tree"] = Accuracy*100

    #Training and Testing of Data for KNN
    def KNN(self):
        cobj = KNeighborsClassifier()
        cobj.fit(self.data_train,np.ravel(self.target_train))
        output = cobj.predict(self.data_test)
        Accuracy = accuracy_score(self.target_test,output)#IMP
        self.AlgoDict["K Nearest Neighbour"] = Accuracy * 100

    #Training and Testing of Data for Naive Bayes
    def NaiveBayes(self):
        cobj = GaussianNB()
        cobj.fit(self.data_train,self.target_train)
        output = cobj.predict(self.data_test)
        Accuracy = accuracy_score(self.target_test,output)#IMP
        self.AlgoDict["Naive Bayes"] = Accuracy * 100

    #Method to prdict algorithm
    def predictAlgorithm(self):

        self.DecisionTree()
        self.KNN()
        self.NaiveBayes()

        max_key = max(self.AlgoDict, key=self.AlgoDict.get)

        print("Preferable Algorithm for your Data Set is: ",max_key)
