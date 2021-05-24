from AlgorithmPredictor import *
from AlgorithmPredictor import AlgorithmPredictor

def loadData():
        titanic_data = pd.read_csv("TitanicDataset.csv")
        print("First five records of dataset : ")
        print(titanic_data.head())

        #Data Cleaning
        titanic_data.drop("zero",axis=1,inplace=True)
        print("Data after column removal : ")
        print(titanic_data.head())

        Sex = pd.get_dummies(titanic_data["Sex"])
        print(Sex.head())
        Sex = pd.get_dummies(titanic_data["Sex"],drop_first=True)
        print("Ssx column after updation : ")
        print(Sex.head())

        Pclass = pd.get_dummies(titanic_data["Pclass"])
        print(Pclass.head())
        Pclass = pd.get_dummies(titanic_data["Pclass"],drop_first=True)
        print("Pclass column after updation : ")
        print(Pclass.head())

        #Concat Sex and P class field in our dataset
        titanic_data = pd.concat([titanic_data,Sex,Pclass],axis=1)
        print("Data after concatination : ")
        print(titanic_data.head())

        #Removing uneccesary fields
        titanic_data.drop(["Sex","sibsp","Parch","Embarked"],axis=1,inplace=True)
        print(titanic_data.head())

        #Divide the dataset into X and Y
        x = titanic_data.drop("Survived",axis=1)
        y = titanic_data["Survived"]

        return x,y


def main():

    Features,Label = loadData()

    data_train,data_test,targe_train,target_test = train_test_split(Features,Label,test_size=0.5)

    aobj = AlgorithmPredictor(data_train,data_test,targe_train,target_test)

    aobj.predictAlgorithm()

if __name__ == '__main__':
    main()
