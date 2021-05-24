from AlgorithmPredictor import *
from AlgorithmPredictor import AlgorithmPredictor

def loadData():
    #Loading Data
    diabetes_data = pd.read_csv('diabetes.csv')

    print("First 5 records of dataset")
    print(diabetes_data.head())

    #Analysing Data
    x = diabetes_data.drop("Outcome",axis=1)
    y = diabetes_data["Outcome"]

    return x,y


def main():

    Features,Label = loadData()

    data_train,data_test,targe_train,target_test = train_test_split(Features,Label,test_size=0.5)

    aobj = AlgorithmPredictor(data_train,data_test,targe_train,target_test)

    aobj.predictAlgorithm()

if __name__ == '__main__':
    main()
