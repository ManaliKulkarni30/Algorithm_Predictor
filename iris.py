from AlgorithmPredictor import *
from AlgorithmPredictor import Algorithm

def loadData():
    df = pd.read_csv('iris.csv')
    le = LabelEncoder()
    df["species_label"] = le.fit_transform(df["species"])
    Features = df.iloc[:, lambda df: [0,1,2,3]]
    Label = df.iloc[:, lambda df: [5]]

    return Features,Label


def main():

    Features,Label = loadData()

    data_train,data_test,targe_train,target_test = train_test_split(Features,Label,test_size=0.5)

    aobj = Algorithm(data_train,data_test,targe_train,target_test)

    aobj.predictAlgorithm()

if __name__ == '__main__':
    main()
