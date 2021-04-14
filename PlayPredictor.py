from AlgorithmPredictor import *
from AlgorithmPredictor import Algorithm

def loadData():
    df = pd.read_csv('PlayPredictor.csv')
    le = LabelEncoder()
    df["Wether_label"] = le.fit_transform(df["Wether"])
    df["Temp_label"] = le.fit_transform(df["Temperature"])
    df["Play_label"] = le.fit_transform(df["Play"])
    #print(df)
    #print(df.head())
    Features = df.iloc[:, lambda df: [3, 4]]
    Label = df.iloc[:, lambda df: [5]]
    #print(Features.head())
    #print(Label.head())

    return Features,Label


def main():

    Features,Label = loadData()

    data_train,data_test,targe_train,target_test = train_test_split(Features,Label,test_size=0.5)

    aobj = Algorithm(data_train,data_test,targe_train,target_test)

    aobj.predictAlgorithm()

if __name__ == '__main__':
    main()
