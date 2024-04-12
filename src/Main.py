from matplotlib import pyplot as plt

from SetupFile import write_file;
from K_Mean import handleDataWithKmean;
import pandas as pd
if __name__ == "__main__":
    pathWineQuantityRed = r"../data/winequality-red.csv";
    pathWineQuantityWhite = r"../data/winequality-white.csv";
    pathData = r"../data/";
    # write_file(pathWineQuantityRed,pathWineQuantityWhite,pathData)
    pathWineQuantityWhiteTrain = r"../data/red_train_split_0.csv";
    pathWineQuantityWhiteTest = r"../data/red_test_split_0.csv";
    data= pd.read_csv(pathWineQuantityWhiteTrain);
    data.hist(bins=20, figsize=(12,12));
    plt.show();
    dataTest = pd.read_csv(pathWineQuantityWhiteTest);
    print(data["quality"].value_counts());
    data.drop(["quality"],axis=1,inplace=True);
    dataTest.drop(["quality"], axis=1, inplace=True);
    print(data.head())
    handleDataWithKmean(DataTrain=data,clusterNum=2,DataTest=dataTest)
