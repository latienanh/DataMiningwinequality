from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from SetupFile import write_file;
from K_Mean import handleDataWithKmean,elbow_Method;
from LogisticRegression import handleDataWithLogisticRegression
import pandas as pd

def handleKmeanMachineLearning(dataTrainRed,dataTestRed,dataTrainWhite,dataTestWhite):
    dataTrainRed.drop(["quality"], axis=1, inplace=True);
    dataTestRed.drop(["quality"], axis=1, inplace=True);
    dataTrainWhite.drop(["quality"], axis=1, inplace=True);
    dataTestWhite.drop(["quality"], axis=1, inplace=True);
    handleDataWithKmean(DataTrain=dataTrainWhite, clusterNum=6, DataTest=dataTestWhite, title=f"White WineQuality{i}")
    handleDataWithKmean(DataTrain=dataTrainRed, clusterNum=6, DataTest=dataTestRed, title=f"Red WineQuality{i}")
def handleLogisticRegressionMachineLearning(dataTrainRed, dataTestRed,dataTrainWhite,dataTestWhite):

    # print(dataTrainRed,dataTestRed,dataTrainWhite,dataTestWhite);
    accuracieWhite,cmWhite=handleDataWithLogisticRegression(DataTrain=dataTrainWhite, DataTest=dataTestWhite, title=f"White WineQuality{i}")
    accuracieRed,cmRed=handleDataWithLogisticRegression(DataTrain=dataTrainRed, DataTest=dataTestRed, title=f"Red WineQuality{i}")
    return accuracieWhite,cmWhite,accuracieRed,cmRed

if __name__ == "__main__":
    pathWineQuantityRed = r"../data/winequality-red.csv";
    pathWineQuantityWhite = r"../data/winequality-white.csv";
    pathData = r"../data/"

    data_red = pd.read_csv(pathWineQuantityRed)
    # data_red.hist(bins=20, figsize=(16, 9))
    # plt.suptitle("Red Quality Histogram")  # Đặt tiêu đề cho toàn bộ hình vẽ
    # plt.show()

    # Đọc dữ liệu rượu trắng và vẽ biểu đồ histogram
    data_white = pd.read_csv(pathWineQuantityWhite)
    # data_white.hist(bins=20, figsize=(16, 9))
    # plt.suptitle("White Quality Histogram")  # Đặt tiêu đề cho toàn bộ hình vẽ
    # plt.show()
    # elbow_Method(data_red,"Winequality White Red")
    # elbow_Method(data_white,"Winequality White")

    accuraciesWhite = []
    true_positivesWhite = []
    true_negativesWhite = []
    false_positivesWhite = []
    false_negativesWhite = []
    accuraciesRed = []
    true_positivesRed = []
    true_negativesRed = []
    false_positivesRed = []
    false_negativesRed = []
    # write_file(pathWineQuantityRed,pathWineQuantityWhite,pathData)
    for i in range(3):
        pathWineQuantityWhiteTrain = f"../data/white_train_split_{i}.csv";
        pathWineQuantityWhiteTest = f"../data/white_test_split_{i}.csv";
        dataTrainWhite = pd.read_csv(pathWineQuantityWhiteTrain)
        dataTestWhite = pd.read_csv(pathWineQuantityWhiteTest);
        pathWineQuantityRedTrain = f"../data/red_train_split_{i}.csv";
        pathWineQuantityRedTest = f"../data/red_test_split_{i}.csv";
        dataTrainRed = pd.read_csv(pathWineQuantityRedTrain)
        dataTestRed = pd.read_csv(pathWineQuantityRedTest);
        # print(f"Data {i}")
        # print("Train data counts:")
        # print(dataTrain["quality"].value_counts())
        # print("\nTest data counts:")
        # print(dataTest["quality"].value_counts())
        # handleKmeanMachineLearning(dataTrainRed, dataTestRed,dataTrainWhite,dataTestWhite)
        accuracieWhite,cmWhite,accuracieRed,cmRed=handleLogisticRegressionMachineLearning(dataTrainRed, dataTestRed,dataTrainWhite,dataTestWhite)
        accuraciesWhite.append(accuracieWhite)
        true_positivesWhite.append(cmWhite[0, 0])
        true_negativesWhite.append(cmWhite[1, 1])
        false_positivesWhite.append(cmWhite[0, 1])
        false_negativesWhite.append(cmWhite[1, 0])
        accuraciesRed.append(accuracieRed)
        true_positivesRed.append(cmRed[0, 0])
        true_negativesRed.append(cmRed[1, 1])
        false_positivesRed.append(cmRed[0, 1])
        false_negativesRed.append(cmRed[1, 0])

    avg_accuracy_white = sum(accuraciesWhite) / len(accuraciesWhite)
    avg_true_positive_white = sum(true_positivesWhite) / len(true_positivesWhite)
    avg_true_negative_white = sum(true_negativesWhite) / len(true_negativesWhite)
    avg_false_positive_white = sum(false_positivesWhite) / len(false_positivesWhite)
    avg_false_negative_white = sum(false_negativesWhite) / len(false_negativesWhite)

    avg_accuracy_red = sum(accuraciesRed) / len(accuraciesRed)
    avg_true_positive_red = sum(true_positivesRed) / len(true_positivesRed)
    avg_true_negative_red = sum(true_negativesRed) / len(true_negativesRed)
    avg_false_positive_red = sum(false_positivesRed) / len(false_positivesRed)
    avg_false_negative_red = sum(false_negativesRed) / len(false_negativesRed)
    # In kết quả
    print("Average Accuracy White:", avg_accuracy_white)
    print("Average True Positive White:", avg_true_positive_white)
    print("Average True Negative White:", avg_true_negative_white)
    print("Average False Positive White:", avg_false_positive_white)
    print("Average False Negative White:", avg_false_negative_white)

    print("Average Accuracy Red:", avg_accuracy_red)
    print("Average True Positive Red:", avg_true_positive_red)
    print("Average True Negative Red:", avg_true_negative_red)
    print("Average False Positive Red:", avg_false_positive_red)
    print("Average False Negative Red:", avg_false_negative_red)


