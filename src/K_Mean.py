from sklearn.cluster import KMeans
import  numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
def handleDataWithKmean(DataTrain,DataTest,clusterNum):
    print('Kmeans')
    # phan_cum = KMeans(n_clusters=clusterNum, random_state=0).fit(DataTrain)
    # centers = phan_cum.cluster_centers_
    # print("In tâm của các cụm thu được:\n")
    # print (centers)
    kmeans = KMeans(n_clusters=clusterNum, random_state=0).fit(DataTrain)

    # Gán nhãn cho từng điểm dữ liệu
    labels = kmeans.labels_

    # Đếm số lượng phần tử trong mỗi cụm
    unique_labels, counts = np.unique(labels, return_counts=True)

    # In số lượng phần tử trong mỗi cụm
    for label, count in zip(unique_labels, counts):
        print(f"Số lượng phần tử trong cụm {label}: {count}")


    y_pred = kmeans.predict(DataTest)
    print(np.unique(y_pred, return_counts=True))
    # Tính ma trận nhầm lẫn
    # cm = confusion_matrix(y_test, y_pred)
    #
    # # Tính các chỉ số hiệu suất
    # accuracy = accuracy_score(y_test, y_pred)
    # true_negative_rate = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    # false_positive_rate = cm[0, 1] / (cm[0, 0] + cm[0, 1])
    # false_negative_rate = cm[1, 0] / (cm[1, 0] + cm[1, 1])
    # true_positive_rate = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    #
    # return cm, accuracy, true_negative_rate, false_positive_rate, false_negative_rate, true_positive_rate;