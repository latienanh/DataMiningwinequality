from sklearn.cluster import KMeans
import  numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
def elbow_Method(data,titleData):
    losses = []
    K = 10
    for i in range(1, K):
        # 1.  Huấn luyện với số cụm = i
        kmeans_i = KMeans(n_clusters=i, random_state=0)
        kmeans_i.fit(data)
        # 2. Tính _hàm biến dạng_
        # 2.1. Khoảng cách tới toàn bộ centroids
        d2centroids = cdist(data, kmeans_i.cluster_centers_, 'euclidean')  # shape (n, k)
        # 2.2. Khoảng cách tới centroid gần nhất
        min_distance = np.min(d2centroids, axis=1)  # shape (n)
        loss = np.sum(min_distance)
        losses.append(loss)

    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(1, K), losses, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Distortion')
    plt.title(f'The Elbow Method using Distortion with {titleData}')
    plt.show()
def reduce_dimensionality(X):
    # Sử dụng tSNE để giảm chiều dữ liệu về 2 chiều
    tsne = TSNE(n_components=2, random_state=0)
    X_embedded = tsne.fit_transform(X)
    return X_embedded
def handleDataWithKmean(DataTrain,DataTest,clusterNum,title):
    print(title)

    kmeans = KMeans(n_clusters=clusterNum, random_state=0).fit(DataTrain)
    # Gán nhãn cho từng điểm dữ liệu
    labels = kmeans.labels_
    # Đếm số lượng phần tử trong mỗi cụm
    unique_labels, counts = np.unique(labels, return_counts=True)
    # In số lượng phần tử trong mỗi cụm
    for label, count in zip(unique_labels, counts):
        print(f"Số lượng phần tử trong cụm {label}: {count}")
    centers = kmeans.cluster_centers_
    print("Các tâm của các cụm thu được:\n")
    print (centers)

    labels_train = kmeans.labels_
    labels_test = kmeans.predict(DataTest)
    print(np.unique(labels_test, return_counts=True))
    distances = np.sqrt(((DataTest - centers[labels_test]) ** 2).sum(axis=1))

    # Tính toán MSE
    mse = mean_squared_error(np.zeros(len(DataTest)), distances)
    print("Mean Square Error:", mse)

    X_train_embedded = reduce_dimensionality(DataTrain)
    X_test_embedded = reduce_dimensionality(DataTest)

    # Tạo và huấn luyện mô hình KMeans trên dữ liệu đã giảm chiều
    kmeansEmbedded = KMeans(n_clusters=clusterNum, random_state=0).fit(X_train_embedded)
    labels_train_embedded = kmeansEmbedded.labels_
    labels_test_embedded = kmeansEmbedded.predict(X_test_embedded)

    # Đếm số lượng phần tử trong mỗi cụm của tập huấn luyện
    # unique_labels_train, counts_train = np.unique(labels_train, return_counts=True)
    # In số lượng phần tử trong mỗi cụm của tập huấn luyện
    # for label, count in zip(unique_labels_train, counts_train):
    #     print(f"Số lượng phần tử trong cụm {label} của tập huấn luyện: {count}")

    # Đếm số lượng phần tử trong mỗi cụm của tập kiểm tra
    # unique_labels_test, counts_test = np.unique(labels_test, return_counts=True)
    # # In số lượng phần tử trong mỗi cụm của tập kiểm tra
    # for label, count in zip(unique_labels_test, counts_test):
    #     print(f"Số lượng phần tử trong cụm {label} của tập kiểm tra: {count}")
    #
    centersEmbedded = kmeansEmbedded.cluster_centers_
    # print("Các tâm của các cụm thu được:\n")
    # print(centers)

    # Biểu diễn dữ liệu của tập huấn luyện và tập kiểm tra cùng với trọng tâm của các cụm
    plt.figure(figsize=(10, 6))
    # Biểu diễn các điểm dữ liệu của tập huấn luyện
    plt.scatter(X_train_embedded[:, 0], X_train_embedded[:, 1], c=labels_train_embedded, cmap='viridis', label='Train Data')
    # Biểu diễn các điểm dữ liệu của tập kiểm tra
    plt.scatter(X_test_embedded[:, 0], X_test_embedded[:, 1], c=labels_test_embedded, cmap='viridis', marker='s',
                label='Test Data')
    # Biểu diễn trọng tâm của các cụm
    plt.scatter(centersEmbedded[:, 0], centersEmbedded[:, 1], c='red', marker='x', s=100, label='Cluster Centers')

    plt.title(f'Data Points and Cluster Centers {title}')
    plt.xlabel('tSNE Dimension 1')
    plt.ylabel('tSNE Dimension 2')
    plt.legend()
    plt.show()

