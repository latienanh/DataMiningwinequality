from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

def print_regression_equation(coefficients, feature_names):
    equation = "y = "
    for i in range(len(coefficients)):
        if i == 0:
            equation += f"{coefficients[i][0]:.2f}"
        else:
            equation += f" + {coefficients[i][0]:.2f} * {feature_names[i-1]}"
    print("Regression equation:")
    print(equation)

# Định nghĩa các tên của các biến đầu vào
feature_names = ["x1", "x2", "x3"]

# Gọi hàm print_regression_equation với ma trận hệ số và tên các biến đầu vào
def calculate_z(X, coefficients):
    # Tính toán phần tử đầu tiên của z là β0
    print(X.shape)
    print(coefficients.shape)

    z = coefficients[0]

    # Tính toán phần tử còn lại của z theo phương trình y = β0 + β1x1 + β2x2 + ... + βnxn
    for i in range(len(coefficients) - 1):
        z += coefficients[i + 1] * X[:, i]

    # Áp dụng hàm sigmoid lên z
    print(z)
    # z = sigmoid(z)

    return z


# Hàm sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Hàm dự đoán xác suất
def predict_probabilities(X, coefficients):
    z = calculate_z(X, coefficients)
    return sigmoid(z)

def convert_to_binary(value):
    if value > 6.5:
        return 1
    else:
        return 0
def plot_data_pca(X_train, y_train, X_test, y_test, title):
    # Sử dụng PCA để giảm chiều dữ liệu xuống còn 2 chiều
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Biểu diễn dữ liệu huấn luyện
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap=plt.cm.coolwarm, label='Train', alpha=0.6, marker='o')

    # Biểu diễn dữ liệu kiểm tra
    plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap=plt.cm.coolwarm, label='Test', alpha=0.6, marker='s')

    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
def plot_decision_boundary(X, y, lr_model):
    # Biểu diễn dữ liệu huấn luyện
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, label='Data', alpha=0.6, marker='o')

    # Vẽ đường ranh giới
    coef = lr_model.coef_
    intercept = lr_model.intercept_

    x_values = np.array([np.min(X[:, 0]), np.max(X[:, 0])])
    y_values = (-1/coef[0][1]) * (coef[0][0] * x_values + intercept)
    plt.plot(x_values, y_values, label='Decision Boundary', color='black')

    plt.title('Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

# Gọi hàm plot_decision_boundary với dữ liệu huấn luyện và mô hình hồi quy logistic của bạn
def handleDataWithLogisticRegression(DataTrain, DataTest, title):
    print(title)
    scaler = StandardScaler()
    DataTrain_scaled = scaler.fit_transform(DataTrain)
    DataTest_scaled = scaler.fit_transform(DataTest)
    X_train = DataTrain.drop(["quality"], axis=1).copy()
    X_test = DataTest.drop(["quality"], axis=1).copy()
    y_train = DataTrain["quality"].apply(convert_to_binary).copy()
    y_test = DataTest["quality"].apply(convert_to_binary).copy()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    print(f"LogisticRegression accuracy: {accuracy_lr*100:.2f}%")
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    print(cm_lr)
    print("True Positive:", cm_lr[0,0])
    print("True Negative:", cm_lr[1,1])
    print("False Positive:", cm_lr[0,1])
    print("False Negative:", cm_lr[1,0])
    plot_data_pca(X_train, y_train,X_test,y_test,title)
    # Gọi hàm predict_probabilities với dữ liệu huấn luyện (X_train) và hệ số của mô hình (lr.coef_)
    probabilities_train = predict_probabilities(X_train, lr.coef_)
    # plot_decision_boundary(X_train, y_train, lr)
    return accuracy_lr,cm_lr
    # y_pred = lr.predict(X_test)


