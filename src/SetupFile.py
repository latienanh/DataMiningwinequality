import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
def write_file(path1,path2,pathData):
    red_wine_data = pd.read_csv(path1)
    white_wine_data = pd.read_csv(path2)

    # Lặp lại quá trình phân chia ba lần
    for i in range(3):
        # Phân chia dữ liệu cho mỗi loại rượu
        red_train, red_test = train_test_split(red_wine_data, test_size=0.1, random_state=i)
        white_train, white_test = train_test_split(white_wine_data, test_size=0.1, random_state=i)

        # Lưu các tệp dữ liệu
        red_train.to_csv(f"{pathData}red_train_split_{i}.csv", index=False)
        red_test.to_csv(f"{pathData}red_test_split_{i}.csv", index=False)
        white_train.to_csv(f"{pathData}white_train_split_{i}.csv", index=False)
        white_test.to_csv(f"{pathData}white_test_split_{i}.csv", index=False)
