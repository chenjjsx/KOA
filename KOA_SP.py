from imblearn.over_sampling import SMOTE
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn as sns

df = pd.read_excel("health_data.xlsx", dtype={"num": str})
# 初始化特征矩阵
feature_matrix = []
feature_matrix_test = []


def extract_features(image, name_part, type_part):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image.astype(float)

    image[(image == 0).all(axis=-1)] = np.nan

    image_height = image.shape[0]
    image_width = image.shape[1]
    # 计算四分之一和四分之三的位置
    quarter_position = image_width // 4
    three_quarter_position = 3 * (image_width // 4)
    features = []
    # 提取竖线像素值
    left = image[:, quarter_position]
    right = image[:, three_quarter_position]

    # 将特征合并为一个特征向量
    feature = np.concatenate((left, right))
    # -------------------将膝盖处的像素值赋予比重--------------------------------
    center_height = image_height // 2
    # 初始化一个空的加权后的特征向量 左腿
    weighted_feature_l = []
    for idx, pixel_value in enumerate(left):
        distance_to_center = abs(idx - center_height)
        # 使用高斯分布函数计算权重
        sigma = 80
        weight = np.exp(- (distance_to_center ** 2) / (2 * sigma ** 2))
        weighted_value = pixel_value * weight
        weighted_feature_l.append(weighted_value)
    # 初始化一个空的加权后的特征向量 右腿
    weighted_feature_r = []
    for idx, pixel_value in enumerate(right):
        distance_to_center = abs(idx - center_height)
        # 使用高斯分布函数计算权重
        sigma = 80
        weight = np.exp(- (distance_to_center ** 2) / (2 * sigma ** 2))
        weighted_value = pixel_value * weight
        weighted_feature_r.append(weighted_value)

    feature = np.concatenate((weighted_feature_l, weighted_feature_r))

    # 计算图像中的中位数像素值
    median_value = np.nanmedian(feature)
    # 计算像素值偏度
    skewness = skew(feature.flatten(), nan_policy='omit')
    # 计算像素值峰度
    kurtosis_value = kurtosis(feature.flatten(), nan_policy='omit')
    # 计算像素值均方根
    rms = np.sqrt(np.nanmean(np.square(feature)))

    # 计算幅值特征：平均值、最大值、最小值、标准差
    mean_value = np.nanmean(feature)
    max_value = np.nanmax(feature)
    min_value = np.nanmin(feature)
    std_value = np.nanstd(feature)
    # 计算变化特征：相邻像素之间的差值的平均值、最大值、最小值、标准差
    # 处理包含NaN值和0的像素值
    feature_without_nan = feature[~np.isnan(feature)]  # 去除NaN值
    non_zero_feature = feature_without_nan[feature_without_nan != 0]  # 去除0值
    differences = np.diff(non_zero_feature)
    mean_difference = np.mean(differences)
    max_difference = np.max(differences)
    min_difference = np.min(differences)
    std_difference = np.std(differences)

    # ------------------纹理特征
    _, image_encoded = cv2.imencode('.jpg', feature)
    image_gray = cv2.imdecode(image_encoded, cv2.IMREAD_GRAYSCALE)
    distance = 1
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(image_gray, [distance], angles=angles, symmetric=True, normed=True)

    dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    # 所有特征
    features.extend([median_value, skewness, kurtosis_value, rms, mean_value, max_value, min_value, std_value,
                     mean_difference, max_difference, min_difference, std_difference])
    features.extend(dissimilarity)
    features.extend(energy)
    features.extend(correlation)
    # 健康数据bingchen、age、身高、体重、BMI、temp
    for index, row in df.iterrows():
        image_name = row["num"]
        feature_vector = row.drop("num").values
        if image_name == name_part:
            features.extend(feature_vector)

    return [features[29], features[11], features[14], features[13], features[15],
            features[12], features[9], features[22], features[28], features[25], features[1]]


def calculate_metrics(labels_test_other, predictions_other):
    """
    计算精确率、召回率、F1分数和准确率。
    """
    accuracy_other = accuracy_score(labels_test_other, predictions_other)
    precision_other = precision_score(labels_test_other, predictions_other, average='weighted', zero_division=1)
    recall_other = recall_score(labels_test_other, predictions_other, average='weighted')
    f1_other = f1_score(labels_test_other, predictions_other, average='weighted')

    return accuracy_other, precision_other, recall_other, f1_other


image_folder = "train"
image_files = [file for file in os.listdir(image_folder) if file.endswith(".jpg") or file.endswith(".png")]
image_names = [name.split('.')[0] for name in image_files]
labels = []
for image_file in image_files:
    parts = image_file.split('_')
    name_part = parts[0]
    type_part = parts[1]
    type_part = type_part.replace('.jpg', '')
    labels.append(type_part)
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)
    image_features = []
    # image_features = extract_features(image, num_horizontal_bars)
    image_features = extract_features(image, name_part, type_part)
    # features_array = np.array(image_features)
    # save_to_excel(name_part, type_part, image_features)
    # 将图像的特征向量添加到特征矩阵中
    feature_matrix.append(image_features)

image_folder_test = "test"  # 测试集
# image_folder_test = "val_jpg" # 验证集rgb三通道拟合后
# image_folder_test = "val_jpg_std" # 标准验证集原图
image_files_test = [file for file in os.listdir(image_folder_test) if file.endswith(".jpg") or file.endswith(".png")]
image_names_test = [name.split('.')[0] for name in image_files_test]
labels_test = []
for image_file_test in image_files_test:
    parts_test = image_file_test.split('_')
    name_part_test = parts_test[0]
    type_part_test = parts_test[1]
    type_part_test = type_part_test.replace('.jpg', '')
    labels_test.append(type_part_test)
    image_path_test = os.path.join(image_folder_test, image_file_test)
    image_test = cv2.imread(image_path_test)
    image_features_test = []
    # image_features = extract_features(image, num_horizontal_bars)
    image_features_test = extract_features(image_test, name_part_test, type_part_test)
    # features_array = np.array(image_features)
    # 存储最终筛选的特征值到excel中，用于特征分布可视化
    # save_to_excel(name_part_test, type_part_test, image_features_test)

    feature_matrix_test.append(image_features_test)

smote = SMOTE(
    sampling_strategy='auto',
    random_state=30,
    k_neighbors=5,
)
X_resampled, y_resampled = smote.fit_resample(feature_matrix, labels)
# 合并特征矩阵
X_combined = np.concatenate((feature_matrix, X_resampled), axis=0)  # 全部数据
# 合并标签数组
y_combined = np.concatenate((labels, y_resampled), axis=0)  # 全部数据

label_mapping = {'1': 1, '2': 2, '3': 3}
y_integer = [label_mapping[label] for label in y_combined]
value_counts = np.bincount(y_integer)
print("Value counts:", value_counts)

# 梯度提升树分类器
clf = GradientBoostingClassifier()
clf.fit(X_combined, y_combined)  # 全部数据（somte生成数据）
# clf.fit(feature_matrix, labels)  # 无somte生成数据）


# 保存模型
with open('model_vertical_2.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)

# 加载模型
with open('model_vertical_2.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

predictions = loaded_model.predict(feature_matrix_test)

# 输出预测结果
for i in range(len(labels_test)):
    true_label = labels_test[i]
    predicted_label = predictions[i]
    print(f"Sample {i + 1}: T = {true_label}, P = {predicted_label}")

# 计算准确率
accuracy_gbc = accuracy_score(labels_test, predictions)
# 计算精确率、召回率和 F1 分数
precision = precision_score(labels_test, predictions, average='weighted', zero_division=1)
recall = recall_score(labels_test, predictions, average='weighted')
f1 = f1_score(labels_test, predictions, average='weighted')

# 输出结果
print("Accuracy:", accuracy_gbc)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# SVM
svm = SVC()
svm.fit(X_combined, y_combined)
svm_predictions = svm.predict(feature_matrix_test)
# 随机森林分类器
rfc = RandomForestClassifier(criterion="entropy", n_estimators=66)
rfc.fit(X_combined, y_combined)
rfc_predictions = rfc.predict(feature_matrix_test)
# K最近邻分类器，设置K的值为3
knc = KNeighborsClassifier(n_neighbors=3)
knc.fit(X_combined, y_combined)
knc_predictions = knc.predict(feature_matrix_test)
# 决策树分类器
dtc = DecisionTreeClassifier()
dtc.fit(X_combined, y_combined)
dtc_predictions = dtc.predict(feature_matrix_test)
# 多层感知器分类器
mlp = MLPClassifier(hidden_layer_sizes=(100,), learning_rate_init=0.001, max_iter=2000)
mlp.fit(X_combined, y_combined)
mlp_predictions = mlp.predict(feature_matrix_test)
# 创建Adaboost分类器对象
abc = AdaBoostClassifier()
abc.fit(X_combined, y_combined)
abc_predictions = abc.predict(feature_matrix_test)

before_X_combined = np.array(feature_matrix)

feature_names = ["temperature", "standard deviation", "dissimilarity 3", "dissimilarity 2", "dissimilarity 4",
                 "dissimilarity 1",
                 "maximum", "correlation", "BMI", "age", "skewness"]

plt.rcParams["font.family"] = "Times New Roman"
num_features = len(feature_names)
fig, axes = plt.subplots(1, num_features, figsize=(20, 6))

# 循环绘制每个特征的箱线图对比分布
for i in range(num_features):
    sns.boxplot(data=[before_X_combined[:, i], X_combined[:, i]], orient="v", palette="Set2", ax=axes[i])
    axes[i].set_xlabel("Before vs. After", fontsize=14)
    axes[i].set_ylabel(feature_names[i], fontsize=14)
    axes[i].set_title(f"{feature_names[i]}", fontsize=16)

plt.tight_layout()

plt.show()

models_all = ["GradientBoosting", "SVM", "RandomForest", "KNeighbors", "DecisionTree", "Adaboost"]
predictions_all = [predictions, svm_predictions, rfc_predictions, knc_predictions, dtc_predictions,
                   abc_predictions]
labels_test_all = [labels_test, labels_test, labels_test, labels_test, labels_test, labels_test]

performance_dict = {
    "Model": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1 Score": []
}

# 计算性能指标并添加到数据字典中
for model, pred, labels in zip(models_all, predictions_all, labels_test_all):
    accuracy, precision, recall, f1 = calculate_metrics(labels, pred)
    performance_dict["Model"].append(model)
    performance_dict["Accuracy"].append(accuracy)
    performance_dict["Precision"].append(precision)
    performance_dict["Recall"].append(recall)
    performance_dict["F1 Score"].append(f1)

performance_df = pd.DataFrame(performance_dict)

print(performance_df)

cm_gbc = confusion_matrix(labels_test, predictions)
cm_svm = confusion_matrix(labels_test, svm_predictions)
cm_rfc = confusion_matrix(labels_test, rfc_predictions)
cm_knc = confusion_matrix(labels_test, knc_predictions)
cm_dtc = confusion_matrix(labels_test, dtc_predictions)
cm_mlp = confusion_matrix(labels_test, mlp_predictions)
cm_abc = confusion_matrix(labels_test, abc_predictions)

confusion_matrices = [cm_gbc, cm_svm, cm_rfc, cm_knc, cm_dtc, cm_abc]

method_names = ["GradientBoosting", "SVM", "RandomForest", "KNeighbors", "DecisionTree", "Adaboost"]
plt.rcParams["font.family"] = "Times New Roman"
fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
fig.subplots_adjust(hspace=1.5, wspace=1.5)  # 调整水平和垂直间距

for i, ax in enumerate(axes.flat):
    ax.set_title(method_names[i] + " Confusion Matrix", fontsize=20, fontweight='bold')
    total = confusion_matrices[i].sum()
    normalized_matrix = confusion_matrices[i] / total
    sns.heatmap(normalized_matrix, annot=True, fmt=".2f", cmap=plt.cm.Blues, cbar=False, ax=ax, annot_kws={"size": 20})
    ax.set_xlabel("Predicted", fontsize=18)
    ax.set_ylabel("Actual", fontsize=18)

plt.tight_layout()
plt.show()
