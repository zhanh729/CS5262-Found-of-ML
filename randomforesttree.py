import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def run_kfold_smote_rf(X, y, n_splits=5, random_state=42):
    """
    Runs a KFold cross-validation with SMOTE and RandomForest using GridSearchCV for hyperparameter tuning.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # 设置网格搜索的参数范围
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [2, 5, 10, None],
        'min_samples_split': [2, 5, 10]
    }

    model = RandomForestClassifier(random_state=random_state)
    grid_search = GridSearchCV(model, param_grid, cv=kf, scoring='accuracy')

    accuracies = []
    for train_index, test_index in kf.split(X):
        # 分割数据
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # SMOTE过采样
        smote = SMOTE(random_state=random_state)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        # 训练并进行网格搜索
        grid_search.fit(X_train_smote, y_train_smote)

        # 预测
        y_pred = grid_search.best_estimator_.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))

    average_accuracy = sum(accuracies) / len(accuracies)
    return accuracies, average_accuracy, grid_search.best_params_


# 读取数据集
df = pd.read_csv('2022-2023 NBA Player Stats cleaned.csv')
features = df.drop(['allstar_selected', 'Rk', 'Player', 'Pos', 'Tm'], axis=1)
labels = df['allstar_selected']

# 运行函数
accuracies, average_accuracy, best_params = run_kfold_smote_rf(features, labels)

# 输出结果
print("Each fold's accuracies:", accuracies)
print("Average accuracy:", average_accuracy)
print("Best parameters:", best_params)

# t-SNE 可视化
tsne = TSNE(n_components=2, random_state=42)
features_tsne = tsne.fit_transform(features)

plt.figure(figsize=(12, 8))
colors = ['red' if label == 1 else 'blue' for label in labels]
plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=colors, alpha=0.5)
plt.title('t-SNE visualization of NBA player stats')
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.savefig('t-sne graph')
plt.show()
