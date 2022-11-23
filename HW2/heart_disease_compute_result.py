import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

from Implementation.decision_tree import DTC
from Implementation.knn import KNN


data_frame = pd.read_csv('data/heart.csv')


def drawGrafics():
    sns.countplot(data=data_frame, x='target', color='orange')
    plt.xlabel('0 = Healthy, 1 = Ill')
    plt.ylabel('Count people')
    plt.title('Healthy/ill ratio => good learning')
    plt.savefig(
        'data_analyze/target_disease.png', transparent=False, facecolor='white', dpi=250
    )

    pd.crosstab(data_frame.thal, data_frame.target).plot(kind='bar', color=['orange', 'red'])
    plt.legend(['Healthy', 'Ill'])
    plt.xlabel('Level disease')
    plt.ylabel('Count people')
    plt.title('Higher level consequently higher risk of disease')
    plt.savefig(
        'data_analyze/level_disease.png',
        transparent=False,
        facecolor='white',
        dpi=250,
    )


drawGrafics()

data_frame = pd.concat(
    [
        data_frame,
        pd.get_dummies(data_frame['cp'], prefix='cp'),
        pd.get_dummies(data_frame['thal'], prefix='thal'),
        pd.get_dummies(data_frame['slope'], prefix='slope'),
    ],
    axis=1,
)
data_frame.drop(columns=['cp', 'thal', 'slope'])

# Scaling
Y = data_frame['target']
X = data_frame.drop(columns=['target'])

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.6, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def writeResult():
    accuracies_algorithm = {}

    knn_searcher = GridSearchCV(
        estimator=KNN(),
        param_grid=[{'k': [i for i in range(1, 7)], 'p_minkoswki': [i for i in range(1, 7)]}],
        cv=5,
    )
    knn_tuned = {'k': 1, 'p_minkoswki': 2}
    knn = KNN(k=1, p_minkowski=2)
    knn.fit(X_train_scaled, Y_train)
    accuracy = knn.score(X_test_scaled, Y_test)
    accuracies_algorithm['KNN'] = accuracy

    builtin_knn_searcher = GridSearchCV(
        estimator=KNeighborsClassifier(),
        param_grid=[
            {'leaf_size': [i for i in range(1, 50)], 'n_neighbors': [i for i in range(1, 30)], 'p': [1, 2]}
        ],
        cv=5,
    )
    builtin_knn_tuned = {'leaf_size': 1, 'n_neighbors': 1, 'p': 1}
    builtin_knn = KNeighborsClassifier(leaf_size=1, n_neighbors=1, p=1)
    builtin_knn.fit(X_train_scaled, Y_train)
    accuracy = builtin_knn.score(X_test_scaled, Y_test)
    accuracies_algorithm['Built-in KNN'] = accuracy

    dtc_searcher = GridSearchCV(
        estimator=DTC(),
        param_grid=[
            {'max_depth': [i for i in range(3, 40)], 'min_samples_split': [5, 10, 20, 50, 100]}
        ],
        cv=5,
    )
    dtc_tuned = {'max_depth': 11, 'min_samples_split': 5}
    dtc = DTC(11, 5)
    dtc.fit(X_train, Y_train)
    accuracy = dtc.score(X_test, Y_test)
    accuracies_algorithm['Decision Tree'] = accuracy

    builtin_dtc_searcher = GridSearchCV(
        estimator=DecisionTreeClassifier(),
        param_grid=[
            {
                'max_depth': [i for i in range(3, 40)],
                'min_samples_split': [5, 10, 20, 50, 100],
                'criterion': ['gini', 'entropy'],
            }
        ],
        cv=5,
    )
    builtin_dtc_searcher.fit(X_train, Y_train)
    builtin_dtc = sklearn.tree.DecisionTreeClassifier(
        criterion='entropy', max_depth=29, min_samples_split=5
    )
    builtin_dtc.fit(X_train, Y_train)
    accuracy = builtin_dtc.score(X_test, Y_test)
    accuracies_algorithm['Built-in Decision Tree'] = accuracy

    lr_searcher = GridSearchCV(
        estimator=LogisticRegression(),
        param_grid=[{'solver': ['newton-cg', 'lbfgs', 'liblinear'], 'penalty': ['none', 'l1', 'l2', 'elasticnet'], 'C': [100, 10, 1.0, 0.1, 0.01]}],
        cv=5,
    )
    lr_searcher.fit(X_train_scaled, Y_train)
    lr = LogisticRegression(C=100, solver='newton-cg')
    lr.fit(X_train_scaled, Y_train)
    accuracy = lr.score(X_test_scaled, Y_test)
    accuracies_algorithm['Logistic Regression'] = accuracy

    svc_searcher = GridSearchCV(
        estimator=SVC(),
        param_grid=[{'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'poly', 'sigmoid']}],
        cv=5,
    )
    svc_searcher.fit(X_train_scaled, Y_train)
    svc = SVC(C=0.1, gamma=1, kernel='poly')
    svc.fit(X_train_scaled, Y_train)
    accuracy = svc.score(X_test_scaled, Y_test)
    accuracies_algorithm['SVC'] = accuracy

    bayes_searcher = GridSearchCV(
        estimator=GaussianNB(),
        param_grid=[{'var_smoothing': np.logspace(0, -9, num=100)}],
        cv=5,
    )
    bayes_searcher.fit(X_train_scaled, Y_train)
    bayes = GaussianNB(var_smoothing=0.15)
    bayes.fit(X_train_scaled, Y_train)
    accuracy = bayes.score(X_test_scaled, Y_test)
    accuracies_algorithm['Naive Bayes'] = accuracy

    file = open('result/heart_disease.txt', 'w')
    file.write('Accuracy for algorithm:\n\n')
    for name, accuracy in accuracies_algorithm.items():
        file.write('Algorithm ' + name + ': ' + str(accuracy) + '\n')
    file.close()

    return knn, builtin_knn, dtc, builtin_dtc, lr, svc, bayes


knn, builtin_knn, dtc, builtin_dtc, lr, svc, bayes = writeResult()


def drawConfusionMatrix(knn, builtin_knn, dtc, builtin_dtc, lr, svc, bayes):
    y_head_knn = knn.predict(X_test_scaled)
    y_head_builtin_knn = builtin_knn.predict(X_test_scaled)
    y_head_dtc = dtc.predict(X_test)
    y_head_builtin_dtc = builtin_dtc.predict(X_test)
    y_head_lr = lr.predict(X_test_scaled)
    y_head_svm = svc.predict(X_test_scaled)
    y_head_bayes = bayes.predict(X_test_scaled)

    cm_knn = confusion_matrix(Y_test, y_head_knn)
    cm_b_knn = confusion_matrix(Y_test, y_head_builtin_knn)
    cm_dtc = confusion_matrix(Y_test, y_head_dtc)
    cm_b_dtc = confusion_matrix(Y_test, y_head_builtin_dtc)
    cm_lr = confusion_matrix(Y_test, y_head_lr)
    cm_svm = confusion_matrix(Y_test, y_head_svm)
    cm_bayes = confusion_matrix(Y_test, y_head_bayes)

    plt.figure(figsize=(24, 12))
    plt.suptitle("Confusion Matrices", fontsize=24)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    plt.subplot(2, 4, 1)
    plt.title("Result for KNN")
    sns.heatmap(cm_knn, annot=True, cmap="Reds", fmt="d", cbar=False, annot_kws={"size": 24})

    plt.subplot(2, 4, 2)
    plt.title("Result for Built-in KNN")
    sns.heatmap(cm_b_knn, annot=True, cmap="Reds", fmt="d", cbar=False, annot_kws={"size": 24})

    plt.subplot(2, 4, 3)
    plt.title("Result for Decision Tree")
    sns.heatmap(cm_dtc, annot=True, cmap="Reds", fmt="d", cbar=False, annot_kws={"size": 24})

    plt.subplot(2, 4, 4)
    plt.title("Result for Built-in Decision Tree")
    sns.heatmap(cm_b_dtc, annot=True, cmap="Reds", fmt="d", cbar=False, annot_kws={"size": 24})

    plt.subplot(2, 4, 5)
    plt.title("Result for Logistic Regression")
    sns.heatmap(cm_lr, annot=True, cmap="Reds", fmt="d", cbar=False, annot_kws={"size": 24})

    plt.subplot(2, 4, 6)
    plt.title("Result for SVM")
    sns.heatmap(cm_svm, annot=True, cmap="Reds", fmt="d", cbar=False, annot_kws={"size": 24})

    plt.subplot(2, 4, 7)
    plt.title("Result for Naive Bayes")
    sns.heatmap(cm_bayes, annot=True, cmap="Reds", fmt="d", cbar=False, annot_kws={"size": 24})

    plt.savefig(
        'result/confusion_matrix_disease.png',
        transparent=False,
        facecolor='white',
        dpi=300,
    )


drawConfusionMatrix(knn, builtin_knn, dtc, builtin_dtc, lr, svc, bayes)
