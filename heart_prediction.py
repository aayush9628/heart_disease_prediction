import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from sklearn.impute import KNNImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.svm import SVC
from sklearn.metrics import classification_report, hinge_loss, accuracy_score, log_loss, plot_roc_curve, roc_curve, auc
from sklearn.linear_model import LogisticRegression
import plotly.express as px

data = pd.read_csv('/Users/aayushshrivastava/Downloads/heart.csv')
numeric_columns = data.describe().columns.to_list()
categorical_columns = []
for i in data.columns.to_list():
    if i not in numeric_columns:
        categorical_columns.append(i)


def visual(data):
    oe = ['g', 'r']
    fig = plt.figure(figsize=(12,7))

    for i in range(len(numeric_columns)):
        sns.set(font_scale=1)
        plt.subplot(2,3,i+1)
        plt.style.use('seaborn')
        plt.tight_layout()
        sns.set_context('talk')
        sns.histplot(data=data, x=numeric_columns[i], hue=data['HeartDisease'], multiple="stack", palette=oe)
        plt.legend(labels=["Normal","Heart Disease"], fontsize = 10)
    plt.show()

    fig = plt.figure(figsize=(12,7))
    for i in range(len(categorical_columns)):
        sns.set(font_scale=1)
        plt.subplot(2,3,i+1)
        plt.style.use('seaborn')
        plt.tight_layout()
        sns.set_context('talk')
        sns.histplot(data=data, x=categorical_columns[i], hue=data['HeartDisease'], multiple="stack", palette=oe)
        plt.legend(labels=["Normal","Heart Disease"], fontsize = 10)
    plt.show()

def outlier_detection(column):
    sigma = np.std(data[column])
    mean = np.mean(data[column])
    # using six sigma for outlier analysis
    val = data[(data[column] <= mean + 3*sigma) & (data[column] >= mean - 3*sigma)]
    sns.histplot(data=val, x=column, hue=data['HeartDisease'], multiple="stack", palette=['g','r'])
    plt.legend(labels=["Normal","Heart Disease"], fontsize = 10)
    plt.show()

# outlier_detection('RestingBP')
# outlier_detection('Cholesterol')
def categorical_converter(column_name):
    global data
    finalcolumns = data[column_name].nunique() - 1 + len(data.columns) - 1
    dummies = pd.get_dummies(data[column_name], drop_first=True, prefix=column_name)
    data = pd.concat([data,dummies], axis='columns')
    data.drop(columns=column_name,axis=1, inplace=True)
    return data

for i in categorical_columns:
    categorical_converter(i)

# removal of left outliers
data = data[data['RestingBP'] >= 84]
# removal of right outliers
data = data[data['Cholesterol'] <= 500]
target = data.pop('HeartDisease')
# replacing the values of 0 cholesterol using a more realistic values
data['Cholesterol'].replace(to_replace=0, value=np.nan, inplace=True)
KNNImputed = KNNImputer(n_neighbors=5)
res = KNNImputed.fit_transform(data)
data['Cholesterol'] = res.T[2]

vif = data.copy()
vif_data = pd.DataFrame()
vif_data['Features'] = vif.columns
vif_data['VIF'] = [variance_inflation_factor(vif.values, i) for i in range(len(vif.columns))]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(vif)
scaled_pca = PCA(n_components=11)
pca_val = scaled_pca.fit_transform(scaled_data)
new_pca = pd.DataFrame(pca_val)
# loadings = scaled_pca.components_.T*np.sqrt(scaled_pca.explained_variance_)
# fig = px.scatter(pca_val, x=0, y=1, color=target)
# for i, feature in enumerate(data.columns):
#     fig.add_shape(
#         type='line',
#         x0=0, y0=0,
#         x1=loadings[i, 0],
#         y1=loadings[i, 1]
#     )
#     fig.add_annotation(
#         x=loadings[i, 0],
#         y=loadings[i, 1],
#         ax=0, ay=0,
#         xanchor="center",
#         yanchor="bottom",
#         text=feature,
#     )
# fig.show()
x_train, x_test, y_train, y_test = train_test_split(new_pca, target, test_size=0.25, random_state=0)

x_train.index = [i for i in range(len(x_train))]
y_train.index = [i for i in range(len(y_train))]
x_test.index = [i for i in range(len(x_test))]
y_test.index = [i for i in range(len(y_test))]

# support vector machine model
classifier = SVC(kernel='rbf',random_state=0)
# logistic regression model
log_reg = LogisticRegression()
# neural network model
model = keras.Sequential([
    keras.layers.Dense(units=32, activation='relu'),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dense(units=1, activation='sigmoid')
])

kfold = StratifiedKFold(n_splits = 10, shuffle=True)
i = 1
fig = plt.figure(figsize=(12,7))
accuracy = []
for train_index, val_index in kfold.split(x_train, y_train):
    X_train, X_val = x_train.loc[train_index,:], x_train.loc[val_index,:]
    Y_train, Y_val = y_train[train_index], y_train[val_index]
    # neural network training
    model.compile(optimizer='adam',loss=keras.losses.binary_crossentropy, 
        metrics=[keras.metrics.BinaryAccuracy(name='accuracy')])
    dataset_train = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(len(Y_train)).batch(10)
    dataset_val = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).shuffle(len(Y_val)).batch(10)
    history = model.fit(dataset_train, epochs=100, batch_size=10, validation_data=dataset_val, validation_steps=2)
    sns.set(font_scale=1)
    plt.subplot(3,4,i)
    plt.style.use('seaborn')
    plt.tight_layout()
    sns.set_context('talk')
    i += 1
    p=sns.lineplot(data=[history.history['loss'], history.history['val_loss']], palette=['r', 'g'])
    p.set_xlabel('Iterations', fontsize=10)
    p.set_ylabel('Loss', fontsize=10)
    plt.legend(labels=["Training loss","Validation loss"], fontsize = 10)
    accuracy.append([history.history['accuracy'], history.history['val_accuracy']])
plt.show()

fig = plt.figure(figsize=(12,7))
for i in range(len(accuracy)):
    sns.set(font_scale=1)
    plt.subplot(3,4,i+1)
    plt.style.use('seaborn')
    plt.tight_layout()
    sns.set_context('talk')
    p=sns.lineplot(data=[accuracy[i][0], accuracy[i][1]], palette=['r', 'g'])
    p.set_xlabel('Iterations', fontsize=10)
    p.set_ylabel('Accuracy', fontsize=10)
    plt.legend(labels=["Training accuracy","Validation accuray"], fontsize = 10)
plt.show()
# SVM training
fig = plt.figure(figsize=(12,7))
for j in range(10):
    classifier.fit(x_train, y_train)
    y_pred_svc = classifier.predict(x_train)
    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(classifier, x_train, y_train, train_sizes=np.array([0.25]), cv=10, shuffle=True,return_times=True)
    sns.set(font_scale=1)
    plt.subplot(3,4,j+1)
    plt.style.use('seaborn')
    plt.tight_layout()
    sns.set_context('talk')
    s = sns.lineplot(data=[train_scores[0], test_scores[0]], palette=['r', 'g'])
    s.set_xlabel('Iterations', fontsize=10)
    s.set_ylabel('Accuracy', fontsize=10)
    plt.legend(labels=["Training accuracy","Validation accuracy"], fontsize = 10)
plt.show()

# logistic regression training
fig = plt.figure(figsize=(12,7))
for j in range(10):
    log_reg.fit(x_train, y_train)
    y_pred_lr = log_reg.predict(x_test)
    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(log_reg, x_train, y_train, train_sizes=np.array([0.25]), cv=10, shuffle=True,return_times=True)
    sns.set(font_scale=1)
    plt.subplot(3,4,j+1)
    plt.style.use('seaborn')
    plt.tight_layout()
    sns.set_context('talk')
    s = sns.lineplot(data=[train_scores[0], test_scores[0]], palette=['r', 'g'])
    s.set_xlabel('Iterations', fontsize=10)
    s.set_ylabel('Accuracy', fontsize=10)
    plt.legend(labels=["Training accuracy","Validation accuracy"], fontsize = 10)
plt.show()
y_pred_nn = model.predict(x_test)

metric = tf.keras.metrics.BinaryAccuracy(threshold = 0.5)
y_pred = metric.update_state(y_test,y_pred_nn)
y_pred_lr = log_reg.predict(x_test)
y_pred_svc = classifier.predict(x_test)
nn_fpr_keras, nn_tpr_keras, nn_thresholds_keras = roc_curve(y_test, y_pred_nn)
auc_keras = auc(nn_fpr_keras, nn_tpr_keras)
plt.plot(nn_fpr_keras, nn_tpr_keras)
plt.legend(labels=['Neural Network (auc = %0.3f)' % auc_keras])
plt.show()
svc_disp = plot_roc_curve(classifier, x_test, y_test)
plt.show()
lg_disp = plot_roc_curve(log_reg, x_test, y_test)
plt.show()
print(accuracy_score(y_test, y_pred_lr))
print(accuracy_score(y_test, y_pred_svc))
print(metric.result().numpy())