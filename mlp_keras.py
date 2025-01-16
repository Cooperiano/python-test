# Load libraries
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

"""
神经网络模型
"""

# 数据读取
df_pima = pd.read_csv('diabetes.csv')
df_pima.head()

# 定义特征和标签
feature_cols = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']
X = df_pima[feature_cols] 
y = df_pima.label 

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 数据集划分(70% training, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=47)

# 模型训练
# keras文档：https://keras.io/zh/
# define
model = Sequential()
model.add(Dense(12, input_shape=(7,), activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(8, activation='relu')) 
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# compile
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0005), metrics=['accuracy'])

# fit
history = model.fit(X_train, y_train, validation_split=0.1, epochs=50, batch_size=32)

# model summary
print(model.summary())

# details on weights and bias
arr_0_weights = model.layers[0].get_weights()[0]
arr_0_bias = model.layers[0].get_weights()[1]

# 训练过程可视化
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# 在测试集上预测结果
y_pred = model.predict(X_test)

# 模型评估
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
print("AUC on the test set is:\n", auc(fpr, tpr))
