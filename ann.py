import os
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
from sklearn import metrics


cwd = os.getcwd()
dataset = pd.read_csv(os.path.join(cwd, 'dataset/Finalized_V2.csv'))

x = dataset.iloc[:, 0: 17].values
y = dataset.iloc[:, 17].values

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)

classifier = Sequential()

# kindda definning weights
# tf.keras.layers.Dense(
#     units,
#     activation=None,
#     use_bias=True,
#     kernel_initializer="glorot_uniform",
#     bias_initializer="zeros",
#     kernel_regularizer=None,
#     bias_regularizer=None,
#     activity_regularizer=None,
#     kernel_constraint=None,
#     bias_constraint=None,
#     **kwargs
# )

classifier.add(Dense(9, input_shape=(17,),  kernel_initializer='uniform', activation='relu')) # 9 = 18/2
classifier.add(Dense(9, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

# structure the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(x_train, y_train, batch_size=10, epochs=10)

y_pred = classifier.predict(x_test)

y_pred = (y_pred > 0.5)

print(metrics.accuracy_score(y_test, y_pred))

