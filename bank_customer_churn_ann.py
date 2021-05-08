import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
from sklearn import metrics


cwd = os.getcwd()  # current working dir
dataset = pd.read_csv(os.path.join(cwd, 'dataset/BankCustomers.csv'))

x = dataset.iloc[:, 3:-1].values  # get all records from column 4 to last column(except last column)
y = dataset.iloc[:, -1].values  # get all records from only last column

"""
    Geography is nominal attributes -> OneHotEncoding along with PCA
    Gender is nominal -> LabelEncoding
"""
labelencoder_gender = LabelEncoder()
x[:, 2] = labelencoder_gender.fit_transform(x[:, 2])      # for gender

""" Selectively applies transformations to a specific column """
columntransform = ColumnTransformer(transformers=[('encoders', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(columntransform.fit_transform(x))[:, 1:]   # 0, 1 is enough to represent three cities

""" Model Building """

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)   # test, train sizes

""" feature scaling """

# normalization
norm = MinMaxScaler().fit(x_train)

x_train_norm = norm.fit_transform(x_train)      # training set normalization
x_test_norm = norm.fit_transform(x_test)        # testing set normalization

# standerdization
stan = StandardScaler()

x_train_stan = stan.fit_transform(x_train)
x_test_stan = stan.fit_transform(x_test)


classifier = Sequential()

classifier.add(Dense(x_train_stan.shape[1],                    # x_train.shape[1] is the # of features(
                     input_shape=(x_train_norm.shape[1],),     #it's 11 because the onehotencoder split countries into three columns and 1 removed
                     kernel_initializer='uniform',
                     activation='relu'
                     )
               )
classifier.add(Dense(x_train_stan.shape[1],                    # x_train.shape[1] is the # of features(
                     kernel_initializer='uniform',
                     activation='relu'
                     )
               )
classifier.add(Dense(1,                    # x_train.shape[1] is the # of features(
                     kernel_initializer='uniform',
                     activation='sigmoid'
                     )
               )

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(x_train_norm, y_train, batch_size=10, epochs=100)

y_pred = classifier.predict(x_test_norm)

y_pred = (y_pred > 0.5)

print('Accuracy', metrics.accuracy_score(y_test, y_pred))
