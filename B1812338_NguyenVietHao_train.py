import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


#load bộ dữ liệu iris
#đọc file csv
df = pd.read_csv('iris.csv')

#tạo 1 dict để đưa variety về kiểu số
varie_number = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
df = df.replace(['Setosa', 'Versicolor', 'Virginica'], [0,1,2])


#tách bỏ cột variety
X = df.drop(columns=['variety'])
#tách lấy variety
y = df['variety']

#hold-out
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3) 
# Xây dựng mô hình knn với k = 4
model = KNeighborsClassifier(n_neighbors=4)
model1 = model.fit(x_train, y_train)

#ghi vào file iri
pickle.dump(model1, open('iri.pkl', 'wb'))