from flask import Flask, render_template, request
import pickle
import numpy as np
import time

model = pickle.load(open('iri.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a'] # lấy length sepal
    data2 = request.form['b'] # lấy width sepal
    data3 = request.form['c'] # lấy length petal
    data4 = request.form['d'] # lấy width petal
    arr = np.array([[data1, data2, data3, data4]]) #chuyển qua mảng numpy
    pred = model.predict(arr)
    return render_template('output.html', variety=pred)

#sử dụng app.run() để chạy web flask
if __name__ == "__main__":
    app.run(debug=True)