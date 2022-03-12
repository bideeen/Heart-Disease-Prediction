from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import joblib
from sklearn.neighbors import KNeighborsClassifier


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    return render_template('result.html')




if __name__ == '__main__':
    app.run(debug=True)
    

