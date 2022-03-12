from flask import Flask,render_template,url_for,request
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

@app.route('/')
def index():    
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = request.form['age']
        sex = request.form['sex']
        ct = request.form['ct']
        trestbps = request.form['trestbps']
        chol = request.form['chol']
        fbs = request.form['fbs']
        restecg = request.form['restecg']
        thalach = request.form['thalach']
        exang = request.form['exang']
        slope = request.form['slope']
        oldpeak = request.form['oldpeak']
        ca = request.form['ca']
        thal = request.form['thal']
        data = np.array([age, sex, ct, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
        scaler = StandardScaler()
        scale_data = scaler.fit_transform(data)
        
        
        knn_model = open('models/KNeighborsClassifier(n_neighbors=9).pkl', 'rb')
        clf = joblib.load(knn_model)
        my_prediction = clf.predict(scale_data)
        
    return render_template('result.html', prediction=my_prediction)




if __name__ == '__main__':
    app.run(debug=True)
    

