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
        ca = request.form['ca']
        slope = request.form['slope']
        oldpeak = request.form['oldpeak']
        thal = request.form['thal']
        data = [age, sex, ct, trestbps, chol, fbs, restecg, thalach, ca, slope, oldpeak, thal]
        scaler = StandardScaler()
        scale_data = scaler.fit_transform(data)
        df = np.array(scale_data).reshape(1, -1)
        
        knn_model = open('models/KNeighborsClassifier(n_neighbors=9).pkl', 'rb')
        clf = joblib.load(knn_model)
        my_prediction = clf.predict(df)
        
    return render_template('result.html', prediction=my_prediction)




if __name__ == '__main__':
    app.run(debug=True)
    

