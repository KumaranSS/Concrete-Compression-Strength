from flask import Flask, render_template, request,jsonify
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model.pkl",'rb'))

@app.route("/")
def main():
   return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():    
    features=[]
    for i in request.form.values():
        try:
            features.append(int(i))
        except:
            features.append(float(i))
    prediction=model.predict([features])
    return render_template('main.html',data='Compression Strength is {:.2f}'.format(float(prediction)))

    
if __name__ == "__main__":
    app.run()
