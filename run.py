import os
from flask import Flask, render_template, request,jsonify
import pickle  
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        model = pickle.load(open('sentiment_model.pkl', 'rb'))
        
        print(model.predict([request.form.get('description')]))
        prediction = model.predict([request.form.get('description')])
                                                                                     
        if prediction == 0:
            result = "Negative"
            emoji = "😶😑😪🥺"
        else:
            result = "Positive"
            emoji = "😂😊😉🤓"
        print(result)
        output = result
    else:
        emoji = ""
        output = ""
    return render_template('index.html', output=output, emoji=emoji)

@app.route('/predict',methods=['GET'])
def predictAPI():
    model = pickle.load(open('sentiment_model.pkl', 'rb'))
    val = model.predict([request.form.get('description')])
    
    return jsonify({"result":str(val[0])})

if __name__ == '__main__':
    app.run(debug=True)

