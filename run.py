import os
from flask import Flask, render_template, request,jsonify
import pickle  
import numpy as np

app = Flask(__name__)

MODEL_LABELS = ['0', '1']

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
def predict():
    model = pickle.load(open('sentiment_model.pkl', 'rb'))

    # Retrieve query parameters related to this request.
    description = request.args.get('description')

    # Our model expects a list of records
    features = ([description])
    
    # Use the model to predict the class
    prediction = model.predict(features)
    # Retrieve the emotion that is associated with the predicted class
    result = MODEL_LABELS[prediction[0]]
    # Create and send a response to the API caller
    return jsonify(result=result)

if __name__ == '__main__':
    app.run(debug=True)

