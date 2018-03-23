import flask
from flask import Flask, render_template, jsonify, request
import pickle
import numpy as np

app = Flask(__name__)

@app.route("/")

@app.route("/index")

def index():
   return flask.render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
    	
        try:
            data = request.form
            
            years_of_experience = int(data["yearsOfExperience"])
            location = int(data["location"])
            sector = int(data["sector"])
            company_size = int(data["company_size"])

            
            lin_reg = pickle.load(open('python_lin_reg_model.pkl', 'rb'))
        except ValueError:
            return jsonify("Please enter a number.")

        result = lin_reg.predict([[ location,years_of_experience,company_size,sector ]])
        result = np.array2string(result, precision=2, separator=',',suppress_small=True)
        return render_template("predict.html",result = result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)