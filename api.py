# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 19:00:20 2020

@author: Shubham Shah
"""

from flask import Flask,jsonify,request
import pandas as pd
from sklearn.externals import joblib
import traceback
import numpy as np
from LogisticRegression import featurescale
app = Flask(__name__)
@app.route("/")
def index():
    """
    this is a root dir of my server
    :return: str
    """
    return "This is root!!!!"
@app.route("/predict",methods=['POST'])
def predict():
    json1 = request.get_json()
    print(json1)
    skill = json1['Skill']
    experience = json1['Experience']
    companyType = json1['CompanyType']
    education = json1['Education']
    testScore = json1['TestScore']
    f_se1 = featurescale(skill,experience,testScore)
    prediction1 = regresson1.predict_proba([[f_se1[0][0],f_se1[0][1],companyType,education,f_se1[0][2]]])
    return jsonify({'prediction':(str(prediction1))})

    
if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except:
        port = 8080
        
    regresson = joblib.load('model1.pkl')
    regresson1 = joblib.load('model2.pkl')
    
    app.run(host='192.168.43.184',port=port,debug=True)