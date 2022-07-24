from asyncore import close_all
from flask import Flask
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import warnings
from flask_cors import CORS, cross_origin
app=Flask(__name__)

@app.route("/",methods=["GET","POST"])

@cross_origin()
def home():
    return render_template('index.html')
scaler=pickle.load(open("preprocessed.pkl","rb"))
cool=pickle.load(open("model_cool.pkl","rb"))
heat=pickle.load(open("model_heat.pkl","rb"))
@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():
    '''
    For rendering results on HTML GUI
    '''
    a=request.form.get("glazing_area")
    b=request.form.get("relative_compactness")
    c=request.form.get("roof_area")
    d=request.form.get("wall_area")
    df=pd.DataFrame({"glazing_area":[a],"relative_compactness":[b],"roof_area":[c],"wall_area":[d]})
    df1=pd.DataFrame(scaler.transform(df),columns=df.columns)
    cool_ans=cool.predict(df1)
    heat_ans=heat.predict(df1)
    return render_template("index.html",prediction_text="cool is "+str(cool_ans)+"and "+"heat is "+str(heat_ans))
if __name__=="__main__":
    app.run(debug=True)