from hours import StudyHours
import uvicorn
from fastapi import FastAPI
import pickle
import pandas as pd
import numpy as  np
app=FastAPI()
pickle_in=open("model.pkl","rb")
classifier=pickle.load(pickle_in)
@app.get('/{name}')
def index():
    return {"message":f'hello'}
@app.post('/predict')
def preduct(data:StudyHours):
    data=data.dict()
    print(data)
    hours=data['hours']
    predict=classifier.predict([[hours]])
    print(predict[0][0])
    ans=predict[0][0]
    return { 'prediction':ans}

if __name__== "__main__":
    uvicorn.run(app,host='127.0.0.1',port=8000)
