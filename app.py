import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import string
import re

app = Flask(__name__)

model1 = pickle.load(open('modelEnglish.pkl', 'rb'))
model2 = pickle.load(open('modelBangla.pkl', 'rb'))
cv1 = pickle.load(open('vectorEng.pickel', 'rb'))
cv2 = pickle.load(open('vectorBan.pickel', 'rb'))
listQuery = ['annual_fee','eligibility','facilities','interest-rate','mobile-recharge','required-documents']
def check(w):
  english_check = string.printable
  return all((True if x in english_check else False for x in w))
  

def sorting(text):
  corpusTest = []
  flag = check(text)
  if(flag == True):
    sen = re.sub('[^a-zA-Z]',' ' , text) 
    sen = sen.lower() 
    sen = sen.split() 
    sen = ' '.join(sen) 
    corpusTest.append(sen)
  else:
    sen = re.sub("[^\u0980-\u09FF']+",' ' , text) 
    sen = sen.split() 
    sen = ' '.join(sen) 
    corpusTest.append(sen)
  return corpusTest,flag

def making_vector(text):
  x_corp,flag = sorting(text)
  if(flag == True):
    X_pred = cv1.transform(x_corp).toarray()
  else:
    X_pred = cv2.transform(x_corp).toarray()
  # print(X_pred)  
  return X_pred

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
  '''
  For rendering results on HTML GUI
  '''
  text = request.form.get('experience')
  X = making_vector(text)
  flag = check(text)
  if(flag == True):
      prediction = model1.predict(X)
  else:
      prediction = model2.predict(X)

  index = prediction.astype(int)
  output = listQuery[int(index[0])-1]

  return render_template('index.html', prediction_text='This query belongs to the category  {}'.format(output))


# if __name__ == "__main__":
#     app.run(debug=True)