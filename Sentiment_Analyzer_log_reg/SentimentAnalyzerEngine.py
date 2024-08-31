from flask import Flask,request,jsonify,render_template
import joblib
import pandas as pd
import datetime
import os

app = Flask(__name__) # it initialize the flask api
#model_path = os.getcwd()+r'\sentimentanalysis\models\model'
#model_path = os.getcwd()+r'/models/model'
classifier = joblib.load(r"D:\samuel testing projects\New folder\log_reg_ppl_sentiment.pkl")

def predictfunc(review):    
      
     prediction = classifier.predict(review)
     if prediction[0]=='1':
          sentiment='Positive'
     else:
          sentiment='Negative'      
     return prediction[0],sentiment

@app.route('/')   # it creates route for the root url
def index():
    return render_template('home.html')  # it is the landing page of web application.

# flask has many methods like get,put,delete,options,etc. we can use any method as per our requirement.
#here we used 'post' method to allow user to post 'data' that could be processed by server.
@app.route('/predict', methods=['POST'])  # it defines route for 'predict' url that only takes 'post' request.
def predict():
     
     if request.method == 'POST':
        result = request.form
        content = request.form['review']
        review = pd.Series(content)
        prediction,sentiment =predictfunc(review)      
     return render_template("predict.html",pred=prediction,sent=sentiment) # it gives the prediction page

if __name__ == '__main__':
     #app.run(debug = True,port=8080)
     app.run(host='0.0.0.0')