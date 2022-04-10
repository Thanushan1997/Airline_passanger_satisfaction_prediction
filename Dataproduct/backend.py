from flask import Flask,request,render_template,redirect,url_for,make_response
import numpy as np 
import joblib
import pandas as pd
import csv
from io import StringIO

model = joblib.load("Xg_model.joblib")

# start flask
app = Flask(__name__)

# render default webpage
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def get_data():
        if request.method == 'POST':
            df = pd.read_csv(request.files.get('csv'))
            data = df.iloc[:,1:]
            tempdata = pd.get_dummies(data, columns=['customer_type','type_of_travel'],drop_first=True)
            columns = ['online_boarding','type_of_travel_Personal Travel','inflight_wifi_service','inflight_entertainment','onboard_service','baggage_handling','inflight_service','seat_comfort','customer_type_disloyal Customer','checkin_service']
            testdata = tempdata.loc[:,columns]
            #predict
            pred = model.predict(testdata)
            #convert it to list
            list1 = list(pred)
            #convert o and 1 to satisfaction and not satisfaction
            satisfactionlist = []
            for l in list1:
                if l == 1:
                    satisfactionlist.append("satisfied") 
                else:
                    satisfactionlist.append("not satisfied")
            output = pd.DataFrame(satisfactionlist,columns=['satisfaction'])
            #return a csv file
            resp = make_response(output.to_csv())
            resp.headers["Content-Disposition"] = "attachment; filename=export.csv"
            resp.headers["Content-Type"] = "text/csv"
            return resp

        #return render_template('index.html',data=tempdata.to_html(header=False,index=False))
        return render_template('index.html',data=output.to_html())

if __name__ == '__main__':
    app.run(debug=True,port=5000)

