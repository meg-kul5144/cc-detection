import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import json
import plotly.express as px
import plotly
import matplotlib.pyplot as plt
import shap
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Create flask app
flask_app = Flask(__name__, template_folder='Templates/')
model = pickle.load(open("model.pkl", "rb"))
x_test = pd.read_csv('x_test.csv')
predictions = model.predict(x_test)

with open('scaler_bal.pkl', 'rb') as f:
    scaler_bal = pickle.load(f)

with open('scaler_bill.pkl', 'rb') as f:
    scaler_bill = pickle.load(f)

with open('scaler_pay1.pkl', 'rb') as f:
    scaler_pay1 = pickle.load(f)

with open('scaler_earlymon.pkl', 'rb') as f:
    scaler_earlymon = pickle.load(f)

with open('scaler_latemon.pkl', 'rb') as f:
    scaler_latemon = pickle.load(f)

with open('scaler_pay.pkl', 'rb') as f:
    scaler_pay = pickle.load(f)

with open('scaler_bill_lt.pkl', 'rb') as f:
    scaler_bill_lt = pickle.load(f)

with open('scaler_pay0bill.pkl', 'rb') as f:
    scaler_pay0bill = pickle.load(f)

with open('scaler_pay2bill.pkl', 'rb') as f:
    scaler_pay2bill = pickle.load(f)


@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    #float_features = [float(x) for x in request.form.values()]
    Client_credit = np.array(float(request.values.get('Client credit'))).reshape(-1,1)
    LIMIT_BAL = scaler_bal.transform(pd.DataFrame(Client_credit, columns=['LIMIT_BAL']))
    
    BILL_AMT1 =  float(request.values.get('September bill'))
    bill_amt2 =  float(request.values.get('August bill'))
    bill_amt3 =  float(request.values.get('July bill'))
    bill_amt4 =  float(request.values.get('June bill'))
    bill_amt5 =  float(request.values.get('May bill'))
    bill_amt6 =  float(request.values.get('April bill'))

    average_bill = (BILL_AMT1 + bill_amt2 + bill_amt3
                     + bill_amt4 + bill_amt5 + bill_amt6)/6
    BILL_AMT1 = np.array(BILL_AMT1).reshape(-1,1)
    BILL_AMT1 =  scaler_bill.transform(pd.DataFrame(BILL_AMT1, columns=['BILL_AMT1']))
    
    
    PAY_AMT1 =  float(request.values.get('September payment'))
    pay_amt2 =  float(request.values.get('August payment'))
    pay_amt3 =  float(request.values.get('July payment'))
    pay_amt4 =  float(request.values.get('June payment'))
    PAY_AMT5 =  float(request.values.get('May payment'))
    PAY_AMT6 =  float(request.values.get('April payment'))

    average_pay = (PAY_AMT1 + pay_amt2 + pay_amt3 + pay_amt4 + PAY_AMT5 + PAY_AMT6)/6

    PAY_AMT1 =  scaler_pay1.transform(pd.DataFrame(np.array(PAY_AMT1).reshape(-1,1), columns=['PAY_AMT1']))
    PAY_AMT5 =  scaler_pay1.transform(pd.DataFrame(np.array(PAY_AMT5).reshape(-1,1), columns=['PAY_AMT1']))
    PAY_AMT6 =  scaler_pay1.transform(pd.DataFrame(np.array(PAY_AMT6).reshape(-1,1), columns=['PAY_AMT1']))


    def pay_score(pay):
        if pay <= 0:
            pay = 0
        elif pay >=1 and pay <=2:
            pay = 1
        else:
            pay =2
        return pay
    
    

    pay_0 =  pay_score(float(request.values.get('Delay in September')))
    pay_2 =  pay_score(float(request.values.get('Delay in August')))
    pay_3 =  pay_score(float(request.values.get('Delay in July')))
    pay_4 =  pay_score(float(request.values.get('Delay in June')))
    pay_5 =  pay_score(float(request.values.get('Delay in May')))
    pay_6 =  pay_score(float(request.values.get('Delay in April')))

    repayment_score_earlymonths = pay_4 + pay_5 + pay_6
    repayment_score_earlymonths = scaler_earlymon.transform(pd.DataFrame(np.array(repayment_score_earlymonths).reshape(-1,1), columns=['repayment_score_earlymonths']))

    repayment_score_latemonths = pay_0 + pay_2 + pay_3
    repayment_score_latemonths = scaler_latemon.transform(pd.DataFrame(np.array(repayment_score_latemonths).reshape(-1,1), columns=['repayment_score_latemonths']))

    payment_by_bill = average_pay/average_bill
    if average_bill == 0:
        payment_by_bill = 1
    payment_by_bill = scaler_pay.transform(np.array(payment_by_bill).reshape(-1,1))
     
    bill_by_limit = average_bill/LIMIT_BAL
    bill_by_limit = scaler_bill_lt.transform(np.array(bill_by_limit).reshape(-1,1))

    PAY0_BILLAMT1 = pay_0 * BILL_AMT1
    PAY0_BILLAMT1 = scaler_pay0bill.transform(np.array(PAY0_BILLAMT1).reshape(-1,1))
    
    PAY2_BILLAMT2 = pay_2 * bill_amt2
    PAY2_BILLAMT2 = scaler_pay2bill.transform(np.array(PAY2_BILLAMT2).reshape(-1,1))


    features = np.hstack((LIMIT_BAL, BILL_AMT1, PAY_AMT1, PAY_AMT5, PAY_AMT6,
            repayment_score_earlymonths, repayment_score_latemonths, payment_by_bill, bill_by_limit, 
                PAY0_BILLAMT1, PAY2_BILLAMT2))
    
    prediction = model.predict(pd.DataFrame(features.reshape(1, -1), columns=[x_test.columns]))
    predict_prob = model.predict_proba(pd.DataFrame(features.reshape(1, -1), columns=[x_test.columns]))

    shap_exp = shap.TreeExplainer(model)
    shap_val = shap_exp.shap_values(features)
    df = pd.DataFrame({'features': list(x_test.columns), 'values':shap_val.flatten().tolist()})
    df_sorted = df.sort_values(by='values', ascending = False)

    fig = px.bar(df_sorted, x='features', y='values', title='feature importance')

    graph_json1 = json.dumps(fig.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)


    if(prediction==1):
        prediction = "Customer is likely to default next month"
        pred = round(predict_prob.flatten()[1],3)*100
    else:
        prediction = "Customer may not default next month"
        pred = round(predict_prob.flatten()[0],3)*100

    
    return render_template("index.html", prediction_text = "Credit card default prediction:  {} with probability of  {} %".format(prediction, pred), data=features, graph_json1=graph_json1)

    

if __name__ == "__main__":

    flask_app.run(debug=True)

