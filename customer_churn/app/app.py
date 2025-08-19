from flask import Flask, render_template, request
import numpy as np
import pickle
import json

app = Flask(__name__)

def churn_prediction(tenure, citytier, warehousetohome, gender, hourspendonapp, numberofdeviceregistered, satisfactionscore, maritalstatus, numberofaddress, complain, orderamounthikefromlastyear, couponused, ordercount, daysincelastorder, cashbackamount):
    with open('models/churn_prediction_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open("models/columns.json", "r") as f:
        data_columns = json.load(f)['data_columns']

    # Convert input_data to a dictionary with appropriate keys
    input_dict = {
        "tenure": tenure,
        "citytier": citytier,
        "warehousetohome": warehousetohome,
        "gender": gender,
        "hourspendonapp": hourspendonapp,
        "numberofdeviceregistered": numberofdeviceregistered,
        "satisfactionscore": satisfactionscore,
        "maritalstatus": maritalstatus,
        "numberofaddress": numberofaddress,
        "complain": complain,
        "orderamounthikefromlastyear": orderamounthikefromlastyear,
        "couponused": couponused,
        "ordercount": ordercount,
        "daysincelastorder": daysincelastorder,
        "cashbackamount": cashbackamount
    }

    # One-hot encode categorical variables
    for col in data_columns:
        if col in input_dict and isinstance(input_dict[col], str):
            input_dict[col] = input_dict[col].lower().replace(' ', '_')

    # Create a list of zeros for all columns
    input_array = np.zeros(len(data_columns))

    # Fill the input array with the values from input_dict
    for i, col in enumerate(data_columns):
        if col in input_dict:
            input_array[i] = input_dict[col]
        elif col in input_dict.keys():
            # One-hot encode the categorical variables
            if f"{col}_{input_dict[col]}" in data_columns:
                input_array[data_columns.index(f"{col}_{input_dict[col]}")] = 1

    output_probab = model.predict_proba([input_array])[0][1]
    return round(output_probab, 4)  # Round to 4 decimal places

@app.route('/', methods=['GET', 'POST'])
def index_page():
    if request.method == 'POST':
        # Retrieve form data
        form_data = [
            request.form['Tenure'],
            request.form['Citytier'],
            request.form['Warehousetohome'],
            request.form['Gender'],
            request.form['Hourspendonapp'],
            request.form['Numberofdeviceregistered'],
            request.form['Satisfactionscore'],
            request.form['Maritalstatus'],
            request.form['Numberofaddress'],
            request.form['Complain'],
            request.form['Orderamounthikefromlastyear'],
            request.form['Couponused'],
            request.form['Ordercount'],
            request.form['Daysincelastorder'],
            request.form['Cashbackamount']
        ]

        # Convert form data to appropriate types if needed
        form_data = [int(i) if i.isdigit() else i for i in form_data]

        # Get prediction
        output_probab = churn_prediction(*form_data)

        pred = "Churn" if output_probab > 0.4 else "Not Churn"

        data = {
            'prediction': pred,
            'predict_probabality': output_probab
        }

        return render_template('result.html', data=data)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
