from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the full pipeline
model = joblib.load('flight_price_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    airline = request.form['airline']
    source = request.form['source']
    destination = request.form['destination']
    stops = int(request.form['stops'])
    dep_date = pd.to_datetime(request.form['date'])

    day = dep_date.day
    month = dep_date.month

    input_df = pd.DataFrame([{
        'Airline': airline,
        'Source': source,
        'Destination': destination,
        'Total_Stops': stops,
        'Journey_day': day,
        'Journey_month': month
    }])

    prediction = model.predict(input_df)[0]
    return render_template('index.html', prediction=f'Predicted Flight Price: ₹{round(prediction, 2)}')

if __name__ == '__main__':
    app.run(debug=True)
