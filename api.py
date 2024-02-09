from flask import Flask,request,jsonify
import joblib
import pandas as pd

# Create Flask App
app = Flask(__name__)


# Connect Post API Call ----> predict() function
@app.route('/predict', methods=['POST'])
def predict():

    # Get JSON Request
    feat_data = request.json

    # Convert JSON to Pandas DF (col names)
    df =pd.DataFrame(feat_data)
    df = df.reindex(columns = col_names)

    # Predict
    prediction = list(model.predict(df))

    return jsonify({'prediction':str(prediction)})

# Load model and load column names

if __name__ == '__main__':

    model = joblib.load('final_model.pkl')
    col_names = joblib.load('column_names.pkl')

    app.run(debug=True)