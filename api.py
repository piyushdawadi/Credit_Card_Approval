from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.impute import SimpleImputer
import joblib
import pandas as pd

# Create Flask App
# Create Flask App
app = Flask(__name__)
CORS(app, origins='*')


# Load model and load column names
model = joblib.load('final_model.pkl')
col_names = joblib.load('column_names.pkl')

# Connect Post API Call ----> predict() function
@app.route('/prediction', methods=['POST'])
def predict():
    try:
        # Get JSON Request
        feat_data = request.json
        feat_series = pd.Series(feat_data)

        # Convert JSON to Pandas DF (col names)
        df = pd.DataFrame([feat_series])
        imputer = SimpleImputer(strategy='mean')
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        df = df.reindex(columns = col_names)

        # Predict
        prediction = list(model.predict(df_imputed))

        return jsonify({'prediction':str(prediction)})
    
    except Exception as e:
        return jsonify({'error':str(e)}), 500


if __name__ == '__main__':

    app.run(debug=True)