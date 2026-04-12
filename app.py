from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle

model        = pickle.load(open("model/best_xgb_model.pkl", "rb"))
preprocessor = pickle.load(open("model/scaler.pkl", "rb"))

app = Flask(__name__)


def feature_engineering(df):
    df["spend_per_txn"]         = df["Total_Trans_Amt_INR"]   / (df["Total_Trans_Ct"] + 1)
    df["util_x_inactive"]       = df["Avg_Utilization_Ratio"] * df["Months_Inactive_12mon"]
    df["trans_momentum"]        = df["Total_Ct_Chng_Q4_Q1"]  * df["Total_Amt_Chng_Q4_Q1"]
    df["revolving_ratio"]       = df["Total_Revolving_Bal"]   / (df["Credit_Limit_INR"] + 1)
    df["contact_inactive_flag"] = ((df["Contacts_Count_12mon"] >= 3) &
                                   (df["Months_Inactive_12mon"] >= 2)).astype(int)
    df["low_trans_flag"]        = (df["Total_Trans_Ct"] < 40).astype(int)
    df["zero_revolving"]        = (df["Total_Revolving_Bal"] == 0).astype(int)
    df["credit_per_month"]      = df["Credit_Limit_INR"] / (df["Months_on_Book"] + 1)
    return df


def validate_and_convert(data):
    try:
        data["Age"]                   = int(data["Age"])
        data["Dependent_Count"]       = int(data["Dependent_Count"])
        data["Months_on_Book"]        = int(data["Months_on_Book"])
        data["Credit_Limit_INR"]      = float(data["Credit_Limit_INR"])
        data["Total_Revolving_Bal"]   = float(data["Total_Revolving_Bal"])
        data["Avg_Open_To_Buy"]       = float(data["Avg_Open_To_Buy"])
        data["Avg_Utilization_Ratio"] = float(data["Avg_Utilization_Ratio"])
        data["Months_Inactive_12mon"] = int(data["Months_Inactive_12mon"])
        data["Contacts_Count_12mon"]  = int(data["Contacts_Count_12mon"])
        data["Total_Trans_Amt_INR"]   = float(data["Total_Trans_Amt_INR"])
        data["Total_Trans_Ct"]        = int(data["Total_Trans_Ct"])
        data["Total_Ct_Chng_Q4_Q1"]   = float(data["Total_Ct_Chng_Q4_Q1"])
        data["Total_Amt_Chng_Q4_Q1"]  = float(data["Total_Amt_Chng_Q4_Q1"])
        if data["Age"] <= 0:
            raise ValueError("Invalid age")
        return data
    except Exception as e:
        raise ValueError(f"Invalid input: {str(e)}")


@app.route("/")
def home():
    return "Credit Card Churn Prediction API is running 🚀"


@app.route("/ui")
def ui():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        data = validate_and_convert(data)
        df   = pd.DataFrame([data])
        df   = feature_engineering(df)

        X_processed  = preprocessor.transform(df)
        probability  = model.predict_proba(X_processed)[0][1]
        prediction   = 1 if probability >= 0.4 else 0

        if probability >= 0.70:
            risk = "Very High Risk"
        elif probability >= 0.50:
            risk = "High Risk"
        elif probability >= 0.35:
            risk = "Medium Risk"
        else:
            risk = "Low Risk"

        return jsonify({
            "prediction"        : int(prediction),
            "churn_probability" : float(round(probability, 4)),
            "risk_level"        : risk
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
