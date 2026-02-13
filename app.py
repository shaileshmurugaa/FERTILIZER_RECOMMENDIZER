from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# -------------------------------
# 1. Initialize Flask App
# -------------------------------
app = Flask(__name__)

# -------------------------------
# 2. Load Models & Objects
# -------------------------------
kmeans_model = joblib.load("kmeans_model.pkl")
dbscan_model = joblib.load("dbscan_tuned_model.pkl")
scaler = joblib.load("scaler.pkl")
expected_feature_cols = joblib.load("expected_columns.pkl")

print("✅ Models and scaler loaded successfully")

# -------------------------------
# 3. Column Definitions
# -------------------------------
categorical_cols = ['Soil_Type', 'State', 'District', 'Crop_Type']
numerical_cols = ['Soil_Moisture', 'Humidity', 'Temperature']

# -------------------------------
# 4. Preprocessing Function
# -------------------------------
def preprocess_input_data(input_data):
    df_input = pd.DataFrame([input_data])

    # One-hot encoding
    df_encoded = pd.get_dummies(df_input, columns=categorical_cols, drop_first=True)

    # Align with training columns
    df_aligned = df_encoded.reindex(columns=expected_feature_cols, fill_value=0)

    # Scale numerical features
    df_aligned[numerical_cols] = scaler.transform(df_aligned[numerical_cols])

    return df_aligned

# -------------------------------
# 5. Home Route (HTML Render)
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -------------------------------
# 6. Prediction Route
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form.to_dict()  # for HTML form

        # Convert numeric values
        data["Soil_Moisture"] = float(data["Soil_Moisture"])
        data["Humidity"] = float(data["Humidity"])
        data["Temperature"] = float(data["Temperature"])

        processed_data = preprocess_input_data(data)

        # Hybrid Clustering Predictions
        kmeans_cluster = kmeans_model.predict(processed_data)[0]
        dbscan_cluster = dbscan_model.fit_predict(processed_data)[0]

        return render_template(
            "index.html",
            prediction_text=f"KMeans Cluster: {kmeans_cluster}, DBSCAN Cluster: {dbscan_cluster}"
        )

    except Exception as e:
        return jsonify({"error": str(e)})

# -------------------------------
# 7. Run App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
