import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Load the original data to fit encoders (for demonstration purposes)
# In a production app, save and load fitted encoders
try:
    original_data = pd.read_csv('/content/adult.csv')
    # Apply the same data cleaning steps as in the notebook
    original_data = original_data[original_data['workclass']!='Without-pay']
    original_data = original_data[original_data['workclass']!='Never-worked']
    original_data = original_data[original_data['education']!='5th-6th']
    original_data = original_data[original_data['education']!='1st-4th']
    original_data = original_data[original_data['education']!='Preschool']
    original_data.replace({'?':'NotListed'},inplace=True) # Handle '?' in workclass and occupation
except FileNotFoundError:
    st.error("Error: adult.csv not found. Please make sure the data file is in the correct path.")
    st.stop()


# Fit encoders for categorical columns
categorical_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
encoders = {}
for col in categorical_cols:
    encoders[col] = LabelEncoder()
    # Fit on the combined data from original and new input to handle all possible categories
    all_categories = pd.concat([original_data[col], original_data[col]]).astype(str).unique()
    encoders[col].fit(all_categories)


# Sidebar inputs (these must match your training feature columns)
st.sidebar.header("Input Employee Details")

# ✨ Replace these fields with your dataset's actual input columns
age = st.sidebar.slider("Age", 17, 75, 30) # Adjusted age range based on notebook cleaning
workclass = st.sidebar.selectbox("Workclass", encoders['workclass'].classes_)
fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", min_value=0, value=100000)
educational_num = st.sidebar.slider("Educational Number", 1, 16, 10) # Based on educational-num range
marital_status = st.sidebar.selectbox("Marital Status", encoders['marital-status'].classes_)
occupation = st.sidebar.selectbox("Occupation", encoders['occupation'].classes_)
relationship = st.sidebar.selectbox("Relationship", encoders['relationship'].classes_)
race = st.sidebar.selectbox("Race", encoders['race'].classes_)
gender = st.sidebar.selectbox("Gender", encoders['gender'].classes_)
capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, value=0)
hours_per_week = st.sidebar.slider("Hours per week", 1, 99, 40) # Adjusted range based on typical hours
native_country = st.sidebar.selectbox("Native Country", encoders['native-country'].classes_)


# Build input DataFrame (⚠️ must match preprocessing of your training data)
input_df = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'fnlwgt': [fnlwgt],
    'educational-num': [educational_num],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country]
})

# Apply the same encoding to the input data
for col in categorical_cols:
     input_df[col] = encoders[col].transform(input_df[col])

# Apply the same scaling to the input data (assuming MinMaxScaler was used)
# You would typically save and load the scaler as well
# For this example, we'll fit a new scaler on the original data's numerical columns
from sklearn.preprocessing import MinMaxScaler
numerical_cols = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
scaler = MinMaxScaler()
# Fit on original numerical data to ensure correct scaling range
scaler.fit(original_data[numerical_cols])

# Scale the numerical input features
input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])


st.write("### 🔎 Input Data (Processed)")
st.write(input_df)

# Predict button
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f"✅ Prediction: {prediction[0]}")

# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())

    # Apply the same preprocessing to batch data
    try:
        # Apply the same data cleaning steps as in the notebook
        batch_data = batch_data[batch_data['workclass']!='Without-pay']
        batch_data = batch_data[batch_data['workclass']!='Never-worked']
        batch_data = batch_data[batch_data['education']!='5th-6th']
        batch_data = batch_data[batch_data['education']!='1st-4th']
        batch_data = batch_data[batch_data['education']!='Preschool']
        batch_data.replace({'?':'NotListed'},inplace=True) # Handle '?' in workclass and occupation
    except KeyError as e:
         st.error(f"Error processing batch data: Missing column {e}. Please ensure the batch CSV has the same columns as the training data.")
         st.stop()


    # Apply encoding and scaling to batch data
    try:
        for col in categorical_cols:
             # Use the fitted encoder from the original data
             batch_data[col] = encoders[col].transform(batch_data[col])
        batch_data[numerical_cols] = scaler.transform(batch_data[numerical_cols])
    except ValueError as e:
         st.error(f"Error encoding/scaling batch data: {e}. Please ensure the batch CSV contains valid values for all features.")
         st.stop()


    batch_preds = model.predict(batch_data.drop(columns=['income'], errors='ignore')) # Drop income if present
    batch_data['PredictedClass'] = batch_preds
    st.write("✅ Predictions:")
    st.write(batch_data.head())
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
