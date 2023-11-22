#loading in libraries
import streamlit as st
import pandas as pd
import warnings
import pickle
warnings.filterwarnings('ignore')

#preliminary app things 
st.title("Fetal Health Classification: A Machine Learning App")
st.image("fetal_health_image.gif", width=650)
st.subheader("Utilize our advanced machine learning application to predict"
             " fetal health classifications.")
st.write("To ensure optimal results, please ensure that your data strictly"
         " adheres to the specific format outlined below:")
example_df = pd.read_csv("fetal_health.csv")
example_df.head()
st.dataframe(example_df)

#reading in random forest pickle file
rf_pickle = open('rf_fetal.pickle', 'rb') 
rf_model = pickle.load(rf_pickle) 
rf_pickle.close()

#request data
file = st.file_uploader("Choose a file")
fetal_df = pd.read_csv(file)

#showing random forest model predictions on user data
st.subheader("Predicting Fetal Health Class")
new_prediction_rf = rf_model.predict(fetal_df)
new_prediction_prob_rf = rf_model.predict_proba(fetal_df).max()
fetal_df["Predicted Fetal Health"] = new_prediction_rf.tolist()
fetal_df["Prediction Probability"] = new_prediction_prob_rf.tolist()
st.dataframe(fetal_df)

#showing other ml items
st.subheader("Prediction Performance")
tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])
with tab1:
    st.image("fetal_feature_imp.svg")
with tab2:
    st.image("fetal_confusion_matrix.svg")
with tab3:
    df = pd.read_csv("fetal_class_report.csv", index_col = 0)
    st.dataframe(df)