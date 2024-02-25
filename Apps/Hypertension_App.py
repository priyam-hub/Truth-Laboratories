import pickle
import time
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_toggle import st_toggle_switch
from Classifier_Models import Classifier_model_builder_hypertension as cmb
import json
from streamlit_lottie import st_lottie


def app():
    def load_lottiefile(filepath: str):
        with open(filepath, "r") as f:
            return json.load(f)

    lottie_coding = load_lottiefile("res/Yoga_Padmasana.json")

    st.title("Hypertension Detector")
    st.info("This Model predicts whether a person is suffering from Hyper Tension or not")

    st.sidebar.header('Report Uploader')


    uploaded_file = st.sidebar.file_uploader("Upload your parameters of your Report through a CSV File", type=["csv"])

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def patient_details():
            age = st.sidebar.number_input('Age', 0, 98)
            sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
            chest_pain_type = st.sidebar.selectbox('Chest Pain Type',
                                                   ['Asymptomatic', 'Typical Angina', 'Atypical Angina', 'Non-anginal'])
            resting_bp = st.sidebar.number_input('Resting Blood Pressure', 94, 200)
            serum_cholesterol = st.sidebar.number_input('Serum Cholesterol', 126, 564)
            fasting_bs = st.sidebar.selectbox('Fasting Blood Sugar',
                                              ['Yes', 'No'])  # if the patient's fasting blood sugar > 120 mg/dl
            resting_ecg = st.sidebar.selectbox('Resting ECG',
                                               ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
            max_hr = st.sidebar.number_input('Max Heart Rate', 71, 202)
            exercise_angina = st.sidebar.selectbox('Exercise-Induced Angina', ['Yes', 'No'])
            oldpeak = st.sidebar.number_input('ST Depression Induced by Exercise Relative to Rest', 0.0, 6.2)
            st_slope = st.sidebar.selectbox('ST Slope', ['Upsloping', 'Flat', 'Downsloping'])
            major_vessels = st.sidebar.number_input('Number of Major Vessels Colored by Fluoroscopy', 0, 4)
            thalassemia = st.sidebar.number_input('Thalassemia', 0, 3)

            data = {'age': age,
                    'sex': sex,
                    'cp': chest_pain_type,
                    'trestbps': resting_bp,
                    'chol': serum_cholesterol,
                    'fbs': fasting_bs,
                    'restecg': resting_ecg,
                    'thalach': max_hr,
                    'exang': exercise_angina,
                    'oldpeak': oldpeak,
                    'slope': st_slope,
                    'ca': major_vessels,
                    'thal': thalassemia, }

            features = pd.DataFrame(data, index=[0])
            return features

        input_df = patient_details()

    hypertension_disease_raw = pd.read_csv('res/dataset/hypertension_data.csv')
    hypertension = hypertension_disease_raw.drop(columns=['target'])
    df = pd.concat([input_df, hypertension], axis=0)

    # Encoding of ordinal features
    encode = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope']
    for col in encode:
        dummy = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummy], axis=1)
        del df[col]
    df = df[:1]  # Selects only the first row (the user input data)
    df.loc[:, ~df.columns.duplicated()]

    if uploaded_file is not None:
        st.write(df)
    else:
        st.write('Waiting for the Report to be Uploaded... Currently displaying the Parameters given manually')
        df = df.loc[:, ~df.columns.duplicated()]
        st.write(df)

    # Load the classification models
    load_clf_NB = pickle.load(open('res/pickle/hypertension_disease_classifier_NB.pkl', 'rb'))
    #load_clf_KNN = pickle.load(open('res/pickle/hypertension_disease_classifier_KNN.pkl', 'rb'))
    load_clf_DT = pickle.load(open('res/pickle/hypertension_disease_classifier_DT.pkl', 'rb'))
    load_clf_LR = pickle.load(open('res/pickle/hypertension_disease_classifier_LR.pkl', 'rb'))
    load_clf_RF = pickle.load(open('res/pickle/hypertension_disease_classifier_RF.pkl', 'rb'))

    # Apply models to make predictions
    prediction_NB = load_clf_NB.predict(df)
    prediction_proba_NB = load_clf_NB.predict_proba(df)
    #prediction_KNN = load_clf_KNN.predict(df)
    #prediction_proba_KNN = load_clf_KNN.predict_proba(df)
    prediction_DT = load_clf_DT.predict(df)
    prediction_proba_DT = load_clf_DT.predict_proba(df)
    prediction_LR = load_clf_LR.predict(df)
    prediction_proba_LR = load_clf_LR.predict_proba(df)
    prediction_RF = load_clf_RF.predict(df)
    prediction_proba_RF = load_clf_RF.predict_proba(df)

    def NB():
        st.subheader('Prediction of Naïve Bayes Classifier')
        NB_prediction = np.array([0, 1])
        if NB_prediction[prediction_NB] == 1:
            st.write("<p style='font-size:20px;color: red'><b>I am sorryy!! You are suffering from HyperTension 😰</b></p>",
                     unsafe_allow_html=True)
        else:
            st.write("<p style='font-size:20px;color: green'><b>You are absolutely Fit 'n Fine 👍</b></p>", unsafe_allow_html=True)
        
        #Toggle Switch
        enabled = st_toggle_switch("Show Detailed Report")
        if enabled:
            st.subheader('Report Generated by Naïve Bayes Classifier')
            st.write(prediction_proba_NB)
            col1, col2 = st.columns(2)
            with col1:
                st.text('Understanding the Report',
                        help="It helps assess the model's ability to correctly identify classes and its overall performance in classifying data.")
            with col2:
                st.text('Understanding Confusion Matrix',
                        help="A confusion matrix is a performance evaluation tool in machine learning that provides a concise summary of the performance of a classification model. It presents a tabular representation of the model's predictions compared to the actual outcomes.")

            cmb.plt_NB()

    # def KNN():
    #     st.subheader('K-Nearest Neighbour Prediction')
    #     knn_prediction = np.array([0, 1])
    #     if knn_prediction[prediction_KNN] == 1:
    #         st.write("<p style='font-size:20px;color: orange'><b>You have hypertension.</b></p>",
    #                  unsafe_allow_html=True)
    #     else:
    #         st.write("<p style='font-size:20px;color: green'><b>You are fine 👍</b></p>", unsafe_allow_html=True)
    #     enabled = st_toggle_switch("See detailed prediction")
    #     if enabled:
    #         st.subheader('KNN Prediction Probability')
    #         st.write(prediction_proba_KNN)
    #         col1, col2 = st.columns(2)
    #         with col1:
    #             st.text('Why Classifier Report',
    #                     help="It helps assess the model's ability to correctly identify classes and its overall performance in classifying data.")
    #         with col2:
    #             st.text('How to read',
    #                     help="By looking at the cells where the true and predicted labels intersect, you can see the counts of correct and incorrect predictions. This helps evaluate the model's performance in distinguishing between 'No Disease' and 'Disease' categories.")

    #         cmb.plt_KNN()

    def DT():
        st.subheader('Prediction of Decision Tree Classifier')
        DT_prediction = np.array([0, 1])
        if DT_prediction[prediction_DT] == 1:
            st.write("<p style='font-size:20px; color: red'><b>I am sorryy!! You are suffering from HyperTension 😰</b></p>",
                     unsafe_allow_html=True)
        else:
            st.write("<p style='font-size:20px;color: green'><b>You are absolutely Fit 'n Fine 👍</b></p>", unsafe_allow_html=True)
        
        #Toggle Switch
        enabled = st_toggle_switch("Show Detailed Report")
        if enabled:
            st.subheader('Report Generated by Decision Tree Classifier')
            st.write(prediction_proba_DT)
            col1, col2 = st.columns(2)
            with col1:
                st.text('Understanding the Report',
                        help="It helps assess the model's ability to correctly identify classes and its overall performance in classifying data.")
            with col2:
                st.text('Understanding Confusion Matrix',
                        help="A confusion matrix is a performance evaluation tool in machine learning that provides a concise summary of the performance of a classification model. It presents a tabular representation of the model's predictions compared to the actual outcomes.")

            cmb.plt_DT()

    def LR():
        st.subheader('Prediction of Logistic Regression')
        LR_prediction = np.array([0, 1])
        if LR_prediction[prediction_LR] == 1:
            st.write("<p style='font-size:20px; color: red'><b>I am sorryy!! You are suffering from HyperTension 😰<b></p>",
                     unsafe_allow_html=True)
        else:
            st.write("<p style='font-size:20px;color: green'><b>You are absolutely Fit 'n Fine 👍</b></p>", unsafe_allow_html=True)
        
        #Toggle Switch
        enabled = st_toggle_switch("Show Detailed Report")
        if enabled:
            st.subheader('Report Generated by Logistic Regression')
            st.write(prediction_proba_LR)
            col1, col2 = st.columns(2)
            with col1:
                st.text('Understanding the Report',
                        help="It helps assess the model's ability to correctly identify classes and its overall performance in classifying data.")
            with col2:
                st.text('Understanding Confusion Matrix',
                        help="A confusion matrix is a performance evaluation tool in machine learning that provides a concise summary of the performance of a classification model. It presents a tabular representation of the model's predictions compared to the actual outcomes.")

            cmb.plt_LR()

    def RF():
        st.subheader('Prediction of Random Forest')
        RF_prediction = np.array([0, 1])
        if RF_prediction[prediction_RF] == 1:
            st.write("<p style='font-size:20px; color: orange'><b>I am sorryy!! You are suffering from HyperTension 😰</b></p>",
                     unsafe_allow_html=True)
        else:
            st.write("<p style='font-size:20px;color: green'><b>You are absolutely Fit 'n Fine 👍</b></p>", unsafe_allow_html=True)
        
        #Toggle Switch
        enabled = st_toggle_switch("Show Detailed Report")
        if enabled:
            st.subheader('Report Generated by Logistic Regression')
            st.write(prediction_proba_RF)
            col1, col2 = st.columns(2)
            with col1:
                st.text('Understanding the Report',
                        help="It helps assess the model's ability to correctly identify classes and its overall performance in classifying data.")
            with col2:
                st.text('Understanding Confusion Matrix',
                        help="A confusion matrix is a performance evaluation tool in machine learning that provides a concise summary of the performance of a classification model. It presents a tabular representation of the model's predictions compared to the actual outcomes.")
            cmb.plt_RF()

    def predict_best_algorithm():
        if cmb.best_model == 'Naive Bayes':
            NB()
        
        # elif cmb.best_model == 'K-Nearest Neighbors (KNN)':
        #     KNN()
        
        elif cmb.best_model == 'Decision Tree':
            DT()
        
        elif cmb.best_model == 'Logistic Regression':
            LR()
        
        elif cmb.best_model == 'Random Forest':
            RF()
        
        else:
            st.write("<p style='font-size:20px;color: green'><b>You are absolutely Fit 'n Fine 👍</b></p>", unsafe_allow_html=True)

    st.markdown("👈 Provide your input data in the sidebar")
    
    # Displays the user input features
    with st.expander("Prediction Results", expanded=False):
        
        # Display the input dataframe
        st.write("Your input values are shown below:")
        st.dataframe(input_df)
        
        # Call the predict_best_algorithm() function
        st.text('Showing you the Best Report Generated by our Service', help='This Report shows Approximate Prediction')
        predict_best_algorithm()

        # Tips, Diagnosis, Treatment, and Recommendations.
        st.subheader("Opinions provided by Our Consultancy on HyperTension Disease 👨‍⚕️")
        tab1, tab2, tab3 = st.tabs(["Advices", "Work-Out", "Diet"])
        with tab1:
            st.subheader("Advices of Our Consultancy:")
            management_tips = [
                "1. **Monitor Blood Pressure:** Regularly check your Blood Pressure at home and keep a record to track changes. Consult to our Experts for personalized targets and guidance",
                "2. **Adopt a healthy Diet:** Embrace a diet rich in fruits, vegetables, whole grains, lean proteins, and low fat-dairy products while limiting sodium, saturated fats, and added sugars. This can help lower blood pressure and reduce the risk of heart disease",
                "3. **Maintain a Healthy Weight:** Achieve and maintain a healthy weight through regular physical activity and a balanced diet. Losing excess weight can significantly reduce blood pressure Levels.",
                "4. **Exercise Regularly:** Engage in ,moderate-intensity aerobic exercise such as brisk walking, swimming or cycling for at least 150 minutes per week, or vigorous-intensity exercise for 75 minutes per week, combined with muscle-strengthening activities on two or more days per week",
                "5. **Quit Smoking:** If you smoke, seek support to quit smoking as it not only increases blood pressure but also raises the risk of heart disease and stroke.",
                "6. **Manage Stress:** Practice stress-reducing techniques such as deep breathing, meditation, yoga, or hobbies to help lower stress levels, which can contribute to elevated blood pressure",
                "7. **Limit Alcohol Consumption:** If you choose to drink alcohol, do so in moderation, which means up to one drink per day for women and upto two drinks per day for men. Excessive alcohol consumption can raise blood pressure."
            ]
            for tip in management_tips:
                st.write(f"- {tip}")
        with tab2:
            st.subheader("Exercise Recommended by our Experts:")
            c1, c2, c3 = st.columns([1, 1, 1], gap="small")
            with c1:
                exercise_recommendation = [
                    "**Mountain Pose (Tadasana)**",
                    "**Standing Forward Bend (Uttanasana)**",
                    "**Seated Forward Pose (Paschimottanasana)**",
                    "**Corpse Pose (Savasana)**",
                    "**Legs up the Wall Pose (Viparita Karani)**"
                ]
                for tip in exercise_recommendation:
                    st.write(f"- {tip}")
            with c3:
                st_lottie(
                    lottie_coding,
                    speed=1,
                    reverse=False,
                    loop=True,
                    quality="medium",
                    height=None,
                    width=None,
                    key=None,
                )
        with tab3:
            st.subheader("Diet Control  Measures Recommended by our Experts:")
            dietary_recommendations = [
                "1. **DASH Diet:** Follow the Dietary Approaches to Stop Hypertension(DASH Diet), which emphasizes fruits, vegetables, whole grains, lean proteins, and low-fat dairy products. This diet is rich in minerals like Potassium, Calcium, and Magnesium, which help lower Blood Pressure",
                "2. **High-Fibre Foods:** Consume foods high in soluble fiber, such as oats, bran, lentils, beans, and fruits like apples and berries. Soluble fiber can help lower cholesterol levels and improve heart health, which is beneficial for managing hypertension. ** ",
                "3. **Limit Sodium Intake:** Reduce the consumption of high-sodium foods like processed snacks, canned soups, and fast food. Excessive sodium intake can increase blood pressure levels, so aim to consume less than 2,300 milligrams of sodium per day or even lower if possible. ** ",
                "4. **Moderate Alcohol Consumption:** Limit alcohol intake to moderate levels, which is defined as up to one drink per day for women and up to two drinks per day for men. Excessive alcohol consumption can raise blood pressure and interfere with hypertension management. **",
                "5. **Healthy Fats:** Choose heart-healthy fats like those found in avocados, olive oil, nuts, and seeds. These fats can help improve cholesterol levels and reduce the risk of cardiovascular diseases associated with hypertension.** "
            ]
            for tip in dietary_recommendations:
                st.write(f"- {tip}")

    # Create a multiselect for all the plot options
    selected_plots = st.multiselect("You can see all the Detailed Reports Here 👇",
                                    ["Naïve Bayes", "Decision Tree", "Logistic Regression",
                                     "Random Forest"], default=[], key="ms_hy")
    if "ms_hy" not in st.session_state:
        st.session_state.selected_plots = []
    # Check the selected plots and call the corresponding plot functions
    if selected_plots:
        col1, col2 = st.columns(2)
        with col1:
            st.text('Understanding the Report',
                        help="It helps assess the model's ability to correctly identify classes and its overall performance in classifying data.")
            with col2:
                st.text('Understanding Confusion Matrix',
                        help="A confusion matrix is a performance evaluation tool in machine learning that provides a concise summary of the performance of a classification model. It presents a tabular representation of the model's predictions compared to the actual outcomes.")

    placeholder = st.empty()

    # Check the selected plots and call the corresponding plot functions
    if "Naïve Bayes" in selected_plots:
        with st.spinner("Generating Naïve Bayes...."):
            cmb.plt_NB()
            time.sleep(1)

    # if "K-Nearest Neighbors" in selected_plots:
    #     with st.spinner("Generating KNN...."):
    #         cmb.plt_KNN()
    #         time.sleep(1)

    if "Decision Tree" in selected_plots:
        with st.spinner("Generating Decision Tree...."):
            cmb.plt_DT()
            time.sleep(1)

    if "Logistic Regression" in selected_plots:
        with st.spinner("Generating Logistic Regression...."):
            cmb.plt_LR()
            time.sleep(1)

    if "Random Forest" in selected_plots:
        with st.spinner("Generating Random Forest...."):
            cmb.plt_RF()
            time.sleep(1)

    # Remove the placeholder to display the list options
    placeholder.empty()
