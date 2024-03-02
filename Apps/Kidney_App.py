import time
import streamlit as st
import pandas as pd
from Classifier_Models import Classifier_model_builder_kidney as cmb
from streamlit_toggle import st_toggle_switch
import pickle
import numpy as np
import json
from streamlit_lottie import st_lottie

def app():
    def load_lottiefile(filepath: str):
        with open(filepath, "r") as f:
            return json.load(f)

    lottie_coding = load_lottiefile("res/Yoga_Bhujangasana.json")
    st.title("Kidney Disease Detector")
    st.info("This Model predicts whether a person is suffering from Kidney Disease or not")

    st.sidebar.header('Report Uploader')

    uploaded_file = st.sidebar.file_uploader("Upload your parameters of your Report through a CSV File", type=["csv"])

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def patient_details():
            age = st.sidebar.number_input('Age', 2, 90)
            bp = st.sidebar.number_input('Blood Pressure', 50, 180)
            sg = st.sidebar.number_input('Salmonella Gallinarum - SG', 1.00, 1.02, step=0.01)
            al = st.sidebar.number_input('Albumin', 0, 5)
            su = st.sidebar.number_input('Sugar - SU', 0, 5)
            bgr = st.sidebar.number_input('Blood Glucose Regulator - BGR', 22, 490)
            bu = st.sidebar.number_input('Blood Urea - BU', 2, 90)
            sc = st.sidebar.number_input('Serum Creatinine - SC', 1.5, 391.0, step=0.1)
            sod = st.sidebar.number_input('Sodium', 45, 163)
            pot = st.sidebar.number_input('Potassium', 2.5, 47.0, step=0.1)
            hemo = st.sidebar.number_input('Hemoglobin', 3.1, 17.8, step=0.1)
            pcv = st.sidebar.number_input('Packed Cell Volume - PCV', 9, 54)
            wc = st.sidebar.number_input('White Blood Cell Count - WC', 3800, 26400)
            rc = st.sidebar.number_input('Red Blood Cell Count - RC', 4, 80)
            rbc = st.sidebar.selectbox('Red Blood Corpulence', ('normal', 'abnormal'))
            pc = st.sidebar.selectbox('Post Cibum - PC', ('normal', 'abnormal'))
            pcc = st.sidebar.selectbox('Prothrombin Complex Concentrates - PCC', ('present', 'notpresent'))
            ba = st.sidebar.selectbox('Bronchial Asthma - BA', ('present', 'notpresent'))
            htn = st.sidebar.selectbox('Hypertension - HTN', ('yes', 'no'))
            dm = st.sidebar.selectbox('Diabetes Mellitus', ('yes', 'no'))
            cad = st.sidebar.selectbox('Coronary Artery Disease - CAD', ('yes', 'no'))
            appet = st.sidebar.selectbox('Appetite', ('poor', 'good'))
            pe = st.sidebar.selectbox('Pulmonary Embolism - PE', ('yes', 'no'))
            ane = st.sidebar.selectbox('Acute Necrotizing Encephalopathy - ANE', ('yes', 'no'))

            data = {'age': age,
                    'rbc': rbc,
                    'pc': pc,
                    'pcc': pcc,
                    'bp': bp,
                    'ba': ba,
                    'htn': htn,
                    'dm': dm,
                    'cad': cad,
                    'appet': appet,
                    'pe': pe,
                    'ane': ane,
                    'sg': sg,
                    'al': al,
                    'su': su,
                    'bgr': bgr,
                    'bu': bu,
                    'sc': sc,
                    'sod': sod,
                    'pot': pot,
                    'hemo': hemo,
                    'pcv': pcv,
                    'wc': wc,
                    'rc': rc,
                    }

            features = pd.DataFrame(data, index=[0])
            return features

        input_df = patient_details()

    url = "res/dataset/kidney.csv"
    kidney_disease_raw = pd.read_csv(url)
    kidney_disease_raw = kidney_disease_raw.loc[:, ~kidney_disease_raw.columns.str.contains('^Unnamed')]
    kidney = kidney_disease_raw.drop(columns=['target'])
    df = pd.concat([input_df, kidney], axis=0)

    # Encoding of ordinal features
    encode = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    for col in encode:
        dummy = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummy], axis=1)
        del df[col]
    df = df[:1]  # Selects only the first row (the user input data)
    df.loc[:, ~df.columns.duplicated()]

    if uploaded_file is not None:
        st.write(df)
    else:
        st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
        df = df.loc[:, ~df.columns.duplicated()]
        st.write(df)

    # Load the classification models
    load_clf_NB = pickle.load(open('res/pickle/kidney_disease_classifier_NB.pkl', 'rb'))
    #load_clf_KNN = pickle.load(open('res/pickle/kidney_disease_classifier_KNN.pkl', 'rb'))
    load_clf_DT = pickle.load(open('res/pickle/kidney_disease_classifier_DT.pkl', 'rb'))
    load_clf_LR = pickle.load(open('res/pickle/kidney_disease_classifier_LR.pkl', 'rb'))
    load_clf_RF = pickle.load(open('res/pickle/kidney_disease_classifier_RF.pkl', 'rb'))

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
            st.write("<p style='font-size:20px;color: red'><b>I am sorryy!! You are suffering from Kidney Disease 😰</b></p>",
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
    #         st.write("<p style='font-size:20px;color: orange'><b>You have kidney disease.</b></p>",
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
            st.write("<p style='font-size:20px; color: red'><b>I am sorryy!! You are suffering from Kidney Disease 😰</b></p>",
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
            st.write("<p style='font-size:20px; color: red'><b>I am sorryy!! You are suffering from Kidney Disease 😰<b></p>",
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
        st.subheader('Prediction of Random Forest ')
        RF_prediction = np.array([0, 1])
        if RF_prediction[prediction_RF] == 1:
            st.write("<p style='font-size:20px; color: red'><b>I am sorryy!! You are suffering from Kidney Disease 😰</b></p>",
                     unsafe_allow_html=True)
        else:
            st.write("<p style='font-size:20px;color: green'><b>You are absolutely Fit 'n Fine 👍</b></p>", unsafe_allow_html=True)
        
        #Toggle Switch
        enabled = st_toggle_switch("Show Detailed Report")
        if enabled:
            st.subheader('Report Generated by Random Forest')
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
    with st.expander("Prediction Results"):
        # Display the input dataframe
        st.write("Your input values are shown below:")
        st.dataframe(input_df)
        # Call the predict_best_algorithm() function
        st.text('Showing you the Best Report Generated by our Service', help='This Report shows Approximate Prediction')
        predict_best_algorithm()

        # Tips, Diagnosis, Treatment, and Recommendations.
        st.subheader("Opinions provided by Our Consultancy on Kidney Disease 👨‍⚕️")
        tab1, tab2, tab3 = st.tabs(["Advices", "Work-Out", "Diet"])
        with tab1:
            st.subheader("Advices of Our Consultancy:")
            management_tips = [
                "1. **Manage Health Conditions:** Take steps to manage health conditions like diabetes and high blood pressure, as they are leading causes of kidney damage ",
                "2. **Maintain a Healthy Weight:** Exercise regularly and follow a balanced diet to maintain a healthy weight, which helps control blood pressure and cholesterol levels",
                "3. **Limit Alcohol Consumption:** Moderating alcohol intake helps protect kidney function and overall health",
                "4. **Monitor Medication Use:** Avoid overuse of over-the-counter pain medications, especially nonsteroidal anti-inflammatory drugs (NSAIDs), as they can harm the kidneys when used excessively",
                "5. **Quit Smoking:** Smoking reduces blood flow to the kidneys and damages kidney function. Quitting smoking can lower the risk of developing kidney disease",
                "6. **Stay Hydrated:** Drink an adequate amount of water each day to maintain proper kidney function and prevent dehydration",
                "7. **Get Regular Check-ups:** Schedule regular check-ups with your healthcare provider to monitor kidney function and address any potential issues early"
            ]
            for tip in management_tips:
                st.write(f"- {tip}")
        with tab2:
            st.subheader("Exercise Recommended by our Experts:")
            c1, c2, c3 = st.columns([1, 1, 1], gap="small")
            with c1:
                kidney_exercises = [
                    "1. **Continuous Activities**",
                    "2. **Strength Training**",
                    "3. **Brisk Walking and Cycling**",
                    "4. **Gardening and Daily Activities**",
                    "5. **Moderate Aerobic Exercise**",
                    "6. **Adapted Exercise Programs**"
                ]
                for exercise in kidney_exercises:
                    st.write(f"- {exercise}")
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
            kidney_diet = [
                "1. **Low Sodium Intake:** Limit sodium intake to help control blood pressure and reduce fluid retention ",
                "2. **Monitor Protein Consumption:** Manage protein intake as excessive consumption can strain the kidneys. Consult with a healthcare professional to determine appropriate levels",
                "3. **Reduce Phosphorus and Potassium:** Limit foods high in phosphorus and potassium, such as dairy products, nuts, seeds, and bananas, to prevent mineral imbalances",
                "4. **Choose Kidney-Friendly Carbohydrates:** Opt for whole grains and healthy carbohydrates found in fruits and vegetables while avoiding sugary foods and beverages ",
                "5. **Balanced Fluid Intake:** Maintain a balanced fluid intake based on individual needs to avoid dehydration or fluid overload",
                "6. **Limit Phosphate Additives:** Avoid processed foods containing phosphate additives, which can contribute to high phosphorus levels",
                "7. **Moderate Potassium Consumption:** Control potassium intake by choosing low-potassium foods and avoiding high-potassium options like bananas, oranges, and tomatoes",
                "8. **Manage Fluid Intake:** Monitor fluid intake, especially for patients with fluid restrictions, to prevent swelling and maintain electrolyte balance ",
                "9. **Maintain Adequate Calcium Levels:** Ensure sufficient calcium intake through dietary sources or supplements to support bone health while managing phosphorus levels "
            ]
            for diet in kidney_diet:
                st.write(f"- {diet}")

    with st.expander("Comparison Study", expanded=False):
    
        col1, col2 = st.columns(2)
        with col1:
            st.header("Naïve Bayes Classifier")
            cmb.CR_NB()
        with col2:
            st.header("Decision Tree Classifier")
            cmb.CR_DT()
    
        col3, col4 = st.columns(2)
        with col3:
            st.header("Logistic Regression Algorithm")
            cmb.CR_LR()
        with col4:
            st.header("Random Forest Algorithm")
            cmb.CR_RF()
    
    
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
        with st.spinner("Generating Report of Naïve Bayes Classifier...."):
            cmb.plt_NB()
            time.sleep(1)

    # if "K-Nearest Neighbors" in selected_plots:
    #     with st.spinner("Generating KNN...."):
    #         cmb.plt_KNN()
    #         time.sleep(1)

    if "Decision Tree" in selected_plots:
        with st.spinner("Generating Report of Decision Tree Classifier...."):
            cmb.plt_DT()
            time.sleep(1)

    if "Logistic Regression" in selected_plots:
        with st.spinner("Generating Report of Logistic Regression...."):
            cmb.plt_LR()
            time.sleep(1)

    if "Random Forest" in selected_plots:
        with st.spinner("Generating Report of Random Forest Algorithm...."):
            cmb.plt_RF()
            time.sleep(1)

    # Remove the placeholder to display the list options
    placeholder.empty()
