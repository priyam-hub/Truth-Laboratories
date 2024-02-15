import time
import streamlit as st
import pandas as pd
import Classifier_Models.Classifier_model_builder_breast_cancer as cmb
import pickle
import numpy as np
from streamlit_toggle import st_toggle_switch
import json
from streamlit_lottie import st_lottie

def app():
    def load_lottiefile(filepath: str):
        with open(filepath, "r") as f:
            return json.load(f)
        
    lottie_coding = load_lottiefile("res/Yoga_Feet.json")
    
    st.title("Breast Cancer Predictor")
    st.info("This Model predicts whether a Female is suffering from Breast Cancer or not")
    st.markdown("""
    **Note** - :red[THIS PREDICTION MODEL IS ONLY FOR FEMALES.]
    """)

    st.sidebar.header('Report Uploader')

    uploaded_file = st.sidebar.file_uploader("Upload your parameters of your Report through a CSV File", type=["csv"])

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def patient_details():
            radius_mean = st.sidebar.number_input('Radius of Lobes', 6.98, 28.10, step=0.01)
            texture_mean = st.sidebar.number_input('Mean of Surface Texture', 9.71, 39.30, step=0.01)
            perimeter_mean = st.sidebar.number_input('Outer Perimeter of Lobes', 43.8, 189.0, step=0.1)
            area_mean = st.sidebar.number_input('Mean Area of Lobes', 144, 2510)
            smoothness_mean = st.sidebar.number_input('Mean of Smoothness Levels', 0.05, 0.16, step=0.01)
            compactness_mean = st.sidebar.number_input('Mean of Compactness', 0.02, 0.35, step=0.01)
            concavity_mean = st.sidebar.number_input('Mean of Concavity', 0.00, 0.43, step=0.01)
            concave_points_mean = st.sidebar.number_input('Mean of Cocave Points', 0.00, 0.20, step=0.01)
            symmetry_mean = st.sidebar.number_input('Mean of Symmetry', 0.11, 0.30, step=0.01)
            fractal_dimension_mean = st.sidebar.number_input('Mean of Fractal Dimension', 0.05, 0.10, step=0.01)
            radius_se = st.sidebar.number_input('SE of Radius', 0.11, 2.87, step=0.01)
            texture_se = st.sidebar.number_input('SE of Texture', 0.36, 4.88, step=0.01)
            perimeter_se = st.sidebar.number_input('Perimeter of SE', 0.76, 22.00, step=0.01)
            area_se = st.sidebar.number_input('Area of SE', 6.8, 542.0, step=0.01)
            smoothness_se = st.sidebar.number_input('SE of Smoothness', 0.00, 0.03, step=0.01)
            compactness_se = st.sidebar.number_input('SE of compactness', 0.00, 0.14, step=0.01)
            concavity_se = st.sidebar.number_input('SE of concavity', 0.00, 0.40, step=0.01)
            concave_points_se = st.sidebar.number_input('SE of concave points', 0.00, 0.05, step=0.01)
            symmetry_se = st.sidebar.number_input('SE of symmetry', 0.01, 0.08, step=0.01)
            fractal_dimension_se = st.sidebar.number_input('SE of Fractal Dimension', 0.00, 0.03, step=0.01)
            radius_worst = st.sidebar.number_input('Worst Radius', 7.93, 36.00, step=0.01)
            texture_worst = st.sidebar.number_input('Worst Texture', 12.0, 49.5, step=0.1)
            perimeter_worst = st.sidebar.number_input('Worst Permimeter', 50.40, 251.20, step=0.01)
            area_worst = st.sidebar.number_input('Worst Area', 185.20, 4250.00, step=0.01)
            smoothness_worst = st.sidebar.number_input('Worst Smoothness', 0.07, 0.22, step=0.01)
            compactness_worst = st.sidebar.number_input('Worst Compactness', 0.03, 1.06, step=0.01)
            concavity_worst = st.sidebar.number_input('Worst Concavity', 0.00, 1.25, step=0.01)
            concave_points_worst= st.sidebar.number_input('Worst Concave Points', 0.00, 0.29, step=0.01)
            symmetry_worst = st.sidebar.number_input('Worst Symmetry', 0.16, 0.66, step=0.01)
            fractal_dimension_worst = st.sidebar.number_input('Worst Fractal Dimension', 0.06, 0.21, step=0.01)

            data = {'radius_mean': radius_mean,
                    'texture_mean': texture_mean,
                    'perimeter_mean': perimeter_mean,
                    'area_mean': area_mean,
                    'smoothness_mean': smoothness_mean,
                    'compactness_mean': compactness_mean,
                    'concavity_mean': concavity_mean,
                    'concave points_mean': concave_points_mean,
                    'symmetry_mean': symmetry_mean,
                    'fractal_dimension_mean': fractal_dimension_mean,
                    'radius_se': radius_se,
                    'texture_se': texture_se,
                    'perimeter_se': perimeter_se,
                    'area_se': area_se,
                    'smoothness_se': smoothness_se,
                    'compactness_se': compactness_se,
                    'concavity_se': concavity_se,
                    'concave points_se': concave_points_se,
                    'symmetry_se': symmetry_se,
                    'fractal_dimension_se': fractal_dimension_se,
                    'radius_worst': radius_worst,
                    'texture_worst': texture_worst,
                    'perimeter_worst': perimeter_worst,
                    'area_worst': area_worst,
                    'smoothness_worst': smoothness_worst,
                    'compactness_worst': compactness_worst,
                    'concavity_worst': concavity_worst,
                    'concave points_worst': concave_points_worst,
                    'symmetry_worst': symmetry_worst,
                    'fractal_dimension_worst': fractal_dimension_worst, }

            features = pd.DataFrame(data, index=[0])
            return features

        input_df = patient_details()
    heart = cmb.X
    df = pd.concat([input_df, heart], axis=0)
    df = df[:1]  # Selects only the first row (the user input data)
    df.loc[:, ~df.columns.duplicated()]

    if uploaded_file is not None:
        st.write(df)

    else:
        st.write('Waiting for the Report to be Uploaded... Currently displaying the Parameters given manually')
        df = df.loc[:, ~df.columns.duplicated()]
        st.write(df)

    # Load the classification models
    load_clf_NB = pickle.load(open('res/pickle/breast-cancer_disease_classifier_NB.pkl', 'rb'))
    #load_clf_KNN = pickle.load(open('res/pickle/breast-cancer_disease_classifier_KNN.pkl', 'rb'))
    load_clf_DT = pickle.load(open('res/pickle/breast-cancer_disease_classifier_DT.pkl', 'rb'))
    load_clf_LR = pickle.load(open('res/pickle/breast-cancer_disease_classifier_LR.pkl', 'rb'))
    load_clf_RF = pickle.load(open('res/pickle/breast-cancer_disease_classifier_RF.pkl', 'rb'))
    
    
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
            st.write("<p style='font-size:20px; color: red'><b>I am Sorry!! You are suffering from Breast Cancer 😰</b></p>",
                     unsafe_allow_html=True)
            st.markdown("""
                        ##### Opps!! You got a Malignant Tumor!! Consult our Expert Opinion Now 😣
                        """)
        else:
            st.write("<p style='font-size:20px;color: green'><b>Hopefully you are safe!! You got a Benign Tumors.😊</b></p>",
                     unsafe_allow_html=True)
            st.markdown("""
                        ##### `A benign tumor is a non-cancerous growth of cells that does not invade nearby tissues or spread to other parts of the body `
                        """)
        
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
    #         st.write("<p style='font-size:20px; color: orange'><b>You have Malignant Tumors.</b></p>",
    #                  unsafe_allow_html=True)
    #         st.markdown("""
    #                     ##### `Malignant tumors are cancerous and have the potential to spread and invade nearby tissues or other parts of the body.`
    #                     """)
    #     else:
    #         st.write("<p style='font-size:20px;color: green'><b>You have Benign Tumors.</b></p>",
    #                  unsafe_allow_html=True)
    #         st.markdown("""
    #                     ##### `Benign tumors are non-cancerous and usually do not invade nearby tissues or spread to other parts of the body.`
    #                     """)
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
            st.write("<p style='font-size:20px; color: orange'><b>I am Sorry!! You are suffering from Breast Cancer 😰</b></p>",
                     unsafe_allow_html=True)
            st.markdown("""
                        ##### `Opps!! You got a Malignant Tumor!! Consult our Expert Opinion Now 😣`
                        """)
        else:
            st.write("<p style='font-size:20px;color: green'><b>Hopefully you are safe!! You got a Benign Tumors.😊</b></p>",
                     unsafe_allow_html=True)
            st.markdown("""
                        ##### `A benign tumor is a non-cancerous growth of cells that does not invade nearby tissues or spread to other parts of the body`
                        """)

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
            st.write("<p style='font-size:20px; color: orange'><b>I am Sorry!! You are suffering from Breast Cancer 😰</b></p>",
                     unsafe_allow_html=True)
            st.markdown("""
                        ##### `Opps!! You got a Malignant Tumor!! Consult our Expert Opinion Now 😣`
                        """)
        else:
            st.write("<p style='font-size:20px;color: green'><b>Hopefully you are safe!! You got a Benign Tumors.😊</b></p>",
                     unsafe_allow_html=True)
            st.markdown("""
                        ##### `A benign tumor is a non-cancerous growth of cells that does not invade nearby tissues or spread to other parts of the body`
                        """)
        
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
        st.subheader('Prediction by Random Forest Algorithm')
        RF_prediction = np.array([0, 1])
        if RF_prediction[prediction_RF] == 1:
            st.write("<p style='font-size:20px; color: orange'><b>I am Sorry!! You are suffering from Breast Cancer 😰</b></p>",
                     unsafe_allow_html=True)
            st.markdown("""
                        ##### `Opps!! You got a Malignant Tumor!! Consult our Expert Opinion Now 😣`
                        """)
        else:
            st.write("<p style='font-size:20px;color: green'><b>Hopefully you are safe!! You got a Benign Tumors.😊</b></p>",
                     unsafe_allow_html=True)
            st.markdown("""
                        ##### `A benign tumor is a non-cancerous growth of cells that does not invade nearby tissues or spread to other parts of the body`
                        """)
        
        #Toggle Switch
        enabled = st_toggle_switch("Show Detailed Report")
        if enabled:
            st.subheader('Report Generated by Random Forest Classifier')
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
            st.write("<p style='font-size:20px;color: green'><b>Hopefully you are safe!! You got a Benign Tumors.😊</b></p>",
                    unsafe_allow_html=True)
            st.markdown("""
                        ##### `A benign tumor is a non-cancerous growth of cells that does not invade nearby tissues or spread to other parts of the body`
                        """)

    st.markdown("👈 Provide your input data in the sidebar")
    
    # Displays the user input features
    with st.expander("Reports", expanded=False):
        # Display the input dataframe
        st.write("Your input values are shown below:")
        st.dataframe(input_df)
        
        # Call the predict_best_algorithm() function
        st.text('Showing you the Best Report Generated by our Service', help='This Report shows Approximate Prediction')
        predict_best_algorithm()

        # Tips, Diagnosis, Treatment, and Recommendations.
        st.subheader("Opinions provided by Our Consultancy on Breast Cancer 👨‍⚕️ ")
        tab1, tab2, tab3 = st.tabs(["Advices", "Diagnosis", "Therapy"])
        with tab1:
            st.subheader("Advices of Our Consultancy:")
            prevention_tips = [
                "1. Stay physically active by engaging in regular exercise, which helps to maintain a healthy weight and reduces the risk of breast cancer.",
                "2. Maintain a healthy weight, especially after menopause, as being overweight increases the risk of breast cancer.",
                "3. Follow a healthy diet rich in fruits, vegetables, whole grains, and lean proteins, and limit the intake of processed and red meats.",
                "4. Limit or avoid alcohol consumption, as excessive alcohol intake is linked to an increased risk of breast cancer.",
                "5. Minimize hormone replacement therapy (HRT) if possible, as it may increase the risk of breast cancer.",
                "6. Breastfeed your children, as breastfeeding may lower the risk of breast cancer.",
                "7. Conduct regular self-exams and mammograms as recommended by healthcare professionals for early detection and treatment of any abnormalities."
            ]
            for tip in prevention_tips:
                st.write(f"- {tip}")
        with tab2:
            st.subheader("Diagnosis Methods provided by Our Service:")
            c1, c2, c3 = st.columns([1, 1, 1], gap="small")
            with c1:
                diagnosis_methods = [
                    "1. **Clinical Breast Examination (CBE)** ",
                    "2. **Mammography** ",
                    "3. **Breast Ultrasound** ",
                    "4. **Breast MRI** ",
                    "5. **Biopsy** "
                ]
                for method in diagnosis_methods:
                    st.write(f"- {method}")
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
            st.subheader("Treatment Category Available Here:")
            treatment_options = [
                "1. **Surgery:** Lumpectomy (removal of the tumor and surrounding tissue) or mastectomy (removal of the breast tissue)",
                "2. **Radiation Therapy:** Used after surgery to destroy any remaining cancer cells and reduce the risk of recurrence",
                "3. **Chemotherapy:** Administered before or after surgery to shrink tumors, kill cancer cells, and prevent metastasis.",
                "4. **Hormone Therapy:** Blocks hormones that fuel certain types of breast cancer, often used in hormone receptor-positive breast cancers",
                "5. **Targeted Therapy:** Targets specific molecules involved in cancer cell growth and survival, such as HER2-targeted drugs for HER2-positive breast cancer",
                "6. **Immunotherapy:** Boosts the body's immune system to fight cancer cells, although its use in breast cancer is still evolving "
            ]
            for option in treatment_options:
                st.write(f"- {option}")

            st.subheader("Note:")
            st.write("Treatment decisions are based on factors like cancer stage, grade, overall health, and menopause status.")
            st.write("You can discuss your treatment options with your healthcare team and ask questions at any time.")


    # Create a multiselect for all the plot options
    selected_plots = st.multiselect("You can see all the Detailed Reports Here 👇",
                                    ["Naïve Bayes", "Decision Tree", "Logistic Regression", "Random Forest"], default=[],key="ms_B")
    if "ms_B" not in st.session_state:
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
        with st.spinner("Generating Report of Naive Bayes Classifier ...."):
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
    