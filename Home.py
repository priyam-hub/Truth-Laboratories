import streamlit as st
from res.multiapp import MultiApp
from Apps import Hypertension_App, Stroke_App, Heart_Disease, Diabetes, Breast_Cancer, \
    Kidney_App  # import your app modules here
from PIL import Image
import json
from res import Header as hd
from streamlit_lottie import st_lottie
from streamlit_extras.switch_page_button import switch_page
from st_pages import show_pages, Page

st.set_page_config(
    page_title="TRUTH LABORATORIES",
    page_icon=Image.open("images/medical-team.png"),
    layout="wide"
)

image = Image.open("images/Logo.png")
st.sidebar.image(image, use_column_width=True)

show_pages(
    [
        Page("Home.py", "DASHBOARD", "🏠"),
        Page("pages/Dataset.py", "RECORDS", ":books:"),
        Page("pages/Diagonizer.py", "EXPERT OPINION", "🏣"),
        Page("pages/Contact.py", "KEEP IN TOUCH", "✉️"),
    ]
)

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


lottie_coding = load_lottiefile("res/Logo_Final.json")

app = MultiApp()

st.markdown(
    """
    <style>
    .markdown-section {
        margin-left: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([1, 1], gap="small")
with col1:
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

    col1.empty()
with col2:
    col2.empty()
    st.title("TRUTH LABORATORIES")
    st.markdown("""

    **TRUTH LABORATORIES WELCOMES YOU ALL**, 
    
    Where every heartbeat matters, and every smile tells a story of healing. Step into a realm where expertise meets empathy, 
    and where hope is the cornerstone of our practice. Explore the corridors of knowledge, where the latest advancements in medicine merge with timeless compassion. Join us on a journey of health and wellness, guided by dedication and driven by a commitment to excellence. Together, let's embark on a path towards brighter tomorrows, where each visit brings reassurance and each interaction fosters trust. Welcome to our medical community, where your well-being is our priority, and where healing begins with a warm embrace.

    _The parameters could include_ `age, gender, lifestyle habits, genetic factors, and existing health conditions` _, among others._
    """)
    st.markdown("""Check-out our Data Records""")

    page_switch = st.button("Check out Records")
    if page_switch:
        switch_page("Dataset")

hd.colored_header(
    label="Select your disease",
    color_name="violet-70",
)


# Add all your application here
app.add_app("Breast Cancer Detector", Breast_Cancer.app)
app.add_app("Diabetes Detector", Diabetes.app)
app.add_app("Heart Disease Detector", Heart_Disease.app)
app.add_app("Hypertension Detector", Hypertension_App.app)
app.add_app("Kidney Disease Detector", Kidney_App.app)
app.add_app("Stroke Detector", Stroke_App.app)
# The main app
app.run()
