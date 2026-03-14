import streamlit as st
import joblib
import pandas as pd
import numpy as np

vectorizer=joblib.load("vectorizer.pkl")
model=joblib.load("sentiment_model.pkl")

st.set_page_config(layout="wide")

# Sidebar Background Color
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#1E3C72,#2A5298);
}
</style>
""", unsafe_allow_html=True)


# Bold Header Text
st.markdown("""
<style>
h1, h2, h3 {
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# Header Background Color
st.markdown("""
<style>
header[data-testid="stHeader"] {
    background: linear-gradient(
        to right,
        rgba(15,32,39,0.6),
        rgba(32,58,67,0.6),
        rgba(44,83,100,0.6)
    );
}
</style>
""", unsafe_allow_html=True)


# Page Background Color
st.markdown("""
<style>
.stApp {
background: linear-gradient(
135deg,
#0B0F2A,
#1A1F5C,
#2B2F77,
#0F5F5A,
#1E8A7A
);
}
</style>
""", unsafe_allow_html=True)


# Sidebar color, font weight and font size
st.markdown("""
<style>
section[data-testid="stSidebar"] * {
    color: white;
    font-weight: bold;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.image("Food_Sentiment_Analysis.png")
st.sidebar.title("About Project")
st.sidebar.write("""🍔
                    This project predicts whether a
                    food review is **Positive 👍 or Negative 👎**""")

st.sidebar.title("Features")  

st.sidebar.write("""
⚫ Single Review Prediction \n
⚫ Bulk Review Prediction 📂
""")

st.sidebar.title("Libraries")
st.sidebar.markdown("""
⚫ 🔢 Numpy \n
⚫ 🐼 Pandas \n
⚫ 🤖 Scikit(sklearn)
""")

st.sidebar.title("Cloud")
st.sidebar.markdown("☁️ Streamlit")

st.sidebar.title("Contact")
st.sidebar.markdown("📞9999999999")

# Header Color
st.markdown("""
<style>
h1, h2, h3, h4, h5, h6 {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# Banner Text 
st.markdown("""
<style>
.banner {
        background: linear-gradient(to right,#0F2027,#1E4D4D,#2E8B57);
        padding: 15px;
        border-radius: 10px;
        padding: 25px;
        border-radius: 10px;
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: white;
    }
</style>
<div class="banner">
Food Sentiment Analysis
</div>
""", unsafe_allow_html=True)
st.write("\n")

col1, col2 = st.columns([.4, .6])

# LEFT COLUMN
with col1:
    st.header("Predict Single Review")
    review = st.text_input("Enter Review")

    if st.button("Predict"):
        X_test = vectorizer.transform([review])
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)

        if pred[0] == 0:
            st.error("Sentiment = Negative 👎")
            st.warning(f"Confidence Score = {prob[0][0]:.2f}")
        else:
            st.success("Sentiment = Positive 👍")
            st.warning(f"Confidence Score = {prob[0][1]:.2f}")

    # st.markdown("")
    st.write("")
    st.write("")

    st.header("Predict Bulk Reviews from CSV")

    file = st.file_uploader("Select a csv file", type=["csv","txt"])

    if file:
        df = pd.read_csv(file, header=None, names=["Review"])

        placeholder = st.empty()
        placeholder.dataframe(df, hide_index=True)

        if st.button("Bulk Prediction"):
            X_test = vectorizer.transform(df.Review)
            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)

            sentiment = ["Positive" if i==1 else "Negative" for i in pred]
            df["Sentiment"] = sentiment
            df["Confidence"] = np.max(prob, axis=1)

            placeholder.dataframe(df, hide_index=True)


# RIGHT COLUMN
with col2:
    st.subheader(" Sample Format")

    sample_data = pd.DataFrame({
        "review": [
            "The food was amazing",
            "Very bad taste",
            "Loved the pizza",
            "The fries were great too",
            "A great touch"
        ]
    })

    st.dataframe(sample_data, hide_index=True)
  
    
    st.subheader("📥 Download Sample File for Bulk Prediction")

    with open("Food_review_Sample_File1.csv","rb") as f1:
            st.download_button(
            "Download Food Review Sample1.CSV",
            data=f1,
            file_name="Food_review_Sample_File1.csv",
            mime="text/csv"
        )

    with open("Food_review_Sample_File2.csv","rb") as f2:
             st.download_button(
            "Download Food Review Sample2.CSV",
            data=f2,
            file_name="Food_review_Sample_File2.csv",
            mime="text/csv"
        )
