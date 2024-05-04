import streamlit as st
from predict_page import show_predict_page
from explore_page import show_explore_page

#set up the stream lit application
st.set_page_config(layout="wide")
st.sidebar.image("Home_loan.jpeg", use_column_width=True)
page = st.sidebar.selectbox("Explore or Classify", ("Classify", "Explore"))


if page == "Classify":
    show_predict_page()
else:
    show_explore_page()