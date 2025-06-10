import streamlit as st
import joblib

st.title("Simulador de TOTAL_OP_(m)")

try:
    pipeline = pickle.load(open('modelo_mejor.pkl', 'rb'))
    st.success("Modelo cargado correctamente.")
except Exception as e:
    st.error("Modelo NO cargado")
