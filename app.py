import streamlit as st
import joblib
import pickle

try:
    pipeline = pickle.load(open('modelo_mejor.pkl', 'rb'))
    print("✅ Modelo cargado correctamente")
except Exception as e:
    print("❌ Error al cargar el modelo:", e)

st.title("Simulador de TOTAL_OP_(m)")