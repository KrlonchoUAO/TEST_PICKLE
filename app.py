import streamlit as st
import joblib
import pandas as pd

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Simulador de Predicción",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Estilos CSS para mejorar la apariencia ---
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
        border: none;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .st-emotion-cache-1r4qj8v {
        border: 2px solid #4CAF50;
        border-radius: 12px;
        padding: 20px;
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


# --- Carga del Modelo ---
# Usamos una función con caché para cargar el modelo una sola vez
@st.cache_resource
def load_model():
    """Carga el pipeline de preprocesamiento y el modelo predictivo."""
    try:
        # Asegúrate de que la ruta al archivo del modelo es correcta
        pipeline = joblib.load('modelo_mejor.pkl')
        return pipeline
    except FileNotFoundError:
        st.error("Error: El archivo 'modelo_mejor.pkl' no fue encontrado. Asegúrate de que esté en la raíz del repositorio de GitHub.")
        return None
    except Exception as e:
        st.error(f"Ocurrió un error al cargar el modelo: {e}")
        return None

pipeline = load_model()

# --- Interfaz de Usuario ---
st.title("Simulador de TOTAL_OP_(m) 📈")
st.write("Esta aplicación utiliza un modelo de Machine Learning para predecir el valor de `TOTAL_OP_(m)`. Por favor, introduce los valores de las variables a continuación.")

if pipeline is not None:
    # Creamos un contenedor para los inputs
    with st.container():
        st.header("Introduce los datos para la predicción:")

        # --- Inputs del Usuario ---
        # !!! IMPORTANTE: Debes reemplazar estos inputs de ejemplo
        # con las variables reales que tu modelo necesita.
        # Usa el tipo de input adecuado (number_input, selectbox, slider, etc.)

        # Ejemplo de variables numéricas
        feature_1 = st.number_input(
            label="Variable Numérica 1 (Ej: Edad)",
            min_value=0.0,
            max_value=100.0,
            value=25.0, # Valor por defecto
            step=1.0,
            help="Introduce un valor para la primera variable."
        )

        feature_2 = st.slider(
            label="Variable Numérica 2 (Ej: Experiencia en años)",
            min_value=0.0,
            max_value=50.0,
            value=10.0,
            step=0.5,
            help="Desliza para seleccionar un valor."
        )

        # Ejemplo de variable categórica
        feature_3 = st.selectbox(
            label="Variable Categórica 1 (Ej: Categoría)",
            options=['Opción A', 'Opción B', 'Opción C'],
            help="Selecciona una categoría de la lista."
        )

    # --- Lógica de Predicción ---
    if st.button("Predecir Valor"):
        try:
            # Crea un DataFrame de Pandas con los inputs del usuario.
            # !!! IMPORTANTE: Los nombres de las columnas DEBEN COINCIDIR
            # EXACTAMENTE con los nombres que usaste al entrenar tu modelo.
            input_data = pd.DataFrame({
                'nombre_real_variable_1': [feature_1],
                'nombre_real_variable_2': [feature_2],
                'nombre_real_variable_3': [feature_3]
                # Añade aquí todas las demás variables que tu modelo necesite
            })

            st.write("Datos de entrada para el modelo:")
            st.dataframe(input_data)

            # Realiza la predicción
            prediction = pipeline.predict(input_data)
            resultado = prediction[0]

            # Muestra el resultado
            st.success(f"**La predicción para TOTAL_OP_(m) es: `{resultado:.2f}`**")

        except Exception as e:
            st.error(f"Ocurrió un error al realizar la predicción: {e}")
else:
    st.warning("El modelo no se pudo cargar. La funcionalidad de predicción está deshabilitada.")

st.info("Nota: Esta es una simulación. Reemplaza las variables de ejemplo con las variables reales de tu modelo.")
