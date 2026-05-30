import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# =====================================================
# CONFIGURACIÓN GENERAL
# =====================================================

st.set_page_config(
    page_title="Dashboard IRIS",
    page_icon="🌸",
    layout="wide"
)

# =====================================================
# CARGA DE DATOS
# =====================================================

iris = load_iris()

columns = [
    "Sepal Length",
    "Sepal Width",
    "Petal Length",
    "Petal Width"
]

species_names = iris.target_names


df = pd.DataFrame(iris.data, columns=columns)
df["Species"] = [species_names[i] for i in iris.target]

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Filtros")

selected_species = st.sidebar.multiselect(
    "Seleccionar especies",
    options=df["Species"].unique(),
    default=df["Species"].unique()
)

filtered_df = df[df["Species"].isin(selected_species)]

# =====================================================
# TÍTULO PRINCIPAL
# =====================================================

st.title("🌸 Dashboard Profesional - Dataset IRIS")

st.markdown(
    """
    Este dashboard permite explorar visualmente el dataset IRIS mediante
    métricas estadísticas y gráficos interactivos.
    """)
st.caption("Aplicación desarrollada con Streamlit y Scikit-Learn")
