# Dashboard Profesional del Dataset IRIS con Streamlit

## Estructura del proyecto

```bash
iris-dashboard/
│
├── app.py
├── requirements.txt
└── README.md
```

---

# Archivo: `app.py`

```python
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
    """
)

# =====================================================
# MÉTRICAS
# =====================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total de muestras",
        value=len(filtered_df)
    )

with col2:
    st.metric(
        label="Media Sepal Length",
        value=round(filtered_df["Sepal Length"].mean(), 2)
    )

with col3:
    st.metric(
        label="Media Petal Length",
        value=round(filtered_df["Petal Length"].mean(), 2)
    )

with col4:
    st.metric(
        label="Número de especies",
        value=filtered_df["Species"].nunique()
    )

st.divider()

# =====================================================
# TABLA DE DATOS
# =====================================================

st.subheader("Vista previa del dataset")

st.dataframe(
    filtered_df,
    use_container_width=True,
    height=350
)

st.divider()

# =====================================================
# GRÁFICOS
# =====================================================

col_g1, col_g2 = st.columns(2)

# -----------------------------------------------------
# HISTOGRAMA
# -----------------------------------------------------

with col_g1:
    st.subheader("Distribución de Sepal Length")

    fig1, ax1 = plt.subplots(figsize=(6, 4))

    cmap = plt.get_cmap("viridis")

    ax1.hist(
        filtered_df["Sepal Length"],
        bins=15,
        color=cmap(0.6),
        edgecolor="black"
    )

    ax1.set_xlabel("Sepal Length")
    ax1.set_ylabel("Frecuencia")
    ax1.grid(alpha=0.3)

    st.pyplot(fig1)

# -----------------------------------------------------
# DISPERSIÓN
# -----------------------------------------------------

with col_g2:
    st.subheader("Relación entre Petal Length y Petal Width")

    fig2, ax2 = plt.subplots(figsize=(6, 4))

    species_unique = filtered_df["Species"].unique()

    colors = np.linspace(0.2, 0.9, len(species_unique))

    for idx, specie in enumerate(species_unique):
        subset = filtered_df[filtered_df["Species"] == specie]

        ax2.scatter(
            subset["Petal Length"],
            subset["Petal Width"],
            label=specie,
            color=cmap(colors[idx]),
            s=70,
            alpha=0.8
        )

    ax2.set_xlabel("Petal Length")
    ax2.set_ylabel("Petal Width")
    ax2.legend()
    ax2.grid(alpha=0.3)

    st.pyplot(fig2)

st.divider()

# =====================================================
# MATRIZ DE CORRELACIÓN
# =====================================================

st.subheader("Matriz de correlación")

numeric_df = filtered_df.drop(columns=["Species"])
correlation = numeric_df.corr()

fig3, ax3 = plt.subplots(figsize=(8, 5))

heatmap = ax3.imshow(correlation, cmap="viridis")

ax3.set_xticks(range(len(correlation.columns)))
ax3.set_xticklabels(correlation.columns, rotation=45)

ax3.set_yticks(range(len(correlation.columns)))
ax3.set_yticklabels(correlation.columns)

for i in range(len(correlation.columns)):
    for j in range(len(correlation.columns)):
        ax3.text(
            j,
            i,
            f"{correlation.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            color="white"
        )

fig3.colorbar(heatmap)

st.pyplot(fig3)

st.divider()

# =====================================================
# ESTADÍSTICAS DESCRIPTIVAS
# =====================================================

st.subheader("Estadísticas descriptivas")

st.dataframe(
    filtered_df.describe(),
    use_container_width=True
)

# =====================================================
# FOOTER
# =====================================================

st.caption("Aplicación desarrollada con Streamlit y Scikit-Learn")
```

---

# Archivo: `requirements.txt`

```txt
streamlit
pandas
numpy
matplotlib
scikit-learn
```

---

# Archivo: `README.md`

````md
# Dashboard IRIS con Streamlit

Dashboard profesional para visualización del dataset IRIS utilizando Streamlit.

## Instalación

```bash
pip install -r requirements.txt
````

## Ejecución local

```bash
streamlit run app.py
```

## Deploy en GitHub + Streamlit Cloud

1. Crear un repositorio en GitHub.
2. Subir los archivos:

   * app.py
   * requirements.txt
   * README.md
3. Ir a Streamlit Cloud:
   [https://streamlit.io/cloud](https://streamlit.io/cloud)
4. Conectar GitHub.
5. Seleccionar el repositorio.
6. Elegir el archivo `app.py`.
7. Deploy.

```

---

# Características del dashboard

- Diseño profesional.
- Paleta de colores Viridis.
- Dashboard responsive.
- Filtros dinámicos.
- Histogramas.
- Scatter plots.
- Matriz de correlación.
- Estadísticas descriptivas.
- Compatible con Streamlit Cloud.

```


