import streamlit as st
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import altair as alt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Url de Iris
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'


# Nombres de las columnas del dataset Iris
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']


# Cargar el dataset Iris en un DataFrame de pandas
iris_df = pd.read_csv(url, names=column_names)

st.title('Descripción del Dataset Iris')
st.write('Iris Dataset:')
st.write(iris_df.head())


#Resumen de los datos
st.write('Resumen de los datos')
resumen = resumen = iris_df.describe().T
st.write(resumen)

# Extraer las clases únicas del dataset
unique_classes = iris_df['class'].unique()

# Crear un selectbox en Streamlit
selected_class = st.selectbox("Selecciona una clase del dataset Iris", unique_classes)

# Filtrar el DataFrame basado en la clase seleccionada
filtered_df = iris_df[iris_df['class'] == selected_class]

# Mostrar las estadísticas descriptivas del DataFrame filtrado
st.write(f"Estadísticas descriptivas para la clase: {selected_class}")
st.write(filtered_df.describe().T)


# Crear un pairplot con Seaborn
pairplot = sns.pairplot(iris_df, hue="class")

# Mostrar el pairplot en Streamlit
st.write("Pairplot de los pares de variables")
st.pyplot(pairplot)
