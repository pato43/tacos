# Archivo: dashboard_taqueria_horizontal.py

import streamlit as st
import pandas as pd
import plotly.express as px

# Activar modo ancho completo
st.set_page_config(layout="wide")

# Ruta del archivo CSV
CSV_PATH = "Taqueria_Los_Compadres_Ventas_Extendido.csv"  # Ruta proporcionada

# Cargar datos
@st.cache_data
def cargar_datos(ruta):
    df = pd.read_csv(ruta)
    df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d-%m-%y', errors='coerce')
    return df.dropna(subset=['Fecha'])

df = cargar_datos(CSV_PATH)

# Sidebar con filtros avanzados
st.sidebar.title("Filtros")
fecha_min, fecha_max = st.sidebar.date_input(
    "Selecciona un rango de fechas:",
    [df['Fecha'].min(), df['Fecha'].max()]
)
tipo_taco = st.sidebar.multiselect(
    "Selecciona los tipos de taco:",
    options=df['Tipo de Taco'].unique(),
    default=df['Tipo de Taco'].unique()
)
dia_semana = st.sidebar.multiselect(
    "Selecciona los días de la semana:",
    options=df['Día de la Semana'].unique(),
    default=df['Día de la Semana'].unique()
)
evento_especial = st.sidebar.selectbox(
    "Filtrar por evento especial:",
    options=["Todos"] + df['Día Especial'].unique().tolist()
)

# Aplicar filtros
df_filtrado = df[
    (df['Fecha'] >= pd.Timestamp(fecha_min)) &
    (df['Fecha'] <= pd.Timestamp(fecha_max)) &
    (df['Tipo de Taco'].isin(tipo_taco)) &
    (df['Día de la Semana'].isin(dia_semana))
]

if evento_especial != "Todos":
    df_filtrado = df_filtrado[df_filtrado['Día Especial'] == evento_especial]

# Layout horizontal optimizado
st.title("Dashboard de Ventas - Taquería Los Compadres")
st.markdown("### Resumen interactivo de ventas")
st.markdown("Este dashboard presenta un análisis detallado de las ventas, horarios, y comportamiento de los clientes.")

# Distribuir gráficos en columnas amplias
with st.container():
    # Primera fila: 2 gráficos principales
    col1, col2 = st.columns([1, 1])  # Ajustar proporciones igualitarias

    with col1:
        st.subheader("Tacos más vendidos")
        tacos_populares = df_filtrado['Tipo de Taco'].value_counts().reset_index()
        tacos_populares.columns = ['Tipo de Taco', 'Cantidad Vendida']
        fig1 = px.bar(
            tacos_populares,
            x='Tipo de Taco',
            y='Cantidad Vendida',
            color='Cantidad Vendida',
            title="Cantidad por tipo de taco",
            labels={'Cantidad Vendida': 'Cantidad', 'Tipo de Taco': 'Taco'},
            color_continuous_scale=px.colors.sequential.Plasma
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Ventas por día de la semana")
        ventas_por_dia = df_filtrado['Día de la Semana'].value_counts().reset_index()
        ventas_por_dia.columns = ['Día de la Semana', 'Cantidad Vendida']
        fig2 = px.bar(
            ventas_por_dia,
            x='Día de la Semana',
            y='Cantidad Vendida',
            color='Cantidad Vendida',
            title="Ventas por día de la semana",
            labels={'Cantidad Vendida': 'Cantidad', 'Día de la Semana': 'Día'},
            color_continuous_scale=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig2, use_container_width=True)

with st.container():
    # Segunda fila: Gráfico de horarios y acumulado por fecha
    col3, col4 = st.columns([1, 1])

    with col3:
        st.subheader("Distribución de ventas por horario")
        ventas_por_hora = df_filtrado['Hora'].value_counts().reset_index()
        ventas_por_hora.columns = ['Hora', 'Cantidad Vendida']
        fig3 = px.bar(
            ventas_por_hora,
            x='Hora',
            y='Cantidad Vendida',
            color='Cantidad Vendida',
            title="Distribución de ventas por horario",
            labels={'Cantidad Vendida': 'Cantidad', 'Hora': 'Horario'},
            color_continuous_scale=px.colors.sequential.Cividis
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("Ventas acumuladas por fecha")
        # Corrección del cálculo de ventas totales por fecha
        ventas_fecha = df_filtrado.groupby('Fecha').agg({'Ganancia': 'sum'}).reset_index()
        fig4 = px.line(
            ventas_fecha,
            x='Fecha',
            y='Ganancia',
            title="Ganancia acumulada por fecha",
            labels={'Ganancia': 'Ganancia Total', 'Fecha': 'Fecha'},
            color_discrete_sequence=["#EF553B"]
        )
        st.plotly_chart(fig4, use_container_width=True)

# Descripción adicional
st.markdown("### Datos Filtrados")
st.dataframe(df_filtrado, use_container_width=True)
