import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np
import io

# Activar modo ancho completo
st.set_page_config(layout="wide")

# Ruta del archivo CSV
CSV_PATH = "Taqueria_Los_Compadres_Ventas_Extendido.csv"

# Cargar datos
@st.cache_data
def cargar_datos(ruta):
    df = pd.read_csv(ruta)
    df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d-%m-%y', errors='coerce')
    df['Hora'] = pd.to_datetime(df['Hora'], format='%I:%M %p', errors='coerce').dt.hour
    return df.dropna(subset=['Fecha', 'Hora'])

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
hora_min, hora_max = st.sidebar.slider(
    "Selecciona un rango de horas:",
    min_value=0,
    max_value=23,
    value=(0, 23),
    step=1,
    format="%02d:00"
)
evento_especial = st.sidebar.selectbox(
    "Selecciona el tipo de evento especial:",
    options=["Todos", "Promoción 2x1", "Festividad Local", "Sin evento"]
)

tipo_grafico = st.sidebar.selectbox(
    "Selecciona el tipo de gráfico:",
    options=["Línea", "Barra"]
)

mostrar_proyeccion = st.sidebar.checkbox("Mostrar proyección de ventas", value=True)

# Aplicar filtros
df_filtrado = df[
    (df['Fecha'] >= pd.Timestamp(fecha_min)) &
    (df['Fecha'] <= pd.Timestamp(fecha_max)) &
    (df['Tipo de Taco'].isin(tipo_taco)) &
    (df['Hora'] >= hora_min) &
    (df['Hora'] <= hora_max)
]

if evento_especial != "Todos":
    df_filtrado = df_filtrado[df_filtrado['Día Especial'] == evento_especial]

# Resumen estadístico
st.title("Dashboard de Ventas - Taquería Los Compadres")
st.markdown("### Resumen interactivo de ventas")

# Layout horizontal con 3 columnas principales
col1, col2, col3 = st.columns([1, 1, 1])

# Columna 1: Tacos más vendidos y distribución por horario
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

    st.subheader("Distribución de ventas por horario")
    ventas_por_hora = df_filtrado['Hora'].value_counts().reset_index()
    ventas_por_hora.columns = ['Hora', 'Cantidad Vendida']
    fig2 = px.bar(
        ventas_por_hora,
        x='Hora',
        y='Cantidad Vendida',
        color='Cantidad Vendida',
        title="Ventas por horario",
        labels={'Cantidad Vendida': 'Cantidad', 'Hora': 'Hora'},
        color_continuous_scale=px.colors.sequential.Cividis
    )
    st.plotly_chart(fig2, use_container_width=True)

# Columna 2: Ventas por día de la semana
with col2:
    st.subheader("Ventas por día de la semana")
    ventas_por_dia = df_filtrado['Día de la Semana'].value_counts().reset_index()
    ventas_por_dia.columns = ['Día de la Semana', 'Cantidad Vendida']
    fig3 = px.bar(
        ventas_por_dia,
        x='Día de la Semana',
        y='Cantidad Vendida',
        color='Cantidad Vendida',
        title="Ventas por día de la semana",
        labels={'Cantidad Vendida': 'Cantidad', 'Día de la Semana': 'Día'},
        color_continuous_scale=px.colors.sequential.Viridis
    )
    st.plotly_chart(fig3, use_container_width=True)

# Columna 3: Proyección de ventas (1 semana)
with col3:
    if mostrar_proyeccion:
        st.subheader("Proyección de ventas (1 semana)")
        ventas_fecha = df_filtrado.groupby('Fecha').agg({'Ganancia': 'sum'}).reset_index()
        if len(ventas_fecha) > 1:
            x = np.arange(len(ventas_fecha)).reshape(-1, 1)
            y = ventas_fecha['Ganancia'].values.reshape(-1, 1)
            modelo = LinearRegression().fit(x, y)

            # Predicción para la próxima semana (7 días)
            dias_proyectados = 7
            x_futuro = np.arange(len(ventas_fecha), len(ventas_fecha) + dias_proyectados).reshape(-1, 1)
            predicciones = modelo.predict(x_futuro).flatten()

            fechas_futuras = pd.date_range(ventas_fecha['Fecha'].iloc[-1] + pd.Timedelta(days=1), periods=dias_proyectados)
            df_predicciones = pd.DataFrame({
                'Fecha': fechas_futuras,
                'Ganancia': predicciones
            })

            fig4 = px.scatter(
                ventas_fecha,
                x='Fecha',
                y='Ganancia',
                title="Proyección de ventas para la próxima semana",
                labels={'Ganancia': 'Ganancia ($)', 'Fecha': 'Fecha'},
                color_discrete_sequence=["#636EFA"]
            )
            fig4.add_scatter(
                x=df_predicciones['Fecha'],
                y=df_predicciones['Ganancia'],
                mode='lines+markers',
                name='Proyección',
                line=dict(color="#EF553B", width=2)
            )
            st.plotly_chart(fig4, use_container_width=True)

# Exportación de datos a Excel
@st.cache_data
def exportar_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Datos Filtrados')
    return output.getvalue()

st.download_button(
    label="Descargar Datos en Excel",
    data=exportar_excel(df_filtrado),
    file_name="reporte_taqueria.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
