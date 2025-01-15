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

# Layout horizontal con 4 columnas principales para los gráficos superiores
col1, col2, col3, col4 = st.columns(4)

# Gráfico 1: Tacos más vendidos
def tacos_mas_vendidos():
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
        st.caption("Este gráfico muestra los tacos más vendidos. Pastor lidera en ventas, seguido de Suadero.")

tacos_mas_vendidos()

# Gráfico 2: Ventas por día de la semana
def ventas_por_dia():
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
        st.caption("Los días viernes y sábado tienen la mayor afluencia de clientes, ideales para promociones.")

ventas_por_dia()

# Gráfico 3: Ventas por horario
def ventas_por_hora():
    with col3:
        st.subheader("Distribución de ventas por horario")
        ventas_hora = df_filtrado.groupby('Hora').agg({'Ganancia': 'sum'}).reset_index()
        fig3 = px.bar(
            ventas_hora,
            x='Hora',
            y='Ganancia',
            color='Ganancia',
            title="Distribución de ventas por horario",
            labels={'Ganancia': 'Ganancia ($)', 'Hora': 'Hora'},
            color_continuous_scale=px.colors.sequential.Sunset
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Este gráfico muestra los horarios con mayores ventas. Las horas pico son ideales para promociones.")

ventas_por_hora()

# Gráfico 4: Ventas acumuladas por fecha
def ventas_acumuladas():
    with col4:
        st.subheader("Ventas acumuladas por fecha")
        ventas_acumuladas = df_filtrado.groupby('Fecha').agg({'Ganancia': 'sum'}).reset_index()
        fig4 = px.line(
            ventas_acumuladas,
            x='Fecha',
            y='Ganancia',
            title="Ganancia acumulada por fecha",
            labels={'Ganancia': 'Ganancia Total ($)', 'Fecha': 'Fecha'},
            line_shape='spline',
            color_discrete_sequence=["#EF553B"]
        )
        st.plotly_chart(fig4, use_container_width=True)
        st.caption("Este gráfico muestra la evolución de las ganancias a lo largo del tiempo.")

ventas_acumuladas()

# Gráfico 5: Proyección de ventas (ubicado debajo de los 4 gráficos principales)
st.markdown("### Proyección de ventas (1 semana)")
def proyeccion_ventas():
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

        fig5 = px.scatter(
            ventas_fecha,
            x='Fecha',
            y='Ganancia',
            title="Proyección de ventas para la próxima semana",
            labels={'Ganancia': 'Ganancia ($)', 'Fecha': 'Fecha'},
            color_discrete_sequence=["#636EFA"]
        )
        fig5.add_scatter(
            x=df_predicciones['Fecha'],
            y=df_predicciones['Ganancia'],
            mode='lines+markers',
            name='Proyección',
            line=dict(color="#EF553B", width=2)
        )
        st.plotly_chart(fig5, use_container_width=True)
        st.caption("Proyección basada en datos históricos, limitada a la próxima semana.")

proyeccion_ventas()

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
