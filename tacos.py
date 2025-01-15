import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np
import io
from fpdf import FPDF

# Activar modo ancho completo
st.set_page_config(layout="wide")

# Ruta del archivo CSV
CSV_PATH = "Taqueria_Los_Compadres_Ventas_Extendido.csv"  # Reemplazar con la ruta local si es necesario

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

# Aplicar filtros
df_filtrado = df[
    (df['Fecha'] >= pd.Timestamp(fecha_min)) &
    (df['Fecha'] <= pd.Timestamp(fecha_max)) &
    (df['Tipo de Taco'].isin(tipo_taco))
]

# Layout del dashboard
st.title("Dashboard de Ventas - Taquería Los Compadres")
st.markdown("### Resumen interactivo de ventas")
st.markdown("Este dashboard presenta un análisis detallado de las ventas, horarios, y comportamiento de los clientes.")

# Distribuir gráficos en columnas amplias
with st.container():
    # Primera fila: 2 gráficos principales
    col1, col2 = st.columns([1, 1])

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
        st.subheader("Ventas acumuladas por fecha y proyección")
        # Cálculo de ventas totales por fecha
        ventas_fecha = df_filtrado.groupby('Fecha').agg({'Ganancia': 'sum'}).reset_index()
        if len(ventas_fecha) > 1:  # Asegurarse de que haya suficientes datos
            x = np.arange(len(ventas_fecha)).reshape(-1, 1)
            y = ventas_fecha['Ganancia'].values.reshape(-1, 1)
            modelo = LinearRegression().fit(x, y)
            
            # Generar predicciones para datos existentes y futuros
            predicciones_existentes = modelo.predict(x).flatten()
            x_futuro = np.arange(len(ventas_fecha), len(ventas_fecha) + 7).reshape(-1, 1)
            predicciones_futuras = modelo.predict(x_futuro).flatten()

            # Crear dataframe de predicción
            fechas_futuras = pd.date_range(ventas_fecha['Fecha'].iloc[-1] + pd.Timedelta(days=1), periods=7)
            df_predicciones = pd.DataFrame({
                'Fecha': list(ventas_fecha['Fecha']) + list(fechas_futuras),
                'Ganancia Proyectada': list(predicciones_existentes) + list(predicciones_futuras)
            })

            # Gráfico de proyección
            fig4 = px.line(
                df_predicciones,
                x='Fecha',
                y='Ganancia Proyectada',
                title="Ganancia acumulada y proyección futura",
                labels={'Ganancia Proyectada': 'Ganancia ($)', 'Fecha': 'Fecha'},
                color_discrete_sequence=["#EF553B"]
            )
            st.plotly_chart(fig4, use_container_width=True)

# Exportación de datos a Excel
@st.cache_data
def exportar_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Datos Filtrados')
    processed_data = output.getvalue()
    return processed_data

st.download_button(
    label="Descargar Datos en Excel",
    data=exportar_excel(df_filtrado),
    file_name="reporte_taqueria.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Exportación de reporte a PDF
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, "Reporte de Ventas - Taquería Los Compadres", 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')

def exportar_pdf():
    pdf = PDF()
    pdf.add_page()
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, "Resumen de Ventas Filtradas:", 0, 1)
    for index, row in df_filtrado.iterrows():
        pdf.cell(0, 10, f"{row['Fecha'].date()} - {row['Tipo de Taco']}: ${row['Ganancia']}", 0, 1)
    output = io.BytesIO()
    pdf.output(output)
    return output.getvalue()

st.download_button(
    label="Descargar Reporte en PDF",
    data=exportar_pdf(),
    file_name="reporte_taqueria.pdf",
    mime="application/pdf"
)
