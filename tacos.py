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
CSV_PATH = "/home/nichi/Descargas/Taqueria_Los_Compadres_Ventas_Extendido.csv"  # Ruta proporcionada

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

# --- Etapa 1: Alertas e Insights Automáticos ---
st.markdown("## Alertas e Insights Automáticos")

# Proyección de ventas usando regresión lineal
st.markdown("### Proyección de Ventas")
ventas_fecha = df_filtrado.groupby('Fecha').agg({'Ganancia': 'sum'}).reset_index()

# Crear modelo de regresión lineal
if len(ventas_fecha) > 5:  # Asegurarse de tener suficientes datos
    x = np.arange(len(ventas_fecha)).reshape(-1, 1)
    y = ventas_fecha['Ganancia'].values.reshape(-1, 1)
    modelo = LinearRegression().fit(x, y)
    predicciones = modelo.predict(np.arange(len(ventas_fecha) + 7).reshape(-1, 1))

    # Crear dataframe de predicción
    fechas_futuras = pd.date_range(ventas_fecha['Fecha'].iloc[-1] + pd.Timedelta(days=1), periods=7)
    df_predicciones = pd.DataFrame({
        'Fecha': list(ventas_fecha['Fecha']) + list(fechas_futuras),
        'Ganancia Proyectada': np.concatenate([y.flatten(), predicciones.flatten()])
    })

    # Gráfico de proyección
    fig_prediccion = px.line(
        df_predicciones,
        x='Fecha',
        y='Ganancia Proyectada',
        title="Proyección de Ventas (Próximos 7 Días)",
        labels={'Ganancia Proyectada': 'Ganancia ($)', 'Fecha': 'Fecha'}
    )
    st.plotly_chart(fig_prediccion, use_container_width=True)

# Alertas: Días con ventas bajas
promedio_ganancia = ventas_fecha['Ganancia'].mean()
dias_bajos = ventas_fecha[ventas_fecha['Ganancia'] < promedio_ganancia]

st.markdown("### Alertas")
if not dias_bajos.empty:
    st.warning(f"Se detectaron {len(dias_bajos)} días con ventas por debajo del promedio.")
    st.dataframe(dias_bajos)
else:
    st.success("No hay días con ventas bajas detectados.")

# --- Etapa 2: Exportación de Reportes ---
st.markdown("## Exportación de Reportes")

# Botón para exportar a Excel
@st.cache_data
def exportar_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Datos Filtrados')
    processed_data = output.getvalue()
    return processed_data

boton_excel = st.download_button(
    label="Descargar Datos en Excel",
    data=exportar_excel(df_filtrado),
    file_name="reporte_taqueria.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Botón para exportar a PDF
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

boton_pdf = st.download_button(
    label="Descargar Reporte en PDF",
    data=exportar_pdf(),
    file_name="reporte_taqueria.pdf",
    mime="application/pdf"
)
