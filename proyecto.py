import streamlit as st
import pandas as pd
import traceback
import io
import plotly.express as px
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from fpdf import FPDF
import matplotlib.pyplot as plt
import base64

contenido = {}

def guardar_dataframe_como_imagen(dataframe, nombre_imagen):
    """
    Guarda un DataFrame como una imagen con estilos personalizados.

    Parámetros:
    - dataframe (pd.DataFrame): El DataFrame que se desea guardar como imagen.
    - nombre_imagen (str): Nombre del archivo de imagen, incluyendo la extensión (.png, .jpg, etc.).

    Retorna:
    - None
    """
    # Configurar la figura
    fig, ax = plt.subplots(figsize=(len(dataframe.columns) * 2, len(dataframe) * 0.5 + 1))  # Tamaño ajustado
    ax.axis('off')  # Quitar los ejes

    # Crear la tabla
    tabla = ax.table(cellText=dataframe.values, colLabels=dataframe.columns, loc='center')

    # Estilizar la cabecera (negritas)
    for (row, col), cell in tabla.get_celld().items():
        if row == 0:  # Cabecera
            cell.set_text_props(weight='bold')  # Negritas
        # Ajustar la altura de las celdas (20% más grande)
        cell.set_height(cell.get_height() * 1.2)

    # Estilizar el tamaño de fuente
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.auto_set_column_width(col=list(range(len(dataframe.columns))))

    # Guardar la tabla como imagen
    plt.savefig(nombre_imagen, dpi=300, bbox_inches='tight')
    plt.close()

def create_download_link(val, filename):
        b64 = base64.b64encode(val)  # val looks like b'...'
        return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Presione para descargar su informe</a>'

def generar_pdf(contenido:dict):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Configuración del título
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Informe de Datos", ln=True, align='C')

    # Contenido del informe
    pdf.ln(7)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 6, txt="En este informe automatizado podremos encontrar estadísticas descriptivas, visualizaciones y modelos de regresión en base a los puntos seleccionados en Streamlit.")
  
    # Informacion de las columnas
    pdf.ln(5) 
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="Informacion de las columnas")
    pdf.ln(10)
    pdf.set_font("Arial", size=9)
    pdf.multi_cell(0, 6, txt="En esta sección encontraremos información básica sobre el tipo de dato de cada columna y la cantidad de elementos que tiene.")
    pdf.ln(10)
    pdf.multi_cell(0, 5, txt=contenido['info'])
    
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="Matriz de correlacion")
    pdf.ln(10)  # Salto de línea antes de la imagen
    pdf.set_font("Arial", size=9)
    pdf.multi_cell(0, 6, txt="Una matriz de correlación no es más que una tabla con los coeficientes de correlación de distintas variables. La matriz muestra cómo se relacionan entre sí todos los posibles pares de valores de una tabla. Es una poderosa herramienta para resumir un gran conjunto de datos y encontrar y mostrar patrones en ellos.\nA menudo se muestra como una tabla, con cada variable enumerada tanto en las filas como en las columnas y el coeficiente de correlación entre cada par de variables escrito en cada celda. El coeficiente de correlación oscila entre -1 y +1, donde -1 significa una correlación negativa perfecta, +1 significa una correlación positiva perfecta y 0 significa que no hay correlación entre las variables.")
    pdf.ln(10)
    pdf.image("corr_matrix.png", x=10, y=pdf.get_y(), w=180)
    
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="Estadisticas descriptivas")
    pdf.ln(10)
    pdf.set_font("Arial", size=9)
    pdf.multi_cell(0, 6, txt="En esta seccion encontraremos las principales estadisticas de las columnas numericas.")
    df = contenido['descripcion']
    df = df.reset_index()
    df = df.round(3)
    guardar_dataframe_como_imagen(df,"tabla.png")
    pdf.image("tabla.png", x=10, y=pdf.get_y(), w=180)
    
    pdf.ln(90)
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(0, 10, txt="Histograma")
    pdf.ln(10)
    pdf.set_font("Arial", size=9)
    pdf.multi_cell(0, 6, txt="Un histograma es una representación gráfica de una variable en forma de barras, donde la superficie de cada barra es proporcional a la frecuencia de los valores representados. Sirven para obtener una 'primera vista' general, o panorama, de la distribución de la población, o de la muestra, respecto a una característica, cuantitativa y continua")
    pdf.ln(5)
    pdf.set_font("Arial", style="B", size=9) 
    pdf.cell(0, 10, txt="Configuracion del histograma: ")
    pdf.set_font("Arial", style="", size=9) 
    pdf.ln(6)
    pdf.cell(0, 10, txt="Variable             --> "+contenido['var_hist'])
    pdf.ln(6)
    pdf.cell(0, 10, txt="Rango de la variable --> "+str(contenido['rango_hist']))
    pdf.ln(6) 
    pdf.cell(0, 10, txt="Bins del histograma  --> "+ str(contenido['bins_hist']))
    pdf.image("hist.png", x=40, y=pdf.get_y()+10, w=130)
    
    pdf.add_page()
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(0, 10, txt="Dispersion")
    pdf.ln(10)
    pdf.set_font("Arial", size=9)
    pdf.multi_cell(0, 6, txt="Un diagrama de dispersión es un tipo de diagrama matemático que utiliza las coordenadas cartesianas para mostrar los valores de dos variables para un conjunto de datos.")
    pdf.ln(5)
    pdf.set_font("Arial", style="B", size=9) 
    pdf.cell(0, 10, txt="Configuracion del diagrama: ")
    pdf.set_font("Arial", style="", size=9) 
    pdf.ln(6)
    pdf.cell(0, 10, txt="Eje X --> " + contenido['eje_x_disp'])
    pdf.ln(6) 
    pdf.cell(0, 10, txt="Eje Y --> "+ contenido['eje_y_disp'])
    pdf.image("disp.png", x=40, y=pdf.get_y()+10, w=130)
    
    pdf.add_page()
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(0, 10, txt="Diagrama Pastel")
    pdf.ln(10)
    pdf.set_font("Arial", size=9)
    pdf.multi_cell(0, 6, txt="Una gráfica de pastel es un recurso estadístico que se utiliza para representar porcentajes y proporciones. El número de elementos comparados dentro de una gráfica circular suele ser de más de cuatro.")
    pdf.ln(5)
    pdf.set_font("Arial", style="B", size=9) 
    pdf.cell(0, 10, txt="Configuracion del diagrama: ")
    pdf.set_font("Arial", style="", size=9) 
    pdf.ln(6)
    pdf.cell(0, 10, txt="Dimension --> " + contenido['dim_pastel'])
    pdf.image("pie.png", x=40, y=pdf.get_y()+10, w=130)
    
    modelo = contenido['modelo']
    pdf.add_page()
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(0, 10, txt="Regresion: " + modelo)    
    pdf.ln(10)
    pdf.set_font("Arial", size=9)
    pdf.multi_cell(0, 6, txt="El análisis de la regresión es un proceso estadístico para entender cómo una variable depende de otra variable.")
    pdf.ln(5)
    pdf.set_font("Arial", style="B", size=9) 
    pdf.cell(0, 10, txt="Configuracion de la regresion: ")
    pdf.set_font("Arial", style="", size=9) 
    pdf.ln(6)
    pdf.cell(0, 10, txt="Regresion seleccionada    --> " + modelo)
    pdf.ln(6)
    pdf.cell(0, 10, txt="Variable objetivo         --> " + contenido['target'])
    pdf.ln(6)
    pdf.cell(0, 10, txt="Variables independientes  --> " + contenido['independientes'])
    pdf.ln(6)
    pdf.cell(0, 10, txt="Porcentaje datos testing  --> " + str(contenido['porcentaje']))
    
    pdf.ln(10)
    pdf.set_font("Arial", style="B", size=9) 
    pdf.cell(0, 10, txt="Resultados: ")
    pdf.set_font("Arial", style="", size=9) 
    pdf.ln(6)
    pdf.cell(0, 10, txt="Acurracy (R2score)  --> " + contenido['accuracy'])
    pdf.ln(6)
    pdf.cell(0, 10, txt="Mean Absolute Error  --> " + contenido['mae'])
    pdf.ln(6)
    pdf.cell(0, 10, txt="Mean Squared Error   --> " + contenido['mse'])
    pdf.ln(15)
    pdf.set_font("Arial", style="B", size=9) 
    pdf.cell(0, 10, txt="Muestra de datos seleccionado: ")
    pdf.ln(10)
    pdf.image("entrenamiento.png", x=10, y=pdf.get_y(), w=180)
    '''
    pdf.set_font("Arial", 'B', 9)
    pdf.set_fill_color(200, 220, 255)  # Color de fondo para los encabezados
    for col in df.columns:
        pdf.cell(30, 5, col, border=1, align='C', fill=True)
    
    pdf.ln()  # Salto de línea después de los encabezados
    pdf.set_font("Arial", size = 9)
    # Agregar las filas de datos del DataFrame
    for i in range(len(df)):
        for j in df.columns:
            pdf.cell(30, 5, str(df[j][i]), border=1, align='C')
        pdf.ln()  # Salto de línea después de cada fila
    '''
    
    return pdf

print("$"*80)
st.set_page_config(
    page_title="Aplicacion de analisis de datos",
    layout="wide"
    )


st.title('Proyecto de la materia')
st.title('Renato Jara')

col1, col2, col3 = st.columns(spec=[0.3,0.4,0.3],gap="medium")

# Barra Lateral (st.sidebar)
st.sidebar.header("Configuración")
archivo = st.sidebar.file_uploader("Cargar un archivo", type=["csv"])


if archivo is not None:
    st.sidebar.subheader('Generar Informe en PDF')
    st.sidebar.write(":red[NOTA:] Para generar un informe, todos los diagramas deben estar correctamente configurados, incluida la seccion de Regresion")
    filename = st.sidebar.text_input("Nombre del archivo (sin extension)")
    export_as_pdf = st.sidebar.button("Generar link de descarga")
        
    ######################## INFORMACION GENERAL ######################
    with col1:
        st.title("1. Informacion general")
        df_cargado = pd.read_csv(archivo,delimiter=",")
        df_vis_hist = df_cargado.copy()
        df_vis_scat = df_cargado.copy()
        df_regresion = df_cargado.copy().dropna()
        columnas_numericas = df_cargado.select_dtypes(include=['number']).columns.tolist()
        columnas_text = df_cargado.select_dtypes(include=['object', 'string']).columns.tolist()

        st.subheader("5 primeros valores del archivo original")
        st.write(df_cargado.head(5))
        st.subheader("Informacion de las columnas")
        buffer = io.StringIO()
        df_cargado.info(buf=buffer)
        info_str = buffer.getvalue()
        st.text(info_str)
        contenido['info'] = info_str
        st.subheader("Correlacion")
        fig_corr=px.imshow(df_cargado[columnas_numericas].corr(),color_continuous_scale='Viridis')
        fig_corr.write_image("corr_matrix.png")
        st.plotly_chart(fig_corr,use_container_width=True)
        
        st.subheader("Estadistica descriptivas")
        columnas = st.multiselect(
            label="Escoja las columnas para analizar:",
            options=columnas_numericas
            )
        try:
            st.write(df_cargado[columnas].describe())
            contenido['descripcion']  = df_cargado[columnas].describe()
        
        except ValueError as e:
            st.write("Solo se permiten columnas numéricas:")
        
        except Exception as e:
            print(df_cargado)
            traceback.print_exc()
            st.write("El siguiente error ha ocurrido: " + str(e))
        
    ############### VISUALIZACION ###################
    with col2:
        st.title("2. Visualizacion de datos")
        st.subheader("Histograma")
        eje_x = st.selectbox(label="Escoja el eje x para el histograma: ",options=columnas_numericas)
        contenido['var_hist']=eje_x
        rango = st.slider(
            "Selecciona el filtro para esta columna del histograma: ",
            df_vis_hist[eje_x].min(),
            df_vis_hist[eje_x].max(),
            (df_vis_hist[eje_x].min(),df_vis_hist[eje_x].max())
            )
        contenido['rango_hist']=rango
        bins = st.slider(
            label="Escoja el numero de bins para el histograma: ",
            min_value=2, 
            max_value=20, 
            value=2
            )
        contenido['bins_hist']=bins
        try:
            df_vis_hist = df_vis_hist[(df_vis_hist[eje_x] >= rango[0]) & (df_vis_hist[eje_x] <= rango[1])]
            st.write("Previsualizacion del filtro")
            st.write(df_vis_hist.head(5))
            fig_hist = px.histogram(df_vis_hist, x=eje_x, nbins=bins,color_discrete_sequence=['blue'])
            fig_hist.write_image("hist.png")
            st.plotly_chart(fig_hist,use_container_width=True)

        except Exception as e:
            traceback.print_exc()
            st.write("El siguiente error ha ocurrido: " + str(e))
        
        st.subheader("Dispersion")
        eje_x2 = st.selectbox(label="Escoja el eje x para el diagrama de distribucion: ",options=columnas_numericas)
        contenido['eje_x_disp']=eje_x2
        eje_y2 = st.selectbox(label="Escoja el eje y para el diagrama de distribucion: ",options=columnas_text)
        contenido['eje_y_disp']=eje_y2
        try:
            fig_disp = px.scatter(df_cargado, x=eje_x2, y=eje_y2,color_discrete_sequence=['blue'])
            fig_disp.write_image("disp.png")
            st.plotly_chart(fig_disp,use_container_width=True)

        except Exception as e:
            traceback.print_exc()
            st.write("El siguiente error ha ocurrido: " + str(e))
            
        st.subheader("Pastel")
        dimension = st.selectbox(label="Escoja la dimension para el diagrama pastel: ",options=columnas_text)
        contenido['dim_pastel']= dimension
        try:
            conteo_categorias = df_cargado[dimension].value_counts()
            fig_pie= px.pie(
                names=conteo_categorias.index, 
                values=conteo_categorias.values,  
                title="Diagrama pastel",
                color_discrete_sequence=[
                    "#0068c9",
                    "#83c9ff",
                    "#ff2b2b",
                    "#ffabab",
                    "#29b09d",
                    "#7defa1",
                    "#ff8700",
                    "#ffd16a",
                    "#6d3fc0",
                    "#d5dae5"
                ]
            )
            fig_pie.write_image("pie.png")
            st.plotly_chart(fig_pie,use_container_width=True)

        except Exception as e:
            traceback.print_exc()
            st.write("El siguiente error ha ocurrido: " + str(e))
    
    ######################## REGRESION ######################
    with col3:
        st.title("3. Regresion")
        st.subheader("Seleccion de datos")
        subcol1, subcol2 = st.columns(2)
        with subcol1:
            variable_objetivo = st.selectbox(
                label="Seleccionar variable dependiente (target):",
                options=columnas_numericas
            )
            y_data = df_regresion[variable_objetivo]
            contenido['target'] = variable_objetivo
        with subcol2:
            variables_independientes = st.multiselect(
                label="Seleccionar variables independientes",
                options=columnas_numericas
            )
            x_data = df_regresion[variables_independientes]
            contenido['independientes'] = str(variables_independientes)
        
        if len(variables_independientes)>0:
            columnas_entrenamiento = variables_independientes
            columnas_entrenamiento = columnas_entrenamiento + [variable_objetivo]
            df_entrenamiento = df_cargado[columnas_entrenamiento].copy().head(5)
            st.write("Muestra de datos de entrenamiento: ")
            try:
                st.write(df_entrenamiento)
            except Exception as e:
                if "Duplicate column names found" in str(e):
                    st.write(":red[La columna objetivo forma parte de las variables independientes]")
                else:
                    st.write(f"Este error ha ocurrido: :red[{str(e)}]")
                    st.sidebar.write(f"Este error ha ocurrido: :red[{str(e)}]")
            guardar_dataframe_como_imagen(df_entrenamiento,'entrenamiento.png')
            porcentaje = st.slider(
                label="Escoja el porcentaje de datos para testing: ",
                min_value=0.1, 
                max_value=0.5, 
                value=0.2
                )
            
            contenido['porcentaje'] = porcentaje
            
            x_train, x_test, y_train, y_test=train_test_split(x_data,y_data,test_size=porcentaje)
            
            st.subheader("Modelo")
            lista_modelos = ["Lineal","Ridge","Lasso","KNNR","DTR"]
            select_modelo = st.selectbox(
                    label="Seleccionar modelo de regresion:",
                    options=lista_modelos
                )
            
            contenido['modelo'] = select_modelo
            
            if select_modelo == "Lineal":
                model = LinearRegression()
                
            elif select_modelo=="Ridge":
                alpha_r = st.slider(
                    label="Escoja el coeficiente de regularizacion (alpha):",
                    min_value=0.1,
                    max_value=10.0,
                    value=1.0,
                    step=0.1,
                )
                model = Ridge(alpha=alpha_r)
                
            elif select_modelo == "Lasso":
                alpha_l = st.slider(
                    label="Escoja el coeficiente de regularizacion (alpha):",
                    min_value=0.001,
                    max_value=1.000,
                    value=0.100,
                    step=0.001,
                )
                model = Lasso(alpha=0.100)
            
            elif select_modelo=="KNNR":
                neighbors = st.slider(
                    label="Escoja la cantidad de vecinos a usar:",
                    min_value=1,
                    max_value=10,
                    value=5,
                    step=1,
                )
                model = KNeighborsRegressor(n_neighbors=neighbors)
            
            elif select_modelo=="DTR":
                depth = st.slider(
                    label="Escoja la cantidad de vecinos a usar:",
                    min_value=1,
                    max_value=10,
                    value=3,
                    step=1,
                )
                model = DecisionTreeRegressor(max_depth=depth)
                
                
            try:
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                
                st.subheader("Resultados")
                
                accuracy = r2_score(y_test,y_pred)
                st.write("Accuracy: " + str(accuracy))
                contenido['accuracy'] = str(accuracy)
                
                mae = mean_absolute_error(y_test, y_pred)
                st.write("Mean Absolute Error: " + str(mae))
                contenido['mae'] = str(mae)
                
                mse = mean_squared_error(y_test, y_pred)
                st.write("Mean Squared Error: " + str(mse))
                contenido['mse'] = str(mse)
                
            except Exception as e:
                st.write("Ocurrio el siguiente error: "+str(e))

    if export_as_pdf:
        export_as_pdf
        placeholder = st.sidebar.empty()
        try:
            placeholder.write("Generando link de descarga, por favor espere....")
            pdf = generar_pdf(contenido)
            html = create_download_link(pdf.output(dest="S").encode("latin-1"), filename)
            st.sidebar.markdown(html, unsafe_allow_html=True)
            placeholder.empty()
        except Exception as e:
            placeholder.empty()
            traceback.print_exc()
            st.sidebar.write("No se pudo generar el link debido a un error.")
            st.sidebar.write(f"Este error ha ocurrido: :red[{str(e)}]")
else:
    st.title("Por favor cargue un archivo CSV desde el submenu desplegable")


