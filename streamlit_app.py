import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import json
import warnings
import time 
from pathlib import Path
from datetime import datetime

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from io import BytesIO

st.set_page_config(
    page_title='CALCULADORA RCV INTEGRAL',
    page_icon='🩸', # This is an emoji shortcode. Could be a URL too.
    layout="wide",
)

# footer="""<style>
# a:link , a:visited{
# color: blue;
# background-color: transparent;
# text-decoration: underline;
# }

# a:hover,  a:active {
# color: red;
# background-color: transparent;
# text-decoration: underline;
# }

# .footer {
# position: fixed;
# left: 0;
# bottom: 0;
# width: 100%;
# background-color: white;
# color: black;
# text-align: center;
# }
# </style>
# <div class="footer">
# <p>Developed with ❤ by <a style='display: block; text-align: center;' href="https://www.heflin.dev/" target="_blank">Heflin Stephen Raj S</a></p>
# </div>
# """
# st.markdown(footer,unsafe_allow_html=True)

# LOGO
cibio_logo = "images/cibio_editado.png"
st.logo(
    cibio_logo,
    size="large",
    icon_image=cibio_logo,
)

st.title = "CALCULADORA RCV INTEGRAL"
'''
# Calculador del Riesgo Cardiovascular
'''

if "intento_calculo" not in st.session_state:
    st.session_state.intento_calculo = False

def mostrar_error(key):
    if st.session_state.get(f"error_{key}"):
        st.write(f":red[{st.session_state[f'error_{key}']}]")

with st.form("form"):

    col_input1, col_input2, col_input3 = st.columns(3, gap='xlarge')

    with col_input1:
        sexo = st.segmented_control(
            "Sexo",
            options={1:"Masculino",0:"Femenino"},
            format_func=lambda x: {1:"Masculino",0:"Femenino"}[x],
            key="sexo",
        )
        mostrar_error("sexo")

        cintura = st.number_input(
            "Perímetro de cintura (cm)",
            min_value=0.0,
            max_value=200.0,
            step=0.1,
            key="cintura"
        )
        mostrar_error("cintura")

        pas = st.number_input(
            "Presión Arterial Sistólica (mmHg)",
            min_value=0.0,
            max_value=300.0,
            step=0.1,
            key="pas"
        )
        mostrar_error("pas")

    with col_input2:
        edad = st.number_input("Edad", min_value=0, max_value=120, step=1, key="edad")
        mostrar_error("edad")

        vop = st.number_input("VOP", min_value=0.0, max_value=30.0, step=0.1, key="vop")
        mostrar_error("vop")

        pad = st.number_input(
            "Presión Arterial Diastólica (mmHg)",
            min_value=0.0,
            max_value=200.0,
            step=0.1,
            key="pad"
        )
        mostrar_error("pad")

    with col_input3:
        tabaco = st.segmented_control(
            "¿Fuma tabaco?",
            options={0:"No",1:"Sí"},
            format_func=lambda x: {0:"No",1:"Sí"}[x],
            key="tabaco"
        )
        mostrar_error("tabaco")

        fc = st.number_input(
            "Frecuencia Cardíaca (lpm)",
            min_value=0.0,
            max_value=200.0,
            step=0.1,
            key="fc"
        )
        mostrar_error("fc")

    nombres = {
        "sexo": "Sexo",
        "cintura": "Perímetro de cintura",
        "pas": "Presión Arterial Sistólica",
        "edad": "Edad",
        "vop": "VOP",
        "pad": "Presión Arterial Diastólica",
        "tabaco": "Si fuma tabaco",
        "fc": "Frecuencia Cardíaca"
    }

    def validar_input_bool(key, nombre):
        valor = st.session_state[key]

        if valor is None:
            st.session_state[f"error_{key}"] = f"{nombre} no debe estar vacío"
        else:
            st.session_state[f"error_{key}"] = ""

    def validar_input_num(key, nombre):
        valor = st.session_state.get(key)

        if valor is None or valor <= 0:
            st.session_state[f"error_{key}"] = f"{nombre} debe ser mayor a 0"
        else:
            st.session_state[f"error_{key}"] = ""

    def validar_todos_inputs():
        for k in nombres.keys():
            if k in ["sexo","tabaco"]:
                validar_input_bool(k, nombres[k])
            else:
                validar_input_num(k, nombres[k])
    

    errores = [st.session_state.get(f"error_{k}") for k in nombres.keys()]

    def clic_calcular():
        validar_todos_inputs()
        errores = [st.session_state.get(f"error_{k}") for k in nombres.keys()]

        if not any(errores):
            st.session_state.intento_calculo = True

    calcular = st.form_submit_button("Calcular", on_click=clic_calcular)

    def barra_carga():
        carga_text = "Operación en curso. Por favor, espere."
        my_bar = st.progress(0, text=carga_text)

        for percent_complete in range(100):
            time.sleep(0.007)
            my_bar.progress(percent_complete + 1, text=carga_text)
        time.sleep(0.7)
        my_bar.empty()

    if st.session_state.intento_calculo and not any(errores):
        barra_carga()
        st.success("Cálculo realizado con éxito.")

    if any(errores):
        st.error("Por favor, corrija los errores antes de calcular.")
        st.stop()




if st.session_state.get("intento_calculo"):
    # RIESGO CARDIOVASCULAR
    #Cálculo del Riesgo Cardiovascular SCORE2
    def RCV_Score2(VOP, EDAD, FC, SEXO, PAS, PAD, TABACO):
        """
        Determinación del RCV NO LAB (SCORE-2 ESC 2021, European Heart Journal.)
        Parámetros: EDAD (40–89), PAS (mmHg), TABACO (1 = SI, 0 = NO), SEXO (1 para hombre, 0 para mujer)
        Retorna:
            riesgo estimado en %
        """
        #Transformaciones estándar del SCORE2
        cage = (EDAD - 60) / 5
        csbp = (PAS - 120) / 20
        #Descriminación por SEXO ()
        if SEXO == 1: #SEXO=1 MASCULINO
            beta_age = 0.3742
            beta_sbp = 0.2777
            beta_smoke = 0.6012
            beta_age_smoke = -0.0755
            beta_age_sbp = -0.0255
            #Supervivencia basal a 10 años (hombres)
            S0 = 0.9605
        else: #SEXO=0 FEMENINO
            beta_age = 0.4648
            beta_sbp = 0.3131
            beta_smoke = 0.7744
            beta_age_smoke = -0.1088
            beta_age_sbp = -0.0277
            #Supervivencia basal a 10 años (mujeres)
            S0 = 0.9776
            #Linear predictor (LP)
        LP = (
            beta_age * cage +
            beta_sbp * csbp +
            beta_smoke * TABACO +
            beta_age_smoke * (cage * TABACO) +
            beta_age_sbp * (cage * csbp)
            )
            #Conversión a RCV
        risk = (1 - (S0 ** math.exp(LP)))*100
            #Clasifica el riesgo SCORE2 según las categorías recomendadas (EU 2021):
        if risk < 2.5:
            clasif= "Bajo"
        elif risk < 7.5:
            clasif= "Moderado"
        elif risk < 10:
            clasif= "Alto"
        else:
            clasif= "Muy alto"
        return max(0, min(risk, 30)), clasif  # limitar entre 0% y 30% (rango normal SCORE2)

        #Cálculo del Riesgo Cardiovascular SCORE2 (OPCIONAL A EVALUAR)
    def score2_base_risk_OP(age, sex, smoker, sbp):
        """
        Cálculo aproximado del SCORE2 (riesgo a 10 años de eventos cardiovasculares fatales y no fatales).
        La fórmula es una simplificación linealizada basada en el patrón de las tablas europeas.

        Parámetros:
            age : edad en años (40–69)
            sex : "M" o "F"
            smoker : True / False
            sbp : presión arterial sistólica (mmHg)

        Retorna:
            riesgo estimado en %
        """
        # Coeficientes aproximados que reproducen el comportamiento de SCORE2
        base = -8.5                           # intercepto
        beta_age = 0.085                      # sensibilidad a la edad
        beta_sbp = 0.015                      # sensibilidad a la PAS
        beta_smoke = 0.8 if smoker else 0     # penalización por tabaquismo
        beta_sex = 0.5 if sex == "M" else 0   # mayor riesgo masculino

        logit = base + beta_age*age + beta_sbp*sbp + beta_smoke + beta_sex
        risk = 100 * (1 / (1 + np.exp(-logit))) # convertir logit a probabilidad %
        return max(0, min(risk, 30))            #Limitar entre 0% y 30% (rango normal SCORE2)

    #RECLASIFICACIÓN Del Riesgo Cardiovascular POR PWV
    def RCV_reclasif_VOP(categoria, pwv):
        """
        Reclasificación basada en PWV según guías europeas:
        - PWV ≥ 10 m/s → aumenta 1 categoría de riesgo
        """
        categorias = ["Bajo", "Moderado", "Alto", "Muy alto"]
        idx = categorias.index(categoria)
        if pwv >= 10:
            idx = min(idx + 1, len(categorias) - 1)
        return categorias[idx]

        #RECLASIFICACIÓN Del Riesgo Cardiovascular POR PERÍMETRO DE CINTURA
    def RCV_reclasif_PC(categoria, sexo, cintura_cm):
        """
        Reclasifica el riesgo cardiovascular SCORE2 incorporando
        obesidad abdominal (perímetro de cintura).
        """
        # Umbral de cintura por sexo
        if sexo == 1:
            cintura_alta = cintura_cm > 102
        elif sexo == 0:
            cintura_alta = cintura_cm > 88
        else:
            raise ValueError("sexo debe ser 'hombre' o 'mujer'")

        # Escala ordinal
        escala = ["Bajo", "Moderado", "Alto", "Muy Alto"]
        idx = escala.index(categoria)

        if cintura_alta and idx < len(escala) - 1:
            riesgo_final = escala[idx + 1]
        else:
            riesgo_final = categoria

        return riesgo_final


    # ANÁLISIS INTEGRAL
    def Ajuste_VOP (VOP, EDAD, FC, SEXO, PAS, PAD):
        #Determinación de la eVOP (estimated PWV, complementaria, Greve SV, et al. “Estimated Pulse Wave Velocity Calculated from Age and Mean Arterial Blood Pressure")
        PAM=PAD+0.4*(PAS-PAD)
        eVOP=9.587-0.402*EDAD+4.560e-3*EDAD**2-2.621e-5*EDAD**2*PAM+3.176e-3*EDAD*PAM-1.832e-2*PAM

        #Ajuste de la VOP por Presión Arterial Sistólica y Frecuencia Cardíaca (Importante para la EDAD VASCULAR)
        #Se utiliza la fórmula de ajuste poblacional VOP_norm=VOP_med-beta_PAS*(PAS-PAS_ref)-beta_FC*(FC-FC_ref) - beta_SEXO*(Sexo-Sexo_ref) (McEniery et al. Normal Vascular Ageing 2005)
        #Se ajusta por SEXO dado que las curvas son poblacionales (no separadas)
        VOP_norm = VOP - 0.03*(PAS-120) - 0.01*(FC-75) - 0.25*(SEXO-0)

        # Armado del dataframe
        datos_vop = {
            #   "DATOS DEMOGRÁFICOS": [""],
            #   "Edad Cronológica (años)": [f"{EDAD:.1f}"],
            #   "Sexo (0=Femeninio, 1=Masculino)": [f"{SEXO}"],
            #   "Presión Arterial Braquial Sistólica (mmHg)": [f"{PAS:.1f}"],
            #   "Presión Arterial Braquial Diastólica (mmHg)": [f"{PAD:.1f}"],
            #   "Frecuencia Cardíaca (lpm)": [f"{FC:.1f}"],
            #   "Consumo de Tabaco (1=SI, 0=NO)": [f"{TABACO}"],
            #   "Perímetro de Cintura (cm)": [f"{PC}"],
            #   "-": [""],
            # "Velocidad de la Onda del Pulso CF medida (m/s)": [f"{VOP:.1f}"],
            "Velocidad de la Onda del Pulso CF ajustada 120/@75 (m/s)": [f"{VOP_norm:.1f}"],
            "Velocidad de la Onda del Pulso CF estimada (clínica) (m/s)": [f"{eVOP:.1f}"],
        }
        df_Eval_vop = pd.DataFrame.from_dict(datos_vop, orient='index', columns=['Valor'])

        return df_Eval_vop, VOP_norm

    def Analizar_RCV (VOP, EDAD, FC, SEXO, PAS, PAD, TABACO, PC, VOP_norm):
        #Cálculo RCV
        RCV_10a, categoria = RCV_Score2(VOP, EDAD, FC, SEXO, PAS, PAD, TABACO)
        #Ajuste del RCV por VOP
        categoria_VOP=RCV_reclasif_VOP(categoria, VOP_norm)
        #Ajuste del RCV por PERÍMETRO DE CINTURA
        categoria_PC=RCV_reclasif_PC(categoria, SEXO, PC)

    #Armado del dataframe
        datos_riesgo = {
            "Riesgo Cardiovascular (No Lab) (%)": [f"{RCV_10a:.1f} ({categoria})"],
            "Riesgo Cardiovascular reclasificado por VOP ajustada": [categoria_VOP],
            "Riesgo Cardiovascular reclasificado por PERÍMETRO DE CINTURA": [categoria_PC],
        }
        # Crear DataFrame vertical: índice = etiquetas, columna única de valores
        df_Eval_riesgo = pd.DataFrame.from_dict(datos_riesgo, orient='index', columns=['valor'])
        #    df_Eval_Individuo = df_Eval_Individuo.style.set_table_styles(
        #     [{'selector': 'th.row_heading',
        #       'props': [('text-align', 'left')]}]
        #    ).set_properties(
        #     subset=pd.IndexSlice[:, :],
        #     **{'text-align': 'left'}
        #    )

        return df_Eval_riesgo

    #Análisis
    Eval_VOP, VOP_norm = Ajuste_VOP(vop, edad, fc, sexo, pas, pad)
    Eval_riesgo = Analizar_RCV(vop, edad, fc, sexo, pas, pad, tabaco, cintura, VOP_norm)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.colheader_justify', 'left')

    #Visualización de los resultados obtenidos
    df_resultados_riesgo = Eval_riesgo.reset_index()
    df_resultados_vop = Eval_VOP.reset_index()
    df_resultados_riesgo.columns = ["Campo", "Valor"]
    df_resultados_vop.columns = ["Campo", "Valor"]


    col_1, col_2, col_3 = st.columns([0.5, 1, 0.5])
    with col_2:
        '''
        ### Riesgo Cardiovascular
        '''
        st.dataframe(
            df_resultados_riesgo,
            width='stretch',
            hide_index=True,
            column_config={
                "Campo": st.column_config.TextColumn(
                    "Campo",
                    width="large"
                ),
                "Valor": st.column_config.TextColumn(
                    "Valor",
                    width="small"
                )
            }
        )

        '''
        ### Velocidad de la Onda del Pulso Arterial (Indicador de rigidez aórtica)
        '''
        st.dataframe(
            df_resultados_vop,
            width='stretch',
            hide_index=True,
            column_config={
                "Campo": st.column_config.TextColumn(
                    "Campo",
                    width="large"
                ),
                "Valor": st.column_config.TextColumn(
                    "Valor",
                    width="small"
                )
            }
        )

        '''
        ---
        ### Análisis de su edad vascular en base a su rigidez arterial
        '''
        # Extraer datos de JSON
        with open('data/vop_data.json') as f:
            data = json.load(f)

        # Configuración según sexo
        titulo = f"Curvas VOP - {'Masculino' if sexo == 1 else 'Femenino'}"
        linea = 'dashed' if sexo == 1 else 'solid'
        letra = 'm' if sexo == 1 else 'f'
        curva_default = f"{letra}50"
        curvas_config = [
            (f"{letra}95", "orchid", "95"),
            (f"{letra}75", "deepskyblue", "75"),
            (f"{letra}25", "peru", "25"),
            (f"{letra}5",  "lightcoral", "5"),
        ]

    # Crear gráfico
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    fig, ax = plt.subplots(figsize=(6, 2))
    curvas = []

    def generar_curva(ax, datos_x, datos_y, color, nombre, linea):
        curva, = ax.plot(datos_x, datos_y, color=color, linestyle=linea, label=nombre)
        return curva

    # Dibujar curva por defecto
    curva_def = generar_curva(ax1, data[curva_default]["x"], data[curva_default]["y"], "limegreen", "50", linea)
    ax1.set_xlabel("Edad Vascular (años)")
    ax1.set_ylabel("VOP")
    ax1.set_title(titulo + " - P50")
    ax1.grid(True)

    # Dibujar curvas restantes
    for key, color, label in curvas_config:
        curva = generar_curva(ax, data[key]["x"], data[key]["y"], color, label, linea)
        curvas.append(curva)

    ax.set_xlabel("Edad vascular (años)")
    ax.set_ylabel("VOP")
    ax.set_title(titulo)
    ax.xaxis.set_major_locator(plt.MultipleLocator(10))
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax.grid(True)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.subplots_adjust(right=0.78)

    # Interpolación
    def edad_por_vop(curva, vop):
        return round(np.interp(vop, curva.get_ydata(), curva.get_xdata()))

    # Función para armar el Dataframe de las curvas
    def armar_dataframe(curvas, vop):
        rows = []
        for curva in curvas:
            edad_vascular = edad_por_vop(curva, vop)
            rows.append({
                "Percentil": curva.get_label(),
                "Edad vascular": round(edad_vascular, 2)
            })
        return pd.DataFrame(rows)

    # Función marcar edad vascular en el gráfico
    def marcar_edad_vascular(ax, edad, vop):
        ax.plot(edad, vop, marker="x", color="red")

    # Calcular edad vascular y marcar en gráfico
    edad_vascular_def = edad_por_vop(curva_def, VOP_norm)
    marcar_edad_vascular(ax1, edad_vascular_def, VOP_norm)
    df_resultado = armar_dataframe([curva_def], VOP_norm)

    for curva in curvas:
        edad_vascular = edad_por_vop(curva, VOP_norm)
        marcar_edad_vascular(ax, edad_vascular, VOP_norm)
    df_resultados = armar_dataframe(curvas, VOP_norm)

    # Mostrar gráfico y tabla
    # Gráfico de curvas P50 - Mediana
    with col_2:
        st.pyplot(fig1)
        # st.dataframe(df_resultado, width=400, hide_index=True)
        st.write("Edad Real:", edad, " vs  Edad Vascular: ", edad_vascular_def)

        # # Gráfico de curvas extras
        # '''
        # ### Edad Vascular Según VOP - Percentiles
        # '''
        # # st.pyplot(fig)
        # st.dataframe(df_resultados, width=400, hide_index=True)



    def percentil_mas_cercano(data, edad_real, vop_norm, sexo):

        letra = 'm' if sexo == 1 else 'f'
        percentiles = [5, 25, 50, 75, 95]

        # Para cada percentil, buscar qué tanto difiere entre las VOPs
        diferencias = {}
        for p in percentiles:
            key = f"{letra}{p}"

            edades = np.array(data[key]["x"])
            vops = np.array(data[key]["y"])

            # Hallar VOP esperada en esa edad
            vop_esperada = np.interp(edad_real, edades, vops)

            diferencias[p] = abs(vop_norm - vop_esperada)

        # Devolver la diferencia con el menor valor
        return min(diferencias, key=diferencias.get)

    p_cercano = percentil_mas_cercano(data, edad, VOP_norm, sexo)

    with col_2:
        st.write(f"Percentil más cercano para su edad: P{p_cercano}")
        st.write(f"El {p_cercano}% de las personas de su misma edad tienen una VOP menor")


        # Descarga PDF
        def generar_pdf(df_riesgo, df_vop, fig, datos_usuario):
            time.sleep(1.5)
            buffer = BytesIO()
            pdf = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            elementos = []
            now = datetime.now()
            elementos.append(Paragraph(f"Informe generado el {now.strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))

            elementos.append(Image("images/cibio.png", width=120, height=120))
            elementos.append(Spacer(1,20))

            elementos.append(Paragraph("Informe del Riesgo Cardiovascular", styles["Title"]))
            elementos.append(Spacer(1,20))

            # DATOS DEL PACIENTE
            elementos.append(Paragraph("Datos del paciente", styles["Heading2"]))
            data_user = [["Campo","Valor"]]
            for k,v in datos_usuario.items():
                data_user.append([k,v])

            tabla_user = Table(data_user)
            tabla_user.setStyle([
                ("BACKGROUND",(0,0),(-1,0),colors.grey),
                ("GRID",(0,0),(-1,-1),1,colors.black)
            ])
            elementos.append(tabla_user)
            elementos.append(Spacer(1,20))

            # RESULTADOS RIESGO
            elementos.append(Paragraph("Resultados de Riesgo Cardiovascular", styles["Heading2"]))
            data_riesgo = [df_riesgo.columns.tolist()] + df_riesgo.values.tolist()
            tabla_riesgo = Table(data_riesgo)
            tabla_riesgo.setStyle([
                ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
                ("GRID",(0,0),(-1,-1),1,colors.black)
            ])
            elementos.append(tabla_riesgo)
            elementos.append(Spacer(1,20))

            # RESULTADOS VOP
            elementos.append(Paragraph("Resultados de VOP", styles["Heading2"]))
            data_vop = [df_vop.columns.tolist()] + df_vop.values.tolist()
            tabla_vop = Table(data_vop)
            tabla_vop.setStyle([
                ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
                ("GRID",(0,0),(-1,-1),1,colors.black)
            ])
            elementos.append(tabla_vop)
            elementos.append(Spacer(1,20))

            # GRÁFICO
            fig.savefig("grafico_temp.png")
            elementos.append(Paragraph("Curvas VOP", styles["Heading2"]))
            elementos.append(Image("grafico_temp.png", width=400, height=250))
        
            # RECOMENDACIONES
            elementos.append(Paragraph("Recomendaciones a seguir", styles["Heading2"]))

            pdf.build(elementos)
            buffer.seek(0)
            return buffer

        # def descarga_msj():
        #     msg = st.toast("Recopilando los archivos...")
        #     time.sleep(1)
        #     msg.toast("Preparando...")
        #     time.sleep(1)
        #     msg.toast("¡Su archivo pdf está listo para su descarga!", icon="👌")

        datos_usuario = {
            "Edad": edad,
            "Sexo": "Masculino" if sexo == 1 else "Femenino",
            "Presión Sistólica": pas,
            "Presión Diastólica": pad,
            "Frecuencia Cardíaca": fc,
            "Tabaco": "Sí" if tabaco == 1 else "No",
            "Perímetro de cintura": cintura,
            "VOP": vop
        }


        if st.button("Preparar informe PDF"):

            with st.spinner("Generando informe..."):
                pdf = generar_pdf(df_resultados_riesgo, df_resultados_vop, fig1, datos_usuario)
                st.download_button(
                    label="Descargar",
                    data=pdf,
                    file_name="informe_rcv.pdf",
                    mime="application/pdf",
                    type="primary",
                    icon=":material/download:",
                )



