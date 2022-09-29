import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image
import base64
from random import randrange
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

st.set_page_config(
    page_title="STARTER TRAINING -UPC",
    page_icon ="img/upc_logo.png",
)

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_upc=get_base64_of_bin_file('img/upc_logo_50x50.png')
st.markdown(
    f"""
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {{
        width: 336px;        
    }}
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {{
        width: 336px;
        margin-left: -336px;
        background-color: #00F;
    }}
    [data-testid="stVerticalBlock"] {{
        flex: 0;
        gap: 0;
    }}
    #starter-training{{
        padding: 0;
    }}
    #div-upc{{
        #border: 1px solid #DDDDDD;
        background-image: url("data:image/png;base64,{img_upc}");
        position: fixed !important;
        right: 14px;
        bottom: 14px;
        width: 50px;
        height: 50px;
        background-size: 50px 50px;
    }}
    .css-10trblm{{
        color: #FFF;
        font-size: 40px;
        font-family: 'PROGRESS PERSONAL USE';
        src: url(fonts/ProgressPersonalUse-EaJdz.ttf);       
    }}
    #.main {{
        background: linear-gradient(135deg,#a8e73d,#09e7db,#092de7);
        background-size: 180% 180%;
        animation: gradient-animation 3s ease infinite;
        }}

        @keyframes gradient-animation {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}
    .block-container{{
        max-width: 100%;
    }}
    </style>
    """ ,
    unsafe_allow_html=True,
)

st.title('STARTER TRAINING')
st.sidebar.markdown('---')
st.markdown("<div id='div-upc'></span>", unsafe_allow_html=True)
st.sidebar.title('The Training App')
st.sidebar.markdown('---')

app_mode = st.sidebar.selectbox('Choose your training:',
    ['(home)','Glute Bridge','Abs', 'Lunges', 'Push Up', 'Squats']
)

if app_mode =='(home)':
    a=0
elif app_mode =='Glute Bridge':
    st.markdown('### __ABS__')
    st.markdown("<hr/>", unsafe_allow_html=True)

    webcam = st.checkbox('Start Webcam')
    st.markdown("Camera status: "+str(webcam))

    trainer, user = st.columns(2)

    with trainer:        
        st.markdown("TRAINER", unsafe_allow_html=True)
        #experto = randrange(3)+1
        experto = 4
        video_trainer_file="videos_trainer/Glute Bridges/Glute Bridges"+str(experto)+".mp4"
        
        df_experto = pd.read_csv("videos_trainer/Glute Bridges/Puntos_Glute_Bridges"+str(experto)+".csv")
        #df_experto = pd.read_csv("videos_trainer/Glute Bridges/Puntos_manu.csv")
        del df_experto['segundo']
        df_costos = pd.read_csv("videos_trainer/Glute Bridges/Costos_Glute Bridge_promedio__.csv")
        #df_costos = pd.read_csv("videos_trainer/Glute Bridges/costo_promedio_manu.csv")

        #st.table(df_experto)

        st.video(video_trainer_file, format="video/mp4", start_time=0)
        #st.table(df_costos)
        

    with user:
        st.markdown("YOU", unsafe_allow_html=True)

        #st.video(video_trainer_file, format="video/mp4", start_time=0)
        def dp(dist_mat):
            """
            Find minimum-cost path through matrix `dist_mat` using dynamic programming.

            The cost of a path is defined as the sum of the matrix entries on that
            path. See the following for details of the algorithm:

            - http://en.wikipedia.org/wiki/Dynamic_time_warping
            - https://www.ee.columbia.edu/~dpwe/resources/matlab/dtw/dp.m

            The notation in the first reference was followed, while Dan Ellis's code
            (second reference) was used to check for correctness. Returns a list of
            path indices and the cost matrix.
            """

            N, M = dist_mat.shape
            
            # Initialize the cost matrix
            cost_mat = np.zeros((N + 1, M + 1))
            for i in range(1, N + 1):
                cost_mat[i, 0] = np.inf
            for i in range(1, M + 1):
                cost_mat[0, i] = np.inf

            # Fill the cost matrix while keeping traceback information
            traceback_mat = np.zeros((N, M))
            for i in range(N):
                for j in range(M):
                    penalty = [
                        cost_mat[i, j],      # match (0)
                        cost_mat[i, j + 1],  # insertion (1)
                        cost_mat[i + 1, j]]  # deletion (2)
                    i_penalty = np.argmin(penalty)
                    cost_mat[i + 1, j + 1] = dist_mat[i, j] + penalty[i_penalty]
                    traceback_mat[i, j] = i_penalty

            # Traceback from bottom right
            i = N - 1
            j = M - 1
            path = [(i, j)]
            while i > 0 or j > 0:
                tb_type = traceback_mat[i, j]
                if tb_type == 0:
                    # Match
                    i = i - 1
                    j = j - 1
                elif tb_type == 1:
                    # Insertion
                    i = i - 1
                elif tb_type == 2:
                    # Deletion
                    j = j - 1
                path.append((i, j))

            # Strip infinity edges from cost_mat before returning
            cost_mat = cost_mat[1:, 1:]
            return (path[::-1], cost_mat)

        def calcular_costos(ej_usuario, df_planchas_experto1, inicio, fin, df_costos):
    
            #inicio=0
            #fin = 1
            
            resultados_costos = []
            resultados_index = []
            resultados_costo_al = []
            resultados_costo_al_normalizado = []

            #while (inicio <= len(df_planchas_experto1)-1 and fin <= len(df_planchas_experto1)):
                #print()
            ej_experto = []
            #st.markdown(df_experto["nose_x"][0], unsafe_allow_html=True)

            for i in df_planchas_experto1.columns:

            #for i in range(0,len(df_planchas_experto1.columns)):

                print(inicio)
                ej_experto.append(df_experto[i][inicio])


            x = np.array(ej_usuario) 

            y = np.array(ej_experto)

            N = x.shape[0]
            M = y.shape[0]
            dist_mat = np.zeros((N, M))
            for i in range(N):
                for j in range(M):
                    dist_mat[i, j] = abs(x[i] - y[j])

                # DTW
            path, cost_mat = dp(dist_mat)

            x_path, y_path = zip(*path)
            resultados_index.append(inicio)
            resultados_costo_al.append(cost_mat[N - 1, M - 1])
            resultados_costo_al_normalizado.append(cost_mat[N - 1, M - 1]/(N + M))
            resultados_costos.append([inicio,cost_mat[N - 1, M - 1],cost_mat[N - 1, M - 1]/(N + M)])
            inicio=inicio+1
            fin = fin+1
            #st.text(type(resultados_costos))

            return resultados_costos



        if webcam:
            stframe = st.empty()

            vid = cv2.VideoCapture(0,cv2.CAP_DSHOW)
            mp_pose = mp.solutions.pose
            start = time.time()
            #"the code you want to test stays here"
            #st.video(vid, format="video/mp4", start_time=0)
            c=0
            val = 0
            with mp_pose.Pose(static_image_mode=False) as pose:

                try:

                    while True:
                        ret, frame = vid.read()

                        if ret == False:
                            break

                        frame = cv2.flip(frame,1)
                        height, width, _ = frame.shape
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = pose.process(frame_rgb)
                        #print("Pose landmarks: ", results.pose_landmarks)
                        
                        if c % 15 == 0: #Procesa 15 frames por segundo
                        
                            resultados = []

                            for i in range(0, len(results.pose_landmarks.landmark)):
                                #time.sleep(1)
                                resultados.append(results.pose_landmarks.landmark[i].x)
                                resultados.append(results.pose_landmarks.landmark[i].y)
                                resultados.append(results.pose_landmarks.landmark[i].z)
                                resultados.append(results.pose_landmarks.landmark[i].visibility)

                            df_puntos = pd.DataFrame(np.reshape(resultados, (132, 1)).T)
                            df_puntos['segundo'] = str(c/15)
                            #print("Dataframe: ", df_puntos)
                            if c==0:
                                df_puntos_final = df_puntos.copy()
                            else:
                                #print(type(df_puntos_final))
                                #print(type(df_puntos))
                                df_puntos_final = pd.concat([df_puntos_final, df_puntos])
                            
                            ej_usuario = resultados
                            
                            if val == 0:
                                
                                inicio=0
                                fin = 1
                                st.text("Intentando el segundo cero")
                                resultados_costos = calcular_costos(ej_usuario,df_experto,inicio,fin, df_costos)
                                #st.text(type(resultados_costos))
                                #st.text(val)
                                if (resultados_costos[0][1]<= df_costos.Costo_alineamiento[inicio]-df_costos.Desviacion_estandar[inicio] or resultados_costos[0][1]<= df_costos.Costo_alineamiento[inicio]+df_costos.Desviacion_estandar[inicio]) and val==0: # promedio +- desviación estandar (para evitar casos rápidos o lentos)
                                
                                    val = 1

                                    costo_desde = "Costo desde: "+ str(round(df_costos.Costo_alineamiento[inicio]-df_costos.Desviacion_estandar[inicio],2))
                                    costo_hasta= "Costo hasta: "+ str(round(df_costos.Costo_alineamiento[inicio]+df_costos.Desviacion_estandar[inicio],2))
                                    costo_resultante = "Costo resultante: "+ str(round(resultados_costos[0][1],2))
                                    st.text(costo_desde)
                                    st.text(costo_hasta)
                                    st.text(costo_resultante)
                                    #st.text("Se realizó la pose del segundo cero")
                                    st.components.v1.html(f"""<span style="color:blue">Se realizó la pose del segundo cero</span>""")

                                    if results.pose_landmarks is not None:

                                        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                            mp_drawing.DrawingSpec(color=(128,0,250), thickness=2, circle_radius=3),
                                                            mp_drawing.DrawingSpec(color=(255,255,255), thickness=2))


                                    title = "Validacion - segundo "+str(inicio)
                                    result_validation = "Pose correcta"

                                    cv2.rectangle(frame, (200,200), (200,73), (245,117,16), -1)
                        
                                    # Rep data-testid
                                    cv2.putText(frame, #frame 
                                                title, #mensaje 
                                                (15,12), #posicion
                                                cv2.FONT_HERSHEY_SIMPLEX, #fuente 
                                                0.5, #opacidad 
                                                (255,255,255), #Color RGB
                                                1, # 
                                                cv2.LINE_AA) #

                                    cv2.putText(frame, #frame 
                                                costo_desde, #mensaje 
                                                (15,30), #posicion
                                                cv2.FONT_HERSHEY_SIMPLEX, #fuente 
                                                0.5, #opacidad 
                                                (0,0,0), #Color RGB
                                                1, # 
                                                cv2.LINE_AA) #


                                    cv2.putText(frame, #frame 
                                                costo_hasta, #mensaje 
                                                (15,45), #posicion
                                                cv2.FONT_HERSHEY_SIMPLEX, #fuente 
                                                0.5, #opacidad 
                                                (255,255,255), #Color RGB
                                                1, # 
                                                cv2.LINE_AA) #

                                    cv2.putText(frame, #frame 
                                                costo_resultante, #mensaje 
                                                (15,55), #posicion
                                                cv2.FONT_HERSHEY_SIMPLEX, #fuente 
                                                0.5, #opacidad 
                                                (255,255,255), #Color RGB
                                                1, # 
                                                cv2.LINE_AA) #

                                    cv2.putText(frame, #frame 
                                                result_validation, #mensaje 
                                                (10,70), #posicion
                                                cv2.FONT_HERSHEY_SIMPLEX, #fuente 
                                                0.5, #opacidad 
                                                (255,255,255), #Color RGB
                                                1, # 
                                                cv2.LINE_AA) #
                                    stframe.image(frame,channels = 'BGR',use_column_width=True)

                                    # CÓDIGO DE GRAN RENZO!!

                                    inicio = inicio+1
                                    fin = fin+1   
                                else:
                                    costo_desde = "Costo desde: "+ str(round(df_costos.Costo_alineamiento[inicio]-df_costos.Desviacion_estandar[inicio],2))
                                    costo_hasta= "Costo hasta: "+ str(round(df_costos.Costo_alineamiento[inicio]+df_costos.Desviacion_estandar[inicio],2))
                                    costo_resultante = "Costo resultante: "+ str(round(resultados_costos[0][1],2))
                                    if resultados_costos[0][1] <=30:
                                        st.text(costo_desde)
                                        st.text(costo_hasta)
                                        st.text(costo_resultante)
                                        st.components.v1.html(f"""<span style="color:red">No se realizó el ejercicio correctamente!!!!!</span>""")

                                    if results.pose_landmarks is not None:

                                        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                            mp_drawing.DrawingSpec(color=(128,0,250), thickness=2, circle_radius=3),
                                                            mp_drawing.DrawingSpec(color=(255,255,255), thickness=2))

                                    title = "Validacion - segundo "+str(inicio)
                                    result_validation = "Pose incorrecta"

                                    cv2.rectangle(frame, (0,0), (225,73), (245,117,16), -1)
                        
                                    # Rep data-testid
                                    cv2.putText(frame, #frame 
                                                title, #mensaje 
                                                (15,12), #posicion
                                                cv2.FONT_HERSHEY_SIMPLEX, #fuente 
                                                0.5, #opacidad 
                                                (0,0,0), #Color RGB
                                                1, # 
                                                cv2.LINE_AA) #

                                    cv2.putText(frame, #frame 
                                                costo_desde, #mensaje 
                                                (15,30), #posicion
                                                cv2.FONT_HERSHEY_SIMPLEX, #fuente 
                                                0.5, #opacidad 
                                                (255,255,255), #Color RGB
                                                1, # 
                                                cv2.LINE_AA) #

                                    cv2.putText(frame, #frame 
                                                costo_hasta, #mensaje 
                                                (15,45), #posicion
                                                cv2.FONT_HERSHEY_SIMPLEX, #fuente 
                                                0.5, #opacidad 
                                                (255,255,255), #Color RGB
                                                1, # 
                                                cv2.LINE_AA) #

                                    cv2.putText(frame, #frame 
                                                costo_resultante, #mensaje 
                                                (15,55), #posicion
                                                cv2.FONT_HERSHEY_SIMPLEX, #fuente 
                                                0.5, #opacidad 
                                                (255,255,255), #Color RGB
                                                1, # 
                                                cv2.LINE_AA) #

                                    cv2.putText(frame, #frame 
                                                result_validation, #mensaje 
                                                (10,70), #posicion
                                                cv2.FONT_HERSHEY_SIMPLEX, #fuente 
                                                0.5, #opacidad 
                                                (255,255,255), #Color RGB
                                                1, # 
                                                cv2.LINE_AA) #
                                    stframe.image(frame,channels = 'BGR',use_column_width=True)
 
                            else:
                                # SE VA A COMPARAR LOS COSTOS DE ALINEAMIENTO POR SEGUNDO EXACTO DEL USUARIO CON RESPECTO AL EXPERTO
                                
                                #inicio=1
                                #fin = 2
                                print("Intentando el segundo", str(inicio))
                                resultados_costos = calcular_costos(ej_usuario,df_experto,inicio,fin, df_costos)
                                
                                if (resultados_costos[0][1]>= df_costos.Costo_alineamiento[inicio]-df_costos.Desviacion_estandar[inicio] and resultados_costos[0][1]<= df_costos.Costo_alineamiento[inicio]+df_costos.Desviacion_estandar[inicio]):
                                    
                                    costo_desde = "Costo desde: "+ str(round(df_costos.Costo_alineamiento[inicio]-df_costos.Desviacion_estandar[inicio],2))
                                    costo_hasta= "Costo hasta: "+ str(round(df_costos.Costo_alineamiento[inicio]+df_costos.Desviacion_estandar[inicio],2))
                                    costo_resultante = "Costo resultante: "+ str(round(resultados_costos[0][1],2))
                                    st.text(costo_desde)
                                    st.text(costo_hasta)
                                    st.text(costo_resultante)
                                    st.components.v1.html(f"""<span style="color:blue">Se realizó la pose del segundo correspondiente</span>""")

                                    if results.pose_landmarks is not None:

                                        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                            mp_drawing.DrawingSpec(color=(128,0,250), thickness=2, circle_radius=3),
                                                            mp_drawing.DrawingSpec(color=(255,255,255), thickness=2))

                                    title = "Validacion - segundo "+str(inicio)
                                    result_validation = "Pose correcta"

                                    cv2.rectangle(frame, (0,0), (225,73), (245,117,16), -1)
                        
                                    # Rep data-testid
                                    cv2.putText(frame, #frame 
                                                title, #mensaje 
                                                (15,12), #posicion
                                                cv2.FONT_HERSHEY_SIMPLEX, #fuente 
                                                0.5, #opacidad 
                                                (255,255,255), #Color RGB
                                                1, # 
                                                cv2.LINE_AA) #

                                    cv2.putText(frame, #frame 
                                                costo_desde, #mensaje 
                                                (15,30), #posicion
                                                cv2.FONT_HERSHEY_SIMPLEX, #fuente 
                                                0.5, #opacidad 
                                                (0,0,0), #Color RGB
                                                1, # 
                                                cv2.LINE_AA) #

                                    cv2.putText(frame, #frame 
                                                costo_hasta, #mensaje 
                                                (15,45), #posicion
                                                cv2.FONT_HERSHEY_SIMPLEX, #fuente 
                                                0.5, #opacidad 
                                                (255,255,255), #Color RGB
                                                1, # 
                                                cv2.LINE_AA) #

                                    cv2.putText(frame, #frame 
                                                costo_resultante, #mensaje 
                                                (15,55), #posicion
                                                cv2.FONT_HERSHEY_SIMPLEX, #fuente 
                                                0.5, #opacidad 
                                                (255,255,255), #Color RGB
                                                1, # 
                                                cv2.LINE_AA) #

                                    cv2.putText(frame, #frame 
                                                result_validation, #mensaje 
                                                (10,70), #posicion
                                                cv2.FONT_HERSHEY_SIMPLEX, #fuente 
                                                0.5, #opacidad 
                                                (255,255,255), #Color RGB
                                                1, # 
                                                cv2.LINE_AA) #
                                    stframe.image(frame,channels = 'BGR',use_column_width=True)

                                    # CÓDIGO DE GRAN RENZO!!

                                    inicio = inicio+1
                                    fin = fin+1 

                                else:

                                    costo_desde = "Costo desde: "+ str(round(df_costos.Costo_alineamiento[inicio]-df_costos.Desviacion_estandar[inicio],2))
                                    costo_hasta= "Costo hasta: "+ str(round(df_costos.Costo_alineamiento[inicio]+df_costos.Desviacion_estandar[inicio],2))
                                    costo_resultante = "Costo resultante: "+ str(round(resultados_costos[0][1],2))
                                    st.text(costo_desde)
                                    st.text(costo_hasta)
                                    st.text(costo_resultante)
                                    st.components.v1.html(f"""<span style="color:red">No se realizó el ejercicio correctamente!!!!!</span>""")

                                    if results.pose_landmarks is not None:

                                        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                            mp_drawing.DrawingSpec(color=(128,0,250), thickness=2, circle_radius=3),
                                                            mp_drawing.DrawingSpec(color=(255,255,255), thickness=2))

                                    title = "Validacion - segundo "+str(inicio)
                                    result_validation = "Pose incorrecta"

                                    cv2.rectangle(frame, (0,0), (225,73), (245,117,16), -1)
                        
                                    # Rep data-testid
                                    cv2.putText(frame, #frame 
                                                title, #mensaje 
                                                (15,12), #posicion
                                                cv2.FONT_HERSHEY_SIMPLEX, #fuente 
                                                0.5, #opacidad 
                                                (0,0,0), #Color RGB
                                                1, # 
                                                cv2.LINE_AA) #

                                    cv2.putText(frame, #frame 
                                                costo_desde, #mensaje 
                                                (15,30), #posicion
                                                cv2.FONT_HERSHEY_SIMPLEX, #fuente 
                                                0.5, #opacidad 
                                                (255,255,255), #Color RGB
                                                1, # 
                                                cv2.LINE_AA) #

                                    cv2.putText(frame, #frame 
                                                costo_hasta, #mensaje 
                                                (15,45), #posicion
                                                cv2.FONT_HERSHEY_SIMPLEX, #fuente 
                                                0.5, #opacidad 
                                                (255,255,255), #Color RGB
                                                1, # 
                                                cv2.LINE_AA) #

                                    cv2.putText(frame, #frame 
                                                costo_resultante, #mensaje 
                                                (15,55), #posicion
                                                cv2.FONT_HERSHEY_SIMPLEX, #fuente 
                                                0.5, #opacidad 
                                                (255,255,255), #Color RGB
                                                1, # 
                                                cv2.LINE_AA) #

                                    cv2.putText(frame, #frame 
                                                result_validation, #mensaje 
                                                (10,70), #posicion
                                                cv2.FONT_HERSHEY_SIMPLEX, #fuente 
                                                0.5, #opacidad 
                                                (255,255,255), #Color RGB
                                                1, # 
                                                cv2.LINE_AA) #
                                    stframe.image(frame,channels = 'BGR',use_column_width=True)
                                
                                #inicio = inicio+1
                                #fin = fin+1  
                                
                                if inicio == len(df_experto):
                                    
                                    inicio=0
                                    fin = 1
                                    val=0
                            c = c+1
                        else:
                            #print("")

                            if results.pose_landmarks is not None:

                                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                    mp_drawing.DrawingSpec(color=(128,0,250), thickness=2, circle_radius=3),
                                                    mp_drawing.DrawingSpec(color=(255,255,255), thickness=2))
                            titulo_windows = "Prueba de ejercicios" 

                            if cv2.waitKey(1) & 0xFF == 27:
                                break
                            c = c+1

                        #key = cv2.waitKey(25)
                        #if key == ord('n') or key == ord('p') or key == ord('q'):
                        #    break

                except Exception as e:
                    st.text(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
                    st.text(e)
                    st.text("Porfavor active su webcam correctamente.")

            vid.release()
            cv2.destroyAllWindows()
            end = time.time()
            print("Segundos transacurridos: ",end - start)




        else:
            vid = cv2.VideoCapture(0)
            vid.release()
        

    st.markdown("<hr/>", unsafe_allow_html=True)


elif app_mode =='Abs':
    a=0
elif app_mode =='Lunges':
    a=0
elif app_mode =='Push Up':
    a=0
elif app_mode =='Squats':
    a=0
else:
    a=0

