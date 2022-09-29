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
        experto = randrange(3)+1

        video_trainer_file="videos_trainer/Glute Bridges/Glute Bridges"+str(experto)+".mp4"
        coord_video = pd.read_csv("videos_trainer/Glute Bridges/Puntos_Glute_Brigdes"+str(experto)+".csv")
        ruta_costos = pd.read_csv("videos_trainer/Glute Bridges/Costos_Glute Bridge_promedio.csv")

        st.table(coord_video)

        st.video(video_trainer_file, format="video/mp4", start_time=0)
        st.table(ruta_costos)
        

    with user:
        st.markdown("YOU", unsafe_allow_html=True)
        if webcam:
            stframe = st.empty()
            vid = cv2.VideoCapture(0)

            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_input = int(vid.get(cv2.CAP_PROP_FPS))
            
            codec = cv2.VideoWriter_fourcc('V','P','0','9')
            fps = 0
            i = 0
            drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

            max_faces=1
            detection_confidence=90
            tracking_confidence=90
            with mp_face_mesh.FaceMesh(
                min_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence, 
                max_num_faces = max_faces) as face_mesh:
                prevTime = 0

                while vid.isOpened():
                    i +=1
                    ret, frame = vid.read()
                    if not ret:
                        continue

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(frame)

                    frame.flags.writeable = True
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    face_count = 0
                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            face_count += 1
                            mp_drawing.draw_landmarks(
                            image = frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACE_CONNECTIONS,
                            landmark_drawing_spec=drawing_spec,
                            connection_drawing_spec=drawing_spec)
                    currTime = time.time()
                    fps = 1 / (currTime - prevTime)
                    prevTime = currTime

                    frame = cv2.resize(frame,(0,0),fx = 0.8 , fy = 0.8)
                    stframe.image(frame,channels = 'BGR',use_column_width=True)
            stframe = st.empty()
            vid.release()

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

