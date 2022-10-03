#!/usr/bin/python

import cv2
import mediapipe as mp
import numpy as np
import time
import streamlit as st
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle 

def start(sets, reps):
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    sets_counter = 0
    stframe = st.empty()
    while sets_counter < sets:
        # Crunches reps_counter variables
        reps_counter = 0 
        stage = None

        # Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            cap.isOpened()
            while reps_counter < reps:
                ret, frame = cap.read()
                
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
            
                # Make detection
                results = pose.process(image)
            
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Get coordinates
                    knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    
                    # Setup status box
                    cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
                    
                    # Rep data
                    cv2.putText(image, 'REPS', (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(reps_counter), 
                                (10,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                    
                    # Stage data
                    cv2.putText(image, 'STAGE', (65,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, stage, 
                                (60,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                    
                    
                    # Render detections
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                            )    
                    # Calculate angle
                    angle = calculate_angle(knee, hip, shoulder)
                    
                    # Visualize angle
                    cv2.putText(image, str(angle), 
                                tuple(np.multiply(hip, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                    
                    # Crunches Increment Logic
                    if angle > 100:
                        stage = "up"
                    if angle < 50 and stage =='up':
                        stage="down"
                        reps_counter +=1
                
                    #cv2.imshow('Mediapipe Feed', image)
                    stframe.image(image,channels = 'BGR',use_column_width=True)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

                except:
                    pass    
            sets_counter += 1
            if (sets_counter!=sets):
                try:
                    cv2.putText(image, 'FINISHED SET', (100,250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, cv2.LINE_AA)
                    #cv2.imshow('Mediapipe Feed', image)
                    stframe.image(image,channels = 'BGR',use_column_width=True)
                    cv2.waitKey(1)
                    time.sleep(20)   

                except:
                    #cv2.imshow('Mediapipe Feed', image)
                    stframe.image(image,channels = 'BGR',use_column_width=True)
                    pass 
                            
    cv2.rectangle(image, (50,180), (600,400), (0,255,0), -1)
    cv2.putText(image, 'FINISHED EXERCISE', (100,250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)
    cv2.putText(image, 'REST FOR 30s' , (155,350), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)   
    #cv2.imshow('Mediapipe Feed', image)
    stframe.image(image,channels = 'BGR',use_column_width=True)
    cv2.waitKey(1) 
    time.sleep(20)                      
    cap.release()
    cv2.destroyAllWindows()         