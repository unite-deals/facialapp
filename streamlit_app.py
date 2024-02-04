import uuid
import shutil
import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from settings1 import *
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

hide_github_link_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visiblity: hidden;}
    header {visibility: hidden;}
        .viewerBadge_container__1QSob {
            display: none !important;
        }
    </style>
"""

st.markdown(hide_github_link_style, unsafe_allow_html=True)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showfileUploaderEncoding', False)

user_color = '#000000'
title_webapp = "Visitor Monitoring Analysis with facial recognition"
image_link = "https://miro.medium.com/v2/resize:fit:828/format:webp/1*9OJXRX98cNHWJLO3XMSIaQ.png"

html_temp = f"""
            <div style="background-color:{user_color};padding:12px">
            <h1 style="color:white;text-align:center;">{title_webapp}
            <img src = "{image_link}" align="right" width=150px ></h1>
            </div>
            """
st.markdown(html_temp, unsafe_allow_html=True)

if st.sidebar.button('Click to Clear out all the data'):
    shutil.rmtree(VISITOR_DB, ignore_errors=True)
    os.mkdir(VISITOR_DB)
    shutil.rmtree(VISITOR_HISTORY, ignore_errors=True)
    os.mkdir(VISITOR_HISTORY)

if not os.path.exists(VISITOR_DB):
    os.mkdir(VISITOR_DB)

if not os.path.exists(VISITOR_HISTORY):
    os.mkdir(VISITOR_HISTORY)

mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection


def detect_faces(image_array):
    face_detection = mp_face_detection.FaceDetection(0.5)
    results = face_detection.process(image_array)
    faces = []
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image_array.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)
            faces.append((x, y, x + w, y + h))
    return faces


def main():
    st.sidebar.header("About")
    st.sidebar.info("This webapp gives a demo of Visitor Monitoring "
                    "Webapp using 'MediaPipe' and AI-based monitoring for multiple use cases like attendance, etc")

    selected_menu = option_menu(None,
                                ['Visitor Validation', 'View Visitor History', 'Add to Database'],
                                icons=['camera', "clock-history", 'person-plus'],
                                menu_icon="cast", default_index=0, orientation="horizontal")

    if selected_menu == 'Visitor Validation':
        visitor_id = uuid.uuid1()
        img_file_buffer = st.camera_input("Take a picture")

        if img_file_buffer is not None:
            bytes_data = img_file_buffer.getvalue()
            image_array = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            image_array_copy = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            with open(os.path.join(VISITOR_HISTORY, f'{visitor_id}.jpg'), 'wb') as file:
                file.write(img_file_buffer.getbuffer())
                st.success('Image Saved Successfully!')

                face_locations = detect_faces(image_array)

                max_faces = len(face_locations)

                if max_faces > 0:
                    col1, col2 = st.columns(2)
                    face_idxs = col1.multiselect("Select face#", range(max_faces), default=range(max_faces))
                    similarity_threshold = col2.slider('Select Threshold for Similarity', min_value=0.0,
                                                       max_value=1.0, value=0.5)

                    flag_show = False

                    if col1.checkbox('Click to proceed!') and len(face_idxs) > 0:
                        dataframe_new = pd.DataFrame()

                        for face_idx in face_idxs:
                            (left, top, right, bottom) = face_locations[face_idx]
                            rois.append(image_array[top:bottom, left:right].copy())

                            database_data = initialize_data()
                            face_encodings = database_data[COLS_ENCODE].values
                            dataframe = database_data[COLS_INFO]

                            faces = detect_faces(image_array)

                            if len(faces) < 1:
                                st.error(f'Please Try Again for face#{face_idx}!')
                            else:
                                face_to_compare = faces[0]
                                dataframe['distance'] = face_recognition.face_distance(face_encodings, face_to_compare)
                                dataframe['distance'] = dataframe['distance'].astype(float)

                                dataframe['similarity'] = dataframe.distance.apply(
                                    lambda distance: f"{face_distance_to_conf(distance):0.2}")
                                dataframe['similarity'] = dataframe['similarity'].astype(float)

                                dataframe_new = dataframe.drop_duplicates(keep='first')
                                dataframe_new.reset_index(drop=True, inplace=True)
                                dataframe_new.sort_values(by="similarity", ascending=True)

                                dataframe_new = dataframe_new[
                                    dataframe_new['similarity'] > similarity_threshold].head(1)
                                dataframe_new.reset_index(drop=True, inplace=True)

                                if dataframe_new.shape[0] > 0:
                                    (left, top, right, bottom) = face_locations[face_idx]
                                    rois.append(image_array_copy[top:bottom, left:right].copy())

                                    cv2.rectangle(image_array_copy, (left, top), (right, bottom), COLOR_DARK, 2)
                                    cv2.rectangle(image_array_copy, (left, bottom + 35), (right, bottom), COLOR_DARK,
                                                  cv2.FILLED)
                                    font = cv2.FONT_HERSHEY_DUPLEX
                                    cv2.putText(image_array_copy, f"#{dataframe_new.loc[0, 'Name']}",
                                                (left + 5, bottom + 25), font, .55, COLOR_WHITE, 1)

                                    name_visitor = dataframe_new.loc[0, 'Name']
                                    attendance(visitor_id, name_visitor)

                                    location = get_location()
                                    st.success(f'Attendance marked for {name_visitor} at {location}')
                                    flag_show = True

                                else:
                                    st.error(f'No Match Found for the given Similarity Threshold! for face#{face_idx}')
                                    st.info('Please Update the database for a new person or click again!')
                                    attendance(visitor_id, 'Unknown')

                        if flag_show:
                            st.image(BGR_to_RGB(image_array_copy), width=720)

                    else:
                        st.error('No human face detected.')

    if selected_menu == 'View Visitor History':
        view_attendace()

    if selected_menu == 'Add to Database':
        col1, col2, col3 = st.columns(3)
        face_name = col1.text_input('Name:', '')
        pic_option = col2.radio('Upload Picture', options=["Upload a Picture", "Click a picture"])

        if pic_option == 'Upload a Picture':
            img_file_buffer = col3.file_uploader('Upload a Picture', type=allowed_image_type)
            if img_file_buffer is not None:
                file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)

        elif pic_option == 'Click a picture':
            img_file_buffer = col3.camera_input("Click a picture")
            if img_file_buffer is not None:
                file_bytes = np.frombuffer(img_file_buffer.getvalue(), np.uint8)

        if img_file_buffer is not None and len(face_name) > 1 and st.button('Click to Save!'):
            image_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            with open(os.path.join(VISITOR_DB, f'{face_name}.jpg'), 'wb') as file:
                file.write(img_file_buffer.getbuffer())

            face_locations = detect_faces(image_array)
            encodesCurFrame = []

            for face_location in face_locations:
                (left, top, right, bottom) = face_location
                roi = image_array[top:bottom, left:right]
                encode = get_face_encoding(roi)
                if encode is not None:
                    encodesCurFrame.append(encode)

            df_new = pd.DataFrame(data=encodesCurFrame, columns=COLS_ENCODE)
            df_new[COLS_INFO] = face_name
            df_new = df_new[COLS_INFO + COLS_ENCODE].copy()

            DB = initialize_data()
            add_data_db(df_new)


def get_face_encoding(roi):
    with mp_face_detection.FaceDetection(0.5) as face_detection:
        results = face_detection.process(roi)
        if results.detections:
            face_encoding = results.detections[0].location_data.relative_keypoints[0].x
            return face_encoding
        else:
            return None


def get_location():
    # Replace these coordinates with actual coordinates
    latitude, longitude = 37.7749, -122.4194
    return f"Latitude: {latitude}, Longitude: {longitude}"


if __name__ == "__main__":
    main()
