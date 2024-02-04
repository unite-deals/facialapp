import os
import streamlit as st
import os
import datetime
import json
import sys
import pathlib
import shutil
import pandas as pd
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

########################################################################################################################
# The Root Directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_CONFIG = os.path.join(ROOT_DIR, 'logging.yml')

STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static'

## We create a downloads directory within the streamlit static asset directory and we write output files to it
DOWNLOADS_PATH = (STREAMLIT_STATIC_PATH / "downloads")
if not DOWNLOADS_PATH.is_dir():
    HOME_DIR = os.path.expanduser('~')
    DOWNLOADS_PATH = pathlib.Path(HOME_DIR) / ".streamlit" / "downloads"

LOG_DIR = (STREAMLIT_STATIC_PATH / "logs")
if not LOG_DIR.is_dir():
    LOG_DIR.mkdir()

OUT_DIR = (STREAMLIT_STATIC_PATH / "output")
if not OUT_DIR.is_dir():
    OUT_DIR.mkdir()

VISITOR_DB = os.path.join(ROOT_DIR, "visitor_database")
# st.write(VISITOR_DB)

if not os.path.exists(VISITOR_DB):
    os.mkdir(VISITOR_DB)

VISITOR_HISTORY = os.path.join(ROOT_DIR, "visitor_history")
# st.write(VISITOR_HISTORY)

if not os.path.exists(VISITOR_HISTORY):
    os.mkdir(VISITOR_HISTORY)

########################################################################################################################
## Defining Parameters

COLOR_DARK = (0, 0, 153)
COLOR_WHITE = (255, 255, 255)
COLS_INFO = ['Name']
COLS_ENCODE = [f'v{i}' for i in range(128)]

## Database
data_path = VISITOR_DB
file_db = 'visitors_db.csv'  ## To store user information
file_history = 'visitors_history.csv'  ## To store visitor history information

## Image formats allowed
allowed_image_type = ['.png', 'jpg', '.jpeg']
################################################### Defining Function ##############################################
def initialize_data():
    if os.path.exists(os.path.join(data_path, file_db)):
        # st.info('Database Found!')
        df = pd.read_csv(os.path.join(data_path, file_db))

    else:
        # st.info('Database Not Found!')
        df = pd.DataFrame(columns=COLS_INFO + COLS_ENCODE)
        df.to_csv(os.path.join(data_path, file_db), index=False)

    return df

#################################################################
def add_data_db(df_visitor_details):
    try:
        df_all = pd.read_csv(os.path.join(data_path, file_db))

        if not df_all.empty:
            df_all = df_all.append(df_visitor_details, ignore_index=False)
            df_all.drop_duplicates(keep='first', inplace=True)
            df_all.reset_index(inplace=True, drop=True)
            df_all.to_csv(os.path.join(data_path, file_db), index=False)
            st.success('Details Added Successfully!')
        else:
            df_visitor_details.to_csv(os.path.join(data_path, file_db), index=False)
            st.success('Initiated Data Successfully!')

    except Exception as e:
        st.error(e)

#################################################################
# convert opencv BRG to regular RGB mode
def BGR_to_RGB(image_in_array):
    return cv2.cvtColor(image_in_array, cv2.COLOR_BGR2RGB)

#################################################################
def detect_faces(image_array):
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
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

#################################################################
def findEncodings(images):
    encode_list = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detect_faces(img)

        for face_location in faces:
            (left, top, right, bottom) = face_location
            roi = img[top:bottom, left:right]
            encode = get_face_encoding(roi)
            if encode is not None:
                encode_list.append(encode)

    return encode_list

#################################################################
def get_face_encoding(roi):
    # Your logic to convert the face region to encoding using mediapipe or other methods
    return None  # Placeholder, replace with actual implementation

#################################################################
def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * np.power(
            (linear_val - 0.5) * 2, 0.2))

#################################################################
def attendance(id, name):
    f_p = os.path.join(VISITOR_HISTORY, file_history)

    now = datetime.datetime.now()
    dtString = now.strftime('%Y-%m-%d %H:%M:%S')
    df_attendace_temp = pd.DataFrame(data={"id": [id],
                                           "visitor_name": [name],
                                           "Timing": [dtString]
                                           })

    if not os.path.isfile(f_p):
        df_attendace_temp.to_csv(f_p, index=False)
    else:
        df_attendace = pd.read_csv(f_p)
        df_attendace = df_attendace.append(df_attendace_temp)
        df_attendace.to_csv(f_p, index=False)

#################################################################
def view_attendace():
    f_p = os.path.join(VISITOR_HISTORY, file_history)
    df_attendace_temp = pd.DataFrame(columns=["id",
                                              "visitor_name", "Timing"])

    if not os.path.isfile(f_p):
        df_attendace_temp.to_csv(f_p, index=False)
    else:
        df_attendace_temp = pd.read_csv(f_p)

    df_attendace = df_attendace_temp.sort_values(by='Timing',
                                                 ascending=False)
    df_attendace.reset_index(inplace=True, drop=True)

    st.write(df_attendace)

    if df_attendace.shape[0] > 0:
        id_chk = df_attendace.loc[0, 'id']
        id_name = df_attendace.loc[0, 'visitor_name']

        selected_img = st.selectbox('Search Image using ID',
                                    options=['None'] + list(df_attendace['id']))

        avail_files = [file for file in list(os.listdir(VISITOR_HISTORY))
                       if ((file.endswith(tuple(allowed_image_type))) &
                           (file.startswith(selected_img) == True))]

        if len(avail_files) > 0:
            selected_img_path = os.path.join(VISITOR_HISTORY,
                                             avail_files[0])
            st.image(Image.open(selected_img_path))


########################################################################################################################

def main():
    st.sidebar.header("About")
    st.sidebar.info("This webapp gives a demo of Visitor Monitoring "
                    "Webapp using 'Face Recognition' and AI based monitoring for multiple use cases like attendance etc")

    selected_menu = st.sidebar.radio("Select an Option", ['Visitor Validation', 'View Visitor History', 'Add to Database'])

    if selected_menu == 'Visitor Validation':
        visitor_id = str(uuid.uuid1())
        img_file_buffer = st.camera_input("Take a picture")

        if img_file_buffer is not None:
            bytes_data = img_file_buffer.getvalue()
            image_array = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            with open(os.path.join(VISITOR_HISTORY, f'{visitor_id}.jpg'), 'wb') as file:
                file.write(img_file_buffer.getbuffer())
                st.success('Image Saved Successfully!')

                face_locations = detect_faces(image_array)

                if face_locations:
                    for face_location in face_locations:
                        (left, top, right, bottom) = face_location
                        roi = image_array[top:bottom, left:right]
                        encode = get_face_encoding(roi)
                        if encode is not None:
                            col1, col2 = st.columns(2)
                            face_idxs = col1.multiselect("Select face#", range(1), default=[0])
                            similarity_threshold = col2.slider('Select Threshold for Similarity', min_value=0.0, max_value=1.0,
                                                               value=0.5)
                            if col1.checkbox('Click to proceed!') and face_idxs:
                                database_data = initialize_data()
                                dataframe = database_data[COLS_INFO]
                                database_encodes = database_data[COLS_ENCODE].values

                                face_to_compare = encode
                                dataframe['distance'] = np.linalg.norm(database_encodes - face_to_compare, axis=1)
                                dataframe['distance'] = dataframe['distance'].astype(float)

                                dataframe['similarity'] = dataframe.distance.apply(
                                    lambda distance: f"{face_distance_to_conf(distance):0.2}")
                                dataframe['similarity'] = dataframe['similarity'].astype(float)

                                dataframe_new = dataframe[dataframe['similarity'] > similarity_threshold].head(1)
                                dataframe_new.reset_index(drop=True, inplace=True)

                                if dataframe_new.shape[0] > 0:
                                    (left, top, right, bottom) = face_location
                                    roi = image_array[top:bottom, left:right]

                                    cv2.rectangle(image_array, (left, top), (right, bottom), COLOR_DARK, 2)
                                    cv2.rectangle(image_array, (left, bottom + 35), (right, bottom), COLOR_DARK,
                                                  cv2.FILLED)
                                    font = cv2.FONT_HERSHEY_DUPLEX
                                    cv2.putText(image_array, f"#{dataframe_new.loc[0, 'Name']}",
                                                (left + 5, bottom + 25), font, .55, COLOR_WHITE, 1)

                                    name_visitor = dataframe_new.loc[0, 'Name']
                                    attendance(visitor_id, name_visitor)
                                    st.image(BGR_to_RGB(image_array), width=720)
                                else:
                                    st.error(f'No Match Found for the given Similarity Threshold!')
                                    attendance(visitor_id, 'Unknown')
                            else:
                                st.error('Please select a face and check the checkbox to proceed.')
                else:
                    st.error('No faces detected in the image.')

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


def get_location():
    # Replace these coordinates with actual coordinates
    latitude, longitude = 37.7749, -122.4194
    return f"Latitude: {latitude}, Longitude: {longitude}"


if __name__ == "__main__":
    main()
