import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import utils


model_path = 'model_InV3_3.h5'

# side_menu = ['Welcome', 'Predict with Webcam']
side_menu = ['Welcome']

choice = st.sidebar.selectbox('Menu', side_menu)

model,_ = utils.create_model()
model.load_weights(model_path)

if choice=='Welcome':
    st.title('Vietnamese Note Prediction')
    st.write('')
    st.image('media/rick_n_morty_flower.jpeg')

    st.header('Please choose an image to upload:')
    upload_image = st.file_uploader('Upload test photo here:', ['png','jpeg','jpg'])
    
    if upload_image!=None:
        img_array = np.asarray(bytearray(upload_image.read()), dtype=np.uint8)
        
        img = cv2.imdecode(img_array, 1)
        st.image(img, channels='BGR')
        fname = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        prediction = utils.test_single(fname,model)
        st.write('Prediction of model: ',prediction)

        st.write('Size of photo: ', upload_image.size)
        st.write('Photo type: ', upload_image.type)


# elif choice=='Predict with Webcam':
#     st.title('Use Your Webcam to Capture Image')
#     st.warning('Webcam show on local computer ONLY')
    
#     task = st.selectbox('Select either show webcam or predict.', ['Show webcam', 'Predict image'])

#     if task == 'Show webcam':
#         show = st.checkbox('Show Webcam')
        
#         FRAME_WINDOW = st.image([])
#         camera = cv2.VideoCapture(0) # device 1 or 2

#         if not camera.isOpened():
#             raise IOError("Cannot open webcam")
    
#         while show:
            
#             _, frame = camera.read()

#             # frame = cv2.flip(frame, 1)
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#             # Display frame on streamlit
#             FRAME_WINDOW.image(frame)

#             # key = cv2.waitKey(1)
#             capture = st.button('Capture Image')
#             if not capture: 
#                 cv2.imwrite('NewPicture.jpg', frame)
#                 print('Screenshot is taken')
#                 capture = True
            
#         camera.release()
#         cv2.destroyAllWindows()

#     elif task == 'Predict image':
#         pred_butt = st.button('Start Prediction')

#         st.write(pred_butt)
#         if pred_butt:
#             upload_image_wc = cv2.imread('NewPicture.jpg')
#             st.image(upload_image_wc, channels='BGR')

#             # img_array_wc = np.asarray(bytearray(upload_image_wc.read()), dtype=np.uint8)

#             img_wc = cv2.imdecode(upload_image_wc, 1)
#             st.image(img_wc, channels='BGR')
#             fname_wc = cv2.cvtColor(img_wc, cv2.COLOR_BGR2RGB)

#             pred_wc = utils.test_single(fname_wc,model)
#             st.write('Prediction of model: ',pred_wc)

#             st.write('Size of photo: ', upload_image_wc.size)
#             st.write('Photo type: ', upload_image_wc.type)
#         else:
#             st.write('You don\'t have an image to predict!')
