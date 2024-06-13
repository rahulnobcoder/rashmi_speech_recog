import streamlit as st
import soundfile as sf
from functions import *
from tensorflow.keras.models import load_model # type: ignore

model_path = 'rashmi_model.keras'
label_mapping={0: 'Angry', 1: 'Base', 2: 'Fear', 3: 'Happy', 4: 'Sad'}
# label_mapping={0: 'angry', 1: 'apaologetic' , 2:'calm',3:'Excited' ,4:'fear', 5: 'happy', 6: 'neutral', 7: 'sad'}

# Load the saved model
model = load_model(model_path)
# Set the title of the Streamlit app
st.title("Simple Streamlit Application")

# Text input for some data

# File uploader for audio files
audio_file = st.file_uploader("Upload an audio file:", type=["mp3", "wav"])

# Button to upload
if st.button("Upload and predict"):
    if audio_file:
        # Read the audio file
        audio_data, samplerate = sf.read(audio_file)
        feat=process(audio_data)
        print(feat)
        feat=feat.reshape((1,20))
        pred=np.argmax(model.predict([feat]),axis=1)
        print("predicted label : ",pred)
        st.write(f"pred: {label_mapping[pred[0]]}")
        st.audio(audio_file)
    else:
        st.error("Please upload an audio file.")

# Button to predict

# if st.button("Predict"):
#     if  audio_file:
#         audio_data, samplerate = librosa.load('uploaded_audio.wav', sr=sr)
#         feat=get_features(audio_data)
#         print(feat)
#         feat=feat.reshape((1,20))
#         pred=np.argmax(model.predict([feat]),axis=1)
#         print("predicted label : ",pred)
#         st.write(f"pred: {label_mapping[pred[0]]}")
#     else:
#         st.error("Please upload an audio file.")

# Additional message at the bottom of the page
st.write("Thank you for using the app!")