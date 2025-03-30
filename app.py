import streamlit as st
import models
import utils

def main():

    st.title('Age Gender and Emotion Detection from Facial Attributes')

    select_choices = {
        'Upload a Video': 1,
        'Open Camera': 0
    }

    source_choice = st.selectbox('Select Input Type', options=list(select_choices.keys()))

    if select_choices[source_choice]==1:
        uploaded_file = st.file_uploader('Upload a Video', type=['mp4'])
        if uploaded_file:
            video_path = utils.save_uploaded_video(uploaded_file)


    submit_btn = st.button('Submit')

    if submit_btn:
        if select_choices[source_choice]==0:
            models.stream_video(0)

        else:
            if uploaded_file and video_path:
                models.stream_video(video_path)
            else:
                st.error('Please upload a video')


if __name__=='__main__':
    main()