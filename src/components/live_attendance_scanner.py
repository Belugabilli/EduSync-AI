import time
import cv2
import numpy as np
import streamlit as st
import av

from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration,
    VideoProcessorBase
)

from src.pipelines.face_pipeline import predict_attendance

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    }
)

class LiveAttendanceProcessor(VideoProcessorBase):
    def __init__(self):
        self.detected_ids = set()
        self.frame_count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Process every 5th frame to prevent video lag
        self.frame_count += 1
        if self.frame_count % 5 == 0:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            detected_student_dict, _, _ = predict_attendance(rgb_img)
            
            if detected_student_dict:
                for sid in detected_student_dict.keys():
                    self.detected_ids.add(sid)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def live_attendance_scanner(enrolled_students):
    st.subheader("🔴 Live Biometric Scanner")
    st.info("Point the camera at the classroom. Students will be marked present automatically as they are detected.")

    # Create a mapping of student ID to Name for quick lookup
    id_to_name = {}
    for node in enrolled_students:
        student = node.get("students")
        if student:
            id_to_name[int(student["student_id"])] = student["name"]

    ctx = webrtc_streamer(
        key="live-attendance-camera",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        video_processor_factory=LiveAttendanceProcessor,
        async_processing=True
    )

    if "final_detected_ids" not in st.session_state:
        st.session_state.final_detected_ids = set()

    status_placeholder = st.empty()

    if ctx.state.playing:
        st.warning("Scanner is ACTIVE. Do not navigate away.")
        
        # Real-time update loop
        while ctx.state.playing:
            if ctx.video_processor:
                detected = ctx.video_processor.detected_ids
                
                if detected:
                    st.session_state.final_detected_ids.update(detected)
                    
                    names = [id_to_name.get(sid, "Unknown") for sid in detected]
                    names.sort()
                    
                    html_names = "".join([f"<li style='color:#EB459E; font-weight:bold;'>{name}</li>" for name in names])
                    
                    status_placeholder.markdown(
                        f"""
                        <div style="background:#EB459E10; padding:20px; border-radius:15px; border:1px solid #EB459E;">
                            <h4>✅ Real-time Scanned ({len(names)} Students):</h4>
                            <ul>{html_names}</ul>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    status_placeholder.info("Scanning for faces... ⏳")
                    
            time.sleep(0.5)
            
    return st.session_state.final_detected_ids
