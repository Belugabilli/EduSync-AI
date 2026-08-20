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
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
            {"urls": ["stun:stun4.l.google.com:19302"]},
            {"urls": ["stun:global.stun.twilio.com:3478"]},
            {"urls": ["stun:stun.stunprotocol.org:3478"]},
            {"urls": ["stun:stun.cloudflare.com:3478"]}
        ]
    }
)

class LiveAttendanceProcessor(VideoProcessorBase):
    def __init__(self):
        self.detected_ids = set()
        self.frame_count = 0
        self.detection_counts = {}

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Process every 5th frame to prevent video lag
        self.frame_count += 1
        if self.frame_count % 5 == 0:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Downscale to max 640px width to ensure real-time ML performance
            if rgb_img.shape[1] > 640:
                ratio = 640.0 / rgb_img.shape[1]
                new_dim = (640, int(rgb_img.shape[0] * ratio))
                ml_img = cv2.resize(rgb_img, new_dim, interpolation=cv2.INTER_AREA)
            else:
                ml_img = rgb_img

            detected_student_dict, _, _ = predict_attendance(ml_img)
            
            if detected_student_dict:
                for sid in detected_student_dict.keys():
                    self.detection_counts[sid] = self.detection_counts.get(sid, 0) + 1
                    
                    # Require 3 positive hits to eliminate 1-frame glitches (False Positives)
                    if self.detection_counts[sid] >= 3:
                        self.detected_ids.add(sid)

        # Return a smaller version of the image to the browser to prevent lag
        if img.shape[1] > 640:
            scale = 640.0 / img.shape[1]
            display_img = cv2.resize(img, (640, int(img.shape[0] * scale)))
        else:
            display_img = img

        return av.VideoFrame.from_ndarray(display_img, format="bgr24")

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
    else:
        # Not playing. Display final results if they exist.
        if st.session_state.final_detected_ids:
            names = [id_to_name.get(sid, "Unknown") for sid in st.session_state.final_detected_ids]
            names.sort()
            
            html_names = "".join([f"<li style='color:#EB459E; font-weight:bold;'>{name}</li>" for name in names])
            
            status_placeholder.markdown(
                f"""
                <div style="background:#EB459E10; padding:20px; border-radius:15px; border:1px solid #EB459E;">
                    <h4>✅ Scanner Stopped. Scanned ({len(names)} Students):</h4>
                    <ul>{html_names}</ul>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
    return st.session_state.final_detected_ids
