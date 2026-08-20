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


POSES = [
    ("front", "Look Straight", "Look directly at the camera."),
    ("left", "Turn LEFT", "Slowly turn your face to the LEFT."),
    ("right", "Turn RIGHT", "Slowly turn your face to the RIGHT."),
    ("up", "Look UP", "Slowly look UP."),
    ("down", "Look DOWN", "Slowly look DOWN."),
    ("blink", "BLINK", "Blink your eyes naturally.")
]


class FaceEnrollmentProcessor(VideoProcessorBase):

    def __init__(self):
        self.latest_frame = None

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        self.latest_frame = img.copy()

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


def face_enrollment_video():

    st.subheader("🎥 AI Face Enrollment")

    st.info(
        "We will capture your face from multiple angles "
        "to make recognition more reliable."
    )

    st.markdown(
        """
        **Enrollment instructions**

        1. Look straight at the camera
        2. Turn your face LEFT
        3. Turn your face RIGHT
        4. Look UP
        5. Look DOWN
        6. Blink naturally
        """
    )

    if "enrollment_frames" not in st.session_state:
        st.session_state.enrollment_frames = []

    if "enrollment_started" not in st.session_state:
        st.session_state.enrollment_started = False

    if "enrollment_complete" not in st.session_state:
        st.session_state.enrollment_complete = False

    if "enrollment_pose" not in st.session_state:
        st.session_state.enrollment_pose = 0

    if not st.session_state.enrollment_started:

        if st.button(
            "Start Face Enrollment",
            type="primary",
            width="stretch",
            icon=":material/face:"
        ):

            st.session_state.enrollment_started = True
            st.session_state.enrollment_pose = 0
            st.session_state.enrollment_frames = []
            st.session_state.enrollment_complete = False

            st.rerun()

        return None

    current_pose = st.session_state.enrollment_pose

    if current_pose >= len(POSES):

        st.session_state.enrollment_complete = True

        st.success(
            f"Face enrollment complete! "
            f"{len(st.session_state.enrollment_frames)} "
            f"samples captured."
        )

        return st.session_state.enrollment_frames

    pose_key, pose_title, pose_instruction = POSES[current_pose]

    st.markdown(
        f"""
        ### Step {current_pose + 1} / {len(POSES)}

        ## {pose_title}

        **{pose_instruction}**

        Keep your entire face visible in the camera.
        """
    )

    ctx = webrtc_streamer(
        key="face-enrollment-camera",

        mode=WebRtcMode.SENDRECV,

        rtc_configuration=RTC_CONFIGURATION,

        media_stream_constraints={
            "video": {
                "width": {"ideal": 1280, "min": 640},
                "height": {"ideal": 720, "min": 480}
            },
            "audio": False
        },

        video_processor_factory=FaceEnrollmentProcessor,

        async_processing=True
    )

    if ctx.state.playing and ctx.video_processor:

        if st.button(
            f"Capture {pose_title}",
            type="primary",
            width="stretch",
            key=f"capture_{pose_key}"
        ):
            frame = ctx.video_processor.latest_frame
            if frame is not None:
                rgb_img = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )
                st.session_state.enrollment_frames.append(
                    rgb_img
                )
                st.session_state.enrollment_pose += 1
                time.sleep(0.2)
                st.rerun()
            else:
                st.warning(
                    "Camera frame not available. "
                    "Please wait a moment and try again."
                )

        frame = ctx.video_processor.latest_frame
        if frame is not None:
            display_img = cv2.flip(
                frame,
                1
            )
            st.image(
                display_img,
                channels="BGR",
                width="stretch"
            )

    st.caption(
        f"Captured "
        f"{len(st.session_state.enrollment_frames)} / "
        f"{len(POSES)} samples"
    )

    return None