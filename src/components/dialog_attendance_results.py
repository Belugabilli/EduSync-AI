import streamlit as st
from src.database.db import enroll_student_to_subject
from src.database.config import supabase
import time


from src.database.db import create_attendance

def show_attendance_result(df, logs):
    st.write('Please review attendance before confirming.')
    st.dataframe(df, hide_index=True, width='stretch')

    col1, col2 = st.columns(2)

    with col1:
        if st.button('Discard', width='stretch'):
            st.session_state.attendance_images = []
            st.session_state.attendance_media_nonce = (
                st.session_state.get(
                    "attendance_media_nonce",
                    0
                ) + 1
            )
            st.rerun()

    with col2:

        if st.button(
            'Confirm & Save',
            width='stretch',
            type='primary',
            icon=':material/save:'
        ):

            try:

                create_attendance(logs)

                st.toast(
                    "Attendance saved successfully! ✅"
                )

                st.session_state.attendance_images = []
                st.session_state.attendance_media_nonce = (
                    st.session_state.get(
                        "attendance_media_nonce",
                        0
                    ) + 1
                )

                st.rerun()

            except Exception as e:

                st.error(
                    f"Failed to save attendance: {str(e)}"
                )



@st.dialog("Attendance Reports")
def attendance_result_dialog(df, logs):
    show_attendance_result(df, logs)

