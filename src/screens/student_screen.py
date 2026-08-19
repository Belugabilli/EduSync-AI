import streamlit as st

from src.ui.base_layout import style_background_dashboard, style_base_layout

from src.components.header import header_dashboard
from src.components.footer import footer_dashboard
from PIL import Image
import numpy as np
from src.pipelines.face_pipeline import (
    predict_attendance,
    train_classifier,
    get_enrollment_embeddings
)
from src.database.db import (
    get_all_students,
    create_student,
    student_login,
    get_student_subjects,
    get_student_attendance,
    unenroll_student_to_subject
)
from src.database.config import supabase
import time

from src.components.dialog_enroll import enroll_dialog
from src.components.subject_card import subject_card
from src.components.face_enrollment import face_enrollment_video

def login_student(registration_number, password):

    if not registration_number or not password:
        return False

    student = student_login(
        registration_number.strip(),
        password
    )

    if student:

        st.session_state.user_role = "student"
        st.session_state.student_data = student
        st.session_state.is_logged_in = True

        return True

    return False

def register_student(
    registration_number,
    student_name,
    password,
    password_confirm
):

    if not registration_number or not student_name or not password:
        return False, "All fields are required!"

    if password != password_confirm:
        return False, "Passwords do not match!"

    return True, "Details validated!"

def student_dashboard():
    student_data = st.session_state.student_data
    student_id = student_data['student_id']
    c1, c2 = st.columns(2, vertical_alignment='center', gap='xxlarge')
    with c1:
        header_dashboard()
    with c2:
        st.subheader(f"""Welcome, {student_data['name']} """)
        if st.button("Logout", type='secondary', key='loginbackbtn', shortcut="control+backspace"):
            st.session_state['is_logged_in'] = False
            del st.session_state.student_data 
            st.rerun()


    st.space()

    c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment='center')
    with c1:
        st.header('Your Enrolled Subjects')
    with c2:
        if st.button('Refresh', type='secondary', width='stretch', icon=":material/refresh:"):
            st.rerun()
    with c3:
        if st.button('Enroll in Subject', type='primary', width='stretch'):
            enroll_dialog()


    st.divider()

    with st.spinner('Loading your enrolled subjects..'):
        subjects = get_student_subjects(student_id)
        logs = get_student_attendance(student_id)

    stats_map = {}

    for log in logs:
        sid = log['subject_id']
        if sid not in stats_map:
            stats_map[sid] = {"total":0, "attended": 0}

        stats_map[sid]['total'] +=1
        if log.get('is_present'):
            stats_map[sid]['attended'] += 1

    tab1, tab2 = st.tabs(["📚 My Subjects", "📊 Attendance History"])

    with tab1:
        st.write("")
        cols = st.columns(2)
        for i, sub_node in enumerate(subjects):
            sub = sub_node['subjects']
            sid = sub['subject_id']

            stats = stats_map.get(sid, {"total":0, "attended": 0})
            def unenroll_button():
                if st.button("Unenroll from this course", type='tertiary', width='stretch', icon=':material/delete_forever:', key=f"unenroll_{sid}"):
                    unenroll_student_to_subject(student_id, sid)
                    st.toast(f"Unenrolled from {sub['name']} successfully!")
                    st.rerun()

            with cols[i % 2]:
                teacher = sub.get('teachers')
                faculty_name = teacher.get('name') if teacher else None

                subject_card(
                    name = sub['name'],
                    code =sub['subject_code'],
                    slot = sub['slot'],
                    faculty = faculty_name,
                    stats = [
                        ('📅', 'Total Classes', stats['total']),
                        ('✅', 'Attended', stats['attended']),
                    ],
                    footer_callback=unenroll_button
                )

    with tab2:
        st.write("")
        if not subjects:
            st.info("You are not enrolled in any subjects yet.")
        else:
            import pandas as pd
            
            # Group logs by subject
            subject_logs = {}
            for log in logs:
                sid = log['subject_id']
                if sid not in subject_logs:
                    subject_logs[sid] = []
                subject_logs[sid].append(log)
                
            for sub_node in subjects:
                sub = sub_node['subjects']
                sid = sub['subject_id']
                stats = stats_map.get(sid, {"total":0, "attended": 0})
                
                # Calculate percentage
                if stats['total'] > 0:
                    percent = (stats['attended'] / stats['total']) * 100
                else:
                    percent = 0.0
                    
                st.subheader(f"{sub['name']} ({sub['subject_code']})")
                
                # Show quick stats
                st.markdown(f"**Overall Attendance:** `{percent:.1f}%` ({stats['attended']}/{stats['total']} classes)")
                
                sub_logs = subject_logs.get(sid, [])
                if sub_logs:
                    # Create DataFrame for display
                    df_data = []
                    for log in sub_logs:
                        date_str = log['attendance_date']
                        status = "✅ Present" if log.get('is_present') else "❌ Absent"
                        df_data.append({"Date": date_str, "Status": status})
                        
                    df = pd.DataFrame(df_data)
                    # Sort by date descending
                    df = df.sort_values(by="Date", ascending=False).reset_index(drop=True)
                    
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.caption("No attendance logs found for this subject.")
                    
                st.divider()

    footer_dashboard()


def student_screen():

    style_background_dashboard()
    style_base_layout()

    # Already logged in
    if "student_data" in st.session_state:
        student_dashboard()
        return

    # Default screen = Login
    if "student_login_type" not in st.session_state:
        st.session_state.student_login_type = "login"

    # =========================
    # HEADER
    # =========================

    c1, c2 = st.columns(
        2,
        vertical_alignment="center",
        gap="xxlarge"
    )

    with c1:
        header_dashboard()

    with c2:
        if st.button(
            "Go back to Home",
            type="secondary",
            key="student_home_btn",
            shortcut="control+backspace"
        ):
            st.session_state["login_type"] = None
            st.session_state.student_login_type = "login"
            st.rerun()

    # =====================================================
    # STUDENT LOGIN
    # =====================================================

    if st.session_state.student_login_type == "login":

        st.header(
            "Student Login",
            text_alignment="center"
        )

        st.caption(
            "Login using your Registration Number and Password",
            text_alignment="center"
        )

        st.space()

        registration_number = st.text_input(
            "Registration Number",
            placeholder="E.g. 25BCE10632"
        )

        student_password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password"
        )

        st.space()

        btn1, btn2 = st.columns(2)

        with btn1:

            if st.button(
                "Login",
                type="primary",
                icon=":material/login:",
                width="stretch"
            ):

                if not registration_number or not student_password:

                    st.warning(
                        "Registration Number and Password are required!"
                    )

                else:

                    student = student_login(
                        registration_number.strip(),
                        student_password
                    )

                    if student:

                        st.session_state.is_logged_in = True
                        st.session_state.user_role = "student"
                        st.session_state.student_data = student

                        st.toast(
                            f"Welcome Back {student['name']}! 👋"
                        )

                        time.sleep(1)
                        st.rerun()

                    else:

                        st.error(
                            "Invalid Registration Number or Password."
                        )

        with btn2:

            if st.button(
                "Register Instead",
                icon=":material/person_add:",
                width="stretch"
            ):

                st.session_state.student_login_type = "register"
                st.rerun()

    # =====================================================
    # STUDENT REGISTRATION
    # =====================================================

    else:

        st.header(
            "Create Student Account",
            text_alignment="center"
        )

        st.caption(
            "Create your EduSync AI student profile",
            text_alignment="center"
        )

        st.space()

        registration_number = st.text_input(
            "Registration Number",
            placeholder="E.g. 25BCE10632"
        )

        student_name = st.text_input(
            "Full Name",
            placeholder="E.g. Hanish Singla"
        )

        student_email = st.text_input(
            "Email Address",
            placeholder="E.g. hanish.25bce10632@vitbhopal.ac.in"
        )

        student_password = st.text_input(
            "Password",
            type="password",
            placeholder="Create a password"
        )

        student_password_confirm = st.text_input(
            "Confirm Password",
            type="password",
            placeholder="Re-enter your password"
        )

        st.divider()

        # =========================
        # FACE ENROLLMENT
        # =========================

        st.subheader("AI Face Enrollment")

        enrollment_frames = face_enrollment_video()

        st.space()

        btn1, btn2 = st.columns(2)

        # =========================
        # CREATE ACCOUNT
        # =========================

        with btn1:

            if st.button(
                "Create Account",
                type="primary",
                icon=":material/person_add:",
                width="stretch"
            ):

                # Required fields
                if (
                    not registration_number
                    or not student_name
                    or not student_email
                    or not student_password
                ):

                    st.warning(
                        "Registration Number, Name, Email, and Password "
                        "are required!"
                    )

                # Password confirmation
                elif student_password != student_password_confirm:

                    st.error(
                        "Passwords do not match!"
                    )

                else:

                    registration_number = (
                        registration_number.strip()
                    )
                    student_email = student_email.strip()

                    # Check unique Registration Number
                    from src.database.db import check_student_exists

                    if check_student_exists(
                        registration_number
                    ):

                        st.error(
                            "This Registration Number is already "
                            "registered!"
                        )

                    # Face is mandatory
                    elif not enrollment_frames:

                        st.warning(
                            "Please complete the AI Face Enrollment "
                            "before creating your account."
                        )

                    else:

                        with st.spinner(
                            "Analyzing your enrollment frames..."
                        ):

                            face_embeddings = get_enrollment_embeddings(
                                enrollment_frames
                            )

                            if not face_embeddings:

                                st.error(
                                    "Couldn't extract facial features "
                                    "from the enrollment video. "
                                    "Please try the enrollment again."
                                )

                            elif len(face_embeddings) < 3:

                                st.error(
                                    "Not enough good face samples were "
                                    "captured. Please complete the "
                                    "enrollment process."
                                )

                            else:

                                final_embedding = (
                                    average_embeddings(
                                        face_embeddings
                                    )
                                )

                                try:

                                    response_data = create_student(
                                        registration_number,
                                        student_name.strip(),
                                        student_password,
                                        face_embedding=final_embedding,
                                        email=student_email
                                    )

                                    if response_data:

                                        train_classifier()

                                        # Clear enrollment data
                                        st.session_state.enrollment_frames = []
                                        st.session_state.enrollment_pose = 0
                                        st.session_state.enrollment_started = False
                                        st.session_state.enrollment_complete = False

                                        st.success(
                                            "Account created successfully! "
                                            "Your AI face profile has been enrolled."
                                        )

                                        st.session_state.student_login_type = "login"

                                        time.sleep(1)

                                        st.rerun()

                                except Exception as e:

                                    if (
                                        "registration_number"
                                        in str(e).lower()
                                    ):

                                        st.error(
                                            "This Registration Number "
                                            "is already registered!"
                                        )

                                    else:

                                        st.error(
                                            f"Registration failed: {str(e)}"
                                        )

        # =========================
        # GO TO LOGIN
        # =========================

        with btn2:

            if st.button(
                "Login Instead",
                icon=":material/login:",
                width="stretch"
            ):

                st.session_state.student_login_type = "login"
                st.rerun()

    footer_dashboard()
