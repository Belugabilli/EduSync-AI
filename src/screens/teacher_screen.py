import streamlit as st
import re
import os
import tempfile
import cv2

from src.ui.base_layout import style_background_dashboard, style_base_layout

from src.components.header import header_dashboard
from src.components.footer import footer_dashboard
from src.components.subject_card import subject_card
from src.database.db import (
    check_teacher_exists,
    create_teacher,
    teacher_login,
    get_teacher_subjects,
    get_attendance_for_teacher,
    get_attendance_for_date,
    update_attendance,
    get_student_by_registration_number,
    get_subject_students,
    get_subject_attendance_summary,
    enroll_student_to_subject
)
from src.components.dialog_create_subject import create_subject_dialog
from src.components.dialog_share_subject import share_subject_dialog
from src.components.dialog_add_photo import add_photos_dialog
from src.components.live_attendance_scanner import live_attendance_scanner

from src.pipelines.face_pipeline import predict_attendance
from src.components.dialog_attendance_results import attendance_result_dialog
import numpy as np

from datetime import datetime

import pandas as pd

from src.database.config import supabase


def teacher_screen():

    style_background_dashboard()
    style_base_layout()

    if "teacher_data" in st.session_state:
        teacher_dashboard()
    elif 'teacher_login_type' not in st.session_state or st.session_state.teacher_login_type=="login":
        teacher_screen_login()
    elif st.session_state.teacher_login_type == "register":
        teacher_screen_register()





def mark_detected_student(
    detected_sources,
    student_id,
    source_label
):

    detected_sources.setdefault(
        int(student_id),
        []
    ).append(source_label)


def get_marked_student_names(
    enrolled_students,
    detected_sources
):

    marked_ids = {
        int(student_id)
        for student_id in detected_sources.keys()
    }

    names = []

    for node in enrolled_students:
        student = node.get("students")

        if (
            student
            and int(student["student_id"]) in marked_ids
        ):
            names.append(student["name"])

    return sorted(names)


def process_attendance_video(
    video_bytes,
    video_label,
    enrolled_students,
    detected_sources,
    progress_placeholder,
    status_placeholder
):

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as temp_file:
            temp_file.write(video_bytes)
            temp_path = temp_file.name

        capture = cv2.VideoCapture(temp_path)

        total_frames = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if total_frames <= 0:
            total_frames = 1

        frame_step = max(1, total_frames // 20)
        frame_index = 0

        while True:

            success, frame = capture.read()

            if not success:
                break

            if frame_index % frame_step == 0:

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                detected, _, _ = predict_attendance(
                    rgb_frame
                )

                if detected:
                    for sid in detected.keys():
                        mark_detected_student(
                            detected_sources,
                            sid,
                            f"{video_label} frame {frame_index}"
                        )

                    marked_names = get_marked_student_names(
                        enrolled_students,
                        detected_sources
                    )

                    status_placeholder.info(
                        "Marked so far: "
                        + ", ".join(marked_names)
                    )

                progress_placeholder.progress(
                    min(
                        (frame_index + 1) / total_frames,
                        1.0
                    )
                )

            frame_index += 1

        capture.release()

    finally:

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def teacher_dashboard():
    teacher_data = st.session_state.teacher_data
    c1, c2 = st.columns(2, vertical_alignment='center', gap='xxlarge')
    with c1:
        header_dashboard()
    with c2:
        st.subheader(f"""Welcome, {teacher_data['name']} """)
        if st.button("Logout", type='secondary', key='loginbackbtn', shortcut="control+backspace"):
            st.session_state['is_logged_in'] = False
            del st.session_state.teacher_data 
            st.rerun()


    st.space()

    if "current_teacher_tab" not in st.session_state:
        st.session_state.current_teacher_tab = 'take_attendance'
    tab1, tab2, tab3, tab4 = st.columns(4)


    with tab1:
        type1 = "primary" if st.session_state.current_teacher_tab == 'take_attendance' else "tertiary"
        if st.button('Take Attendance',type=type1, width='stretch', icon=':material/ar_on_you:'):
            st.session_state.current_teacher_tab = 'take_attendance'
            st.rerun()

    with tab2:
        type2 = "primary" if st.session_state.current_teacher_tab == 'manage_subjects' else "tertiary"
        if st.button('Manage Subjects', type=type2, width='stretch', icon=':material/book_ribbon:'):
            st.session_state.current_teacher_tab = 'manage_subjects'
            st.rerun()

    with tab3:
        type3 = "primary" if st.session_state.current_teacher_tab == 'manage_students' else "tertiary"
        if st.button('Manage Students',type=type3, width='stretch', icon=':material/groups:'):
            st.session_state.current_teacher_tab = 'manage_students'
            st.rerun()

    with tab4:
        type4 = "primary" if st.session_state.current_teacher_tab == 'attendance_records' else "tertiary"
        if st.button('Attendance Records',type=type4, width='stretch', icon=':material/cards_stack:'):
            st.session_state.current_teacher_tab = 'attendance_records'
            st.rerun()


    st.divider()

    if st.session_state.current_teacher_tab == "take_attendance":
        teacher_tab_take_attendance()
    if st.session_state.current_teacher_tab == "manage_subjects":
        teacher_tab_manage_subjects()
    if st.session_state.current_teacher_tab == "manage_students":
        teacher_tab_manage_students()
    if st.session_state.current_teacher_tab == "attendance_records":
        teacher_tab_attendance_records()

    


    footer_dashboard()

def teacher_tab_take_attendance():

    teacher_id = st.session_state.teacher_data['teacher_id']

    st.header('Take AI Attendance')

    # ---------------------------------------------------------
    # Attendance images state
    # ---------------------------------------------------------

    if 'attendance_images' not in st.session_state:
        st.session_state.attendance_images = []

    if "attendance_media_nonce" not in st.session_state:
        st.session_state.attendance_media_nonce = 0

    # ---------------------------------------------------------
    # Get teacher's courses
    # ---------------------------------------------------------

    subjects = get_teacher_subjects(teacher_id)

    if not subjects:
        st.warning(
            "You haven't created any courses yet! "
            "Please create one to begin."
        )
        return

    # ---------------------------------------------------------
    # Course selection
    # ---------------------------------------------------------

    subject_options = {
        f"{s['name']} - {s['subject_code']} - Slot {s['slot']}":
            s['subject_id']
        for s in subjects
    }

    selected_subject_label = st.selectbox(
        'Select Course',
        options=list(subject_options.keys())
    )

    selected_subject_id = subject_options[
        selected_subject_label
    ]

    # ---------------------------------------------------------
    # Attendance date
    # ---------------------------------------------------------

    selected_date = st.date_input(
        "Attendance Date",
        value=datetime.now().date(),
        format="DD/MM/YYYY"
    )

    # ---------------------------------------------------------
    # Check whether attendance already exists
    # ---------------------------------------------------------

    existing_attendance = (
        supabase
        .table("attendance_logs")
        .select("id")
        .eq("subject_id", selected_subject_id)
        .eq("attendance_date", selected_date.isoformat())
        .execute()
    )

    if existing_attendance.data:

        st.warning(
            "Attendance has already been recorded for this "
            "course on this date."
        )

        st.info(
            "Use the Attendance Records section to edit "
            "the existing attendance."
        )

    # ---------------------------------------------------------
    # Add photos
    # ---------------------------------------------------------

    col1, col2 = st.columns(
        [3, 1],
        vertical_alignment='bottom'
    )

    with col1:

        st.caption(
            f"Attendance Date: "
            f"**{selected_date.strftime('%d/%m/%Y')}**"
        )

    with col2:

        if st.button(
            'Add Photos',
            type='primary',
            icon=':material/photo_prints:',
            width='stretch',
            disabled=bool(existing_attendance.data)
        ):
            add_photos_dialog()

    # ---------------------------------------------------------
    # Display added photos
    # ---------------------------------------------------------

    st.divider()

    if st.session_state.attendance_images:

        st.header('Added Photos')

        gallery_cols = st.columns(4)

        for idx, img in enumerate(
            st.session_state.attendance_images
        ):

            with gallery_cols[idx % 4]:

                st.image(
                    img,
                    width='stretch',
                    caption=f'Photo {idx + 1}'
                )

    has_photos = bool(
        st.session_state.attendance_images
    )

    st.divider()

    uploader_key_suffix = (
        st.session_state.attendance_media_nonce
    )

    uploaded_videos = st.file_uploader(
        "Upload classroom videos",
        type=["mp4", "mov", "avi", "mkv"],
        accept_multiple_files=True,
        disabled=bool(existing_attendance.data),
        key=f"attendance_videos_{uploader_key_suffix}"
    )

    has_videos = bool(uploaded_videos)

    if uploaded_videos:
        st.caption(
            "Videos selected: "
            + ", ".join(video.name for video in uploaded_videos)
        )

    # ---------------------------------------------------------
    # Live Biometric Scanner
    # ---------------------------------------------------------

    st.divider()
    
    enrolled_res = (
        supabase
        .table('subject_students')
        .select("*, students(*)")
        .eq('subject_id', selected_subject_id)
        .execute()
    )
    enrolled_students = enrolled_res.data or []

    if not existing_attendance.data:
        live_attendance_scanner(enrolled_students)
        
    has_live_scans = bool(st.session_state.get('final_detected_ids', set()))

    # ---------------------------------------------------------
    # Attendance controls
    # ---------------------------------------------------------

    st.divider()

    c1, c2, c3 = st.columns(3)

    with c1:

        if st.button(
            'Clear all media',
            width='stretch',
            type='tertiary',
            icon=':material/delete:',
            disabled=not (
                has_photos
                or has_videos
                or has_live_scans
            )
        ):

            st.session_state.attendance_images = []
            st.session_state.attendance_media_nonce += 1
            if 'final_detected_ids' in st.session_state:
                st.session_state.final_detected_ids = set()

            st.rerun()

    with c2:

        if st.button(
            'Run AI Attendance',
            width='stretch',
            type='secondary',
            icon=':material/analytics:',
            disabled=(
                (
                    not has_photos
                    and not has_videos
                    and not has_live_scans
                )
                or bool(existing_attendance.data)
            )
        ):

            status_placeholder = st.empty()
            progress_placeholder = st.progress(0.0)

            with st.spinner(
                'Scanning class and marking attendance...'
            ):

                detected_sources = {}

                enrolled_res = (
                    supabase
                    .table('subject_students')
                    .select("*, students(*)")
                    .eq(
                        'subject_id',
                        selected_subject_id
                    )
                    .execute()
                )

                enrolled_students = enrolled_res.data

                if not enrolled_students:

                    st.warning(
                        'No students are enrolled in this course.'
                    )

                    return

                total_steps = max(
                    1,
                    len(st.session_state.attendance_images)
                    + len(uploaded_videos or [])
                )

                completed_steps = 0

                for idx, img in enumerate(
                    st.session_state.attendance_images
                ):

                    status_placeholder.info(
                        f"Scanning photo {idx + 1}..."
                    )

                    img_np = np.array(
                        img.convert('RGB')
                    )

                    detected, _, _ = predict_attendance(
                        img_np
                    )

                    if detected:
                        for sid in detected.keys():
                            mark_detected_student(
                                detected_sources,
                                sid,
                                f"Photo {idx + 1}"
                            )

                        marked_names = get_marked_student_names(
                            enrolled_students,
                            detected_sources
                        )

                        status_placeholder.info(
                            "Marked so far: "
                            + ", ".join(marked_names)
                        )

                    completed_steps += 1
                    progress_placeholder.progress(
                        completed_steps / total_steps
                    )

                for idx, video in enumerate(
                    uploaded_videos or []
                ):
                    status_placeholder.info(
                        f"Scanning video {idx + 1}: {video.name}"
                    )

                    process_attendance_video(
                        video.getvalue(),
                        f"Video {idx + 1}",
                        enrolled_students,
                        detected_sources,
                        progress_placeholder,
                        status_placeholder
                    )

                    completed_steps += 1
                    progress_placeholder.progress(
                        completed_steps / total_steps
                    )

                if has_live_scans:
                    for sid in st.session_state.final_detected_ids:
                        mark_detected_student(
                            detected_sources,
                            sid,
                            "Live Biometric Scanner"
                        )

                results = []
                attendance_to_log = []

                for node in enrolled_students:

                    student = node['students']

                    sources = detected_sources.get(
                        int(student['student_id']),
                        []
                    )

                    is_present = len(sources) > 0

                    results.append({
                        "Name": student['name'],
                        "Registration Number":
                            student['registration_number'],
                        "Source":
                            ", ".join(sources)
                            if is_present
                            else "-",
                        "Status":
                            "✅ Present"
                            if is_present
                            else "❌ Absent"
                    })

                    attendance_to_log.append({
                        "student_id":
                            student['student_id'],
                        "subject_id":
                            selected_subject_id,
                        "attendance_date":
                            selected_date.isoformat(),
                        "is_present":
                            is_present
                    })

                progress_placeholder.progress(1.0)
                status_placeholder.success(
                    "Class scanning completed."
                )

                if results:
                    attendance_result_dialog(
                        pd.DataFrame(results),
                        attendance_to_log
                    )

def teacher_tab_manage_subjects():
    teacher_id = st.session_state.teacher_data['teacher_id']
    col1, col2 = st.columns(2)
    with col1:
        st.header('Manage Subjects', width='stretch')

    with col2:
        if st.button('Create New Subject', width='stretch'):
            create_subject_dialog(teacher_id)


    # LIST all SUBJECTS
    subjects = get_teacher_subjects(teacher_id)
    if subjects:
        for sub in subjects:
            stats = [
                ("🫂", "Students", sub['total_students']),
                ("🕰️", "Classes", sub['total_classes']),
            ]
        def share_btn():
            if st.button(
                f"Share Code: {sub['name']}",
                key=f"share_{sub['subject_id']}",
                icon=":material/share:"
            ):
                share_subject_dialog(
                    sub['name'],
                    sub['subject_code'],
                    sub['slot'],
                    sub['joining_code']
                )
            st.space()

        subject_card(
            name = sub['name'],
            code = sub['subject_code'],
            slot = sub['slot'],
            stats=stats,
            footer_callback=share_btn
        )
    else:
        st.info("NO SUBJECTS FOUND. CREATE ONE ABOVE")


def teacher_tab_manage_students():

    st.header("Manage Students")

    teacher_id = st.session_state.teacher_data["teacher_id"]

    subjects = get_teacher_subjects(teacher_id)

    if not subjects:
        st.info("No courses found.")
        return

    subject_options = {
        f"{s['name']} - {s['subject_code']} - Slot {s['slot']}":
            s["subject_id"]
        for s in subjects
    }

    selected_subject_label = st.selectbox(
        "Select Course",
        list(subject_options.keys()),
        key="manage_students_subject"
    )

    selected_subject_id = subject_options[
        selected_subject_label
    ]

    st.caption(selected_subject_label)

    registration_number_input = st.text_area(
        "Add students by registration number",
        placeholder=(
            "Enter registration numbers separated by commas, "
            "spaces, or new lines"
        ),
        key="manage_students_registration_number"
    )

    if st.button(
        "Add Student",
        type="primary",
        icon=":material/person_add:",
        width="stretch"
    ):

        raw_registration_numbers = [
            value.strip()
            for value in re.split(
                r"[\s,]+",
                registration_number_input.strip()
            )
            if value.strip()
        ]

        if not raw_registration_numbers:
            st.warning(
                "Please enter at least one registration number."
            )
        else:
            existing_students = get_subject_students(
                selected_subject_id
            )

            enrolled_student_ids = {
                node["student_id"]
                for node in existing_students
            }

            added_students = []
            already_enrolled_numbers = []
            not_found_numbers = []

            try:
                for registration_number in raw_registration_numbers:
                    student = get_student_by_registration_number(
                        registration_number
                    )

                    if not student:
                        not_found_numbers.append(
                            registration_number
                        )
                        continue

                    if (
                        student["student_id"]
                        in enrolled_student_ids
                    ):
                        already_enrolled_numbers.append(
                            registration_number
                        )
                        continue

                    enroll_student_to_subject(
                        student["student_id"],
                        selected_subject_id
                    )

                    enrolled_student_ids.add(
                        student["student_id"]
                    )
                    added_students.append(
                        student["name"]
                    )

                if added_students:
                    st.success(
                        f"Added {len(added_students)} student(s) successfully."
                    )
                    st.toast(
                        "Students added and attendance backfilled."
                    )

                if already_enrolled_numbers:
                    st.info(
                        "Already enrolled: "
                        + ", ".join(already_enrolled_numbers)
                    )

                if not_found_numbers:
                    st.warning(
                        "Not found: "
                        + ", ".join(not_found_numbers)
                    )

                if added_students:
                    st.rerun()

            except Exception as e:
                st.error(
                    f"Unable to add students: {str(e)}"
                )

    st.divider()

    students = get_subject_students(selected_subject_id)

    if not students:
        st.info("No students enrolled in this course yet.")
        return

    attendance_summary = get_subject_attendance_summary(
        selected_subject_id
    )

    st.subheader("Enrolled Students")
    st.caption(f"Total students: {len(students)}")

    for node in students:

        student = node.get("students")

        if not student:
            continue

        stats = attendance_summary.get(
            student["student_id"],
            {"total": 0, "present": 0}
        )

        total_classes = stats["total"]
        present_count = stats["present"]

        attendance_percent = (
            (present_count / total_classes) * 100
            if total_classes
            else 0
        )

        col1, col2 = st.columns(
            [4, 2],
            vertical_alignment="center"
        )

        with col1:
            st.write(f"**{student['name']}**")
            st.caption(
                f"Reg. No.: {student['registration_number']}"
            )

        with col2:
            st.metric(
                "Attendance %",
                f"{attendance_percent:.1f}%"
            )

        st.divider()


def teacher_tab_attendance_records():

    st.header("Attendance Records")

    teacher_id = st.session_state.teacher_data["teacher_id"]

    subjects = get_teacher_subjects(teacher_id)

    if not subjects:
        st.info("No courses found.")
        return

    # ---------------------------------------------------------
    # Select Course
    # ---------------------------------------------------------

    subject_options = {
        f"{s['name']} - {s['subject_code']} - Slot {s['slot']}":
            s["subject_id"]
        for s in subjects
    }

    selected_subject_label = st.selectbox(
        "Select Course",
        list(subject_options.keys()),
        key="attendance_record_subject"
    )

    selected_subject_id = subject_options[
        selected_subject_label
    ]

    # ---------------------------------------------------------
    # Select Date
    # ---------------------------------------------------------

    selected_date = st.date_input(
        "Select Attendance Date",
        value=datetime.now().date(),
        format="DD/MM/YYYY",
        key="attendance_record_date"
    )

    # ---------------------------------------------------------
    # Load attendance
    # ---------------------------------------------------------

    if st.button(
        "Load Attendance",
        type="primary",
        icon=":material/search:",
        width="stretch"
    ):

        attendance_records = get_attendance_for_date(
            selected_subject_id,
            selected_date.isoformat()
        )

        if not attendance_records:

            st.warning(
                "No attendance has been recorded for this "
                "course on the selected date."
            )

            st.session_state.edit_attendance_records = None

        else:

            st.session_state.edit_attendance_records = (
                attendance_records
            )

    # ---------------------------------------------------------
    # Display editable attendance
    # ---------------------------------------------------------

    attendance_records = st.session_state.get(
        "edit_attendance_records"
    )

    if not attendance_records:
        return

    st.divider()

    st.subheader(
        f"Attendance — {selected_date.strftime('%d/%m/%Y')}"
    )

    st.caption(
        f"{selected_subject_label}"
    )

    search_query = st.text_input(
        "Search students",
        placeholder="Search by name or registration number",
        key="attendance_record_search"
    ).strip().lower()

    edited_records = []
    visible_records = []

    # ---------------------------------------------------------
    # Student attendance controls
    # ---------------------------------------------------------

    for record in attendance_records:

        student_id = record["student_id"]

        # Get student details
        student_response = (
            supabase
            .table("students")
            .select(
                "student_id, name, registration_number"
            )
            .eq("student_id", student_id)
            .single()
            .execute()
        )

        student = student_response.data

        if not student:
            continue

        student_name = str(student.get("name", ""))
        registration_number = str(
            student.get("registration_number", "")
        )

        if search_query and (
            search_query not in student_name.lower()
            and search_query not in registration_number.lower()
        ):
            continue

        visible_records.append(record)

        col1, col2, col3 = st.columns(
            [3, 2, 2],
            vertical_alignment="center"
        )

        with col1:

            st.write(
                f"**{student_name}**"
            )

            st.caption(
                f"Reg. No.: "
                f"{registration_number}"
            )

        with col2:

            current_status = bool(
                record.get("is_present", False)
            )

            status = st.selectbox(
                "Status",
                ["Present", "Absent"],
                index=0 if current_status else 1,
                key=f"attendance_status_{record['id']}"
            )

        with col3:

            st.caption(
                "Current"
            )

            if current_status:
                st.success("Present")
            else:
                st.error("Absent")

        edited_records.append({
            "id": record["id"],
            "is_present": status == "Present"
        })

        st.divider()

    if not visible_records:
        st.info(
            "No students match your search."
        )
        return

    # ---------------------------------------------------------
    # Save changes
    # ---------------------------------------------------------

    if st.button(
        "Save Attendance Changes",
        type="primary",
        icon=":material/save:",
        width="stretch"
    ):

        try:

            for record in edited_records:

                update_attendance(
                    record["id"],
                    record["is_present"]
                )

            st.success(
                "Attendance updated successfully! ✅"
            )

            st.toast(
                "Attendance changes saved!"
            )

            # Clear loaded records so fresh data
            # is fetched next time
            st.session_state.edit_attendance_records = None

            st.rerun()

        except Exception as e:

            st.error(
                f"Failed to update attendance: {str(e)}"
            )

def login_teacher(employee_id, password):

    if not employee_id or not password:
        return False

    teacher = teacher_login(
        employee_id.strip(),
        password
    )

    if teacher:

        st.session_state.user_role = 'teacher'
        st.session_state.teacher_data = teacher
        st.session_state.is_logged_in = True

        return True

    return False


def teacher_screen_login():

    c1, c2 = st.columns(
        2,
        vertical_alignment='center',
        gap='xxlarge'
    )

    with c1:
        header_dashboard()

    with c2:

        if st.button(
            "Go back to Home",
            type='secondary',
            key='teacher_login_back',
            shortcut="control+backspace"
        ):

            st.session_state['login_type'] = None
            st.rerun()

    st.header(
        'Login using Employee ID',
        text_alignment='center'
    )

    st.space()
    st.space()

    employee_id = st.text_input(
        "Employee ID",
        placeholder='EMP1025'
    )

    teacher_pass = st.text_input(
        "Password",
        type='password',
        placeholder="Enter password"
    )

    st.divider()

    btnc1, btnc2 = st.columns(2)

    with btnc1:

        if st.button(
            'Login',
            icon=':material/login:',
            shortcut='control+enter',
            width='stretch'
        ):

            if login_teacher(
                employee_id,
                teacher_pass
            ):

                st.toast(
                    "Welcome back!",
                    icon="👋"
                )

                import time
                time.sleep(1)

                st.rerun()

            else:

                st.error(
                    "Invalid Employee ID or password."
                )

    with btnc2:

        if st.button(
            'Register Instead',
            type="primary",
            icon=':material/person_add:',
            width='stretch'
        ):

            st.session_state.teacher_login_type = 'register'
            st.rerun()

    footer_dashboard()


def register_teacher(
    employee_id,
    teacher_name,
    teacher_pass,
    teacher_pass_confirm
):

    if (
        not employee_id
        or not teacher_name
        or not teacher_pass
    ):

        return False, "All fields are required!"

    if check_teacher_exists(
        employee_id.strip()
    ):

        return False, "Employee ID already registered!"

    if teacher_pass != teacher_pass_confirm:

        return False, "Passwords do not match!"

    try:

        create_teacher(
            employee_id.strip(),
            teacher_pass,
            teacher_name.strip()
        )

        return True, "Successfully created! Login now."

    except Exception as e:

        print("Teacher registration error:", e)

        return False, "Unexpected error while creating account."


def teacher_screen_register():

    c1, c2 = st.columns(
        2,
        vertical_alignment='center',
        gap='xxlarge'
    )

    with c1:
        header_dashboard()

    with c2:

        if st.button(
            "Go back to Home",
            type='secondary',
            key='teacher_register_back',
            shortcut="control+backspace"
        ):

            st.session_state['login_type'] = None
            st.rerun()

    st.header(
        'Register your teacher profile',
        text_alignment='center'
    )

    st.space()
    st.space()

    employee_id = st.text_input(
        "Employee ID",
        placeholder='Enter Employee ID'
    )

    teacher_name = st.text_input(
        "Full Name",
        placeholder='Enter your Full Name'
    )

    teacher_pass = st.text_input(
        "Password",
        type='password',
        placeholder="Create a password"
    )

    teacher_pass_confirm = st.text_input(
        "Confirm Password",
        type='password',
        placeholder="Re-enter your password"
    )

    st.divider()

    btnc1, btnc2 = st.columns(2)

    with btnc1:

        if st.button(
            'Register Now',
            icon=':material/person_add:',
            shortcut='control+enter',
            width='stretch'
        ):

            success, message = register_teacher(
                employee_id,
                teacher_name,
                teacher_pass,
                teacher_pass_confirm
            )

            if success:

                st.success(message)

                import time
                time.sleep(2)

                st.session_state.teacher_login_type = "login"

                st.rerun()

            else:

                st.error(message)

    with btnc2:

        if st.button(
            'Login Instead',
            type="primary",
            icon=':material/login:',
            width='stretch'
        ):

            st.session_state.teacher_login_type = 'login'
            st.rerun()

    footer_dashboard()
