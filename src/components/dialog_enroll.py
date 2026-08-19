import streamlit as st

from src.database.db import enroll_student_to_subject

from src.database.config import supabase


@st.dialog("Enroll in Course")
def enroll_dialog():

    st.write(
        "Enter the Joining Code provided by your teacher."
    )

    joining_code = st.text_input(
        "Joining Code",
        placeholder="EMP1025-CSE2001-A1"
    )

    if st.button(
        "Join Course",
        type="primary",
        width="stretch",
        icon=":material/group_add:"
    ):

        joining_code = joining_code.strip().upper()

        if not joining_code:

            st.warning(
                "Please enter the Joining Code."
            )

            return

        # Find course using Joining Code
        response = (
            supabase
            .table("subjects")
            .select("*")
            .eq("joining_code", joining_code)
            .execute()
        )

        if not response.data:

            st.error(
                "Invalid Joining Code."
            )

            return

        subject = response.data[0]

        student_id = (
            st.session_state
            .student_data["student_id"]
        )

        # Check whether student is already enrolled
        existing = (
            supabase
            .table("subject_students")
            .select("*")
            .eq("student_id", student_id)
            .eq("subject_id", subject["subject_id"])
            .execute()
        )

        if existing.data:

            st.warning(
                "You are already enrolled in this course."
            )

            return

        try:

            enroll_student_to_subject(
                student_id,
                subject["subject_id"]
            )

            st.success(
                f"Successfully enrolled in "
                f"{subject['name']}!"
            )

            st.toast(
                "Course joined successfully! 🎉"
            )

            st.rerun()

        except Exception as e:

            st.error(
                f"Unable to enroll: {str(e)}"
            )