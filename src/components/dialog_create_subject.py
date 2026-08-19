import streamlit as st
from src.database.db import create_subject


@st.dialog("Create New Course")
def create_subject_dialog(teacher_id):

    st.write("Enter the details of your new course")

    course_code = st.text_input(
        "Course Code",
        placeholder="CSE2001"
    )

    course_name = st.text_input(
        "Course Name",
        placeholder="Data Structures"
    )

    slot = st.text_input(
        "Slot",
        placeholder="A1"
    )

    if st.button(
        "Create Course Now",
        type="primary",
        width="stretch"
    ):

        if course_code and course_name and slot:

            try:

                response = create_subject(
                    course_code.strip().upper(),
                    course_name.strip(),
                    slot.strip().upper(),
                    teacher_id
                )

                if response:

                    joining_code = response[0]["joining_code"]

                    st.success(
                        f"Course created successfully!\n\n"
                        f"Joining Code: {joining_code}"
                    )

                    st.toast(
                        "Course Created Successfully!"
                    )

                    st.rerun()

            except Exception as e:

                if "joining_code" in str(e).lower():

                    st.error(
                        "This Joining Code already exists. "
                        "Please check the Course Code and Slot."
                    )

                else:

                    st.error(
                        f"Error creating course: {str(e)}"
                    )

        else:

            st.warning(
                "Please fill all the fields."
            )