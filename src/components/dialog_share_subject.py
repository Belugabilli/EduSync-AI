import streamlit as st


@st.dialog("Share Course")
def share_subject_dialog(
    subject_name,
    subject_code,
    slot,
    joining_code
):

    st.write(
        f"Share **{subject_name}** with your students."
    )

    st.write(
        f"**Course Code:** {subject_code}"
    )

    st.write(
        f"**Slot:** {slot}"
    )

    st.subheader("Joining Code")

    st.code(
        joining_code,
        language="text"
    )

    st.caption(
        "Give this Joining Code to your students. "
        "They can use it to enroll in this course."
    )