import dlib
import numpy as np
import face_recognition_models
from sklearn.svm import SVC
import streamlit as st

from src.database.db import get_all_students


@st.cache_resource
def load_dlib_models():

    detector = dlib.get_frontal_face_detector()

    sp = dlib.shape_predictor(
        face_recognition_models.pose_predictor_model_location()
    )

    facerec = dlib.face_recognition_model_v1(
        face_recognition_models.face_recognition_model_location()
    )

    return detector, sp, facerec


def get_face_embeddings(image_np):

    detector, sp, facerec = load_dlib_models()

    faces = detector(image_np, 1)

    encodings = []

    for face in faces:

        shape = sp(image_np, face)

        face_descriptor = facerec.compute_face_descriptor(
            image_np,
            shape,
            1
        )

        encodings.append(
            np.array(face_descriptor)
        )

    return encodings


def get_enrollment_embeddings(frames):

    """
    Generate multiple face embeddings from enrollment frames.

    Each frame should contain one clear face.

    Returns:
        list of 128-dimensional face embeddings
    """

    all_embeddings = []

    for frame in frames:

        encodings = get_face_embeddings(frame)

        # Only accept frames containing exactly one face
        if len(encodings) == 1:

            all_embeddings.append(
                encodings[0].tolist()
            )

    return all_embeddings


@st.cache_resource
def get_trained_model():

    X = []
    y = []

    student_db = get_all_students()

    if not student_db:
        return None

    for student in student_db:

        embeddings = student.get("face_embedding")

        if not embeddings:
            continue

        # -------------------------------------------------
        # New format:
        # [
        #     [128 values],
        #     [128 values],
        #     ...
        # ]
        # -------------------------------------------------

        if (
            isinstance(embeddings, list)
            and embeddings
            and isinstance(embeddings[0], (list, tuple))
        ):

            for embedding in embeddings:

                if len(embedding) == 128:

                    X.append(
                        np.array(
                            embedding,
                            dtype=np.float64
                        )
                    )

                    y.append(
                        student.get("student_id")
                    )

        # -------------------------------------------------
        # Backward compatibility:
        # old format:
        # [128 values]
        # -------------------------------------------------

        elif (
            isinstance(embeddings, list)
            and len(embeddings) == 128
        ):

            X.append(
                np.array(
                    embeddings,
                    dtype=np.float64
                )
            )

            y.append(
                student.get("student_id")
            )

    if len(X) == 0:
        return None

    clf = SVC(
        kernel="linear",
        probability=True,
        class_weight="balanced"
    )

    try:

        clf.fit(X, y)

    except ValueError:

        return None

    return {
        "clf": clf,
        "X": X,
        "y": y
    }


def train_classifier():

    st.cache_resource.clear()

    model_data = get_trained_model()

    return bool(model_data)


def predict_attendance(class_image_np):

    encodings = get_face_embeddings(
        class_image_np
    )

    detected_student = {}

    model_data = get_trained_model()

    if not model_data:

        return (
            detected_student,
            [],
            len(encodings)
        )

    clf = model_data["clf"]

    X_train = model_data["X"]
    y_train = model_data["y"]

    all_students = sorted(
        list(set(y_train))
    )

    if not all_students:

        return (
            detected_student,
            [],
            len(encodings)
        )

    for encoding in encodings:

        # ---------------------------------------------
        # SVM candidate prediction
        # ---------------------------------------------

        if len(all_students) >= 2:

            predicted_id = int(
                clf.predict(
                    [encoding]
                )[0]
            )

        else:

            predicted_id = int(
                all_students[0]
            )

        # ---------------------------------------------
        # Compare against ALL embeddings belonging
        # to the predicted student.
        # ---------------------------------------------

        student_embeddings = [
            X_train[index]
            for index, student_id
            in enumerate(y_train)
            if student_id == predicted_id
        ]

        if not student_embeddings:

            continue

        distances = [
            np.linalg.norm(
                embedding - encoding
            )
            for embedding
            in student_embeddings
        ]

        best_match_score = min(
            distances
        )

        resemblance_threshold = 0.6

        if best_match_score <= resemblance_threshold:

            detected_student[
                predicted_id
            ] = True

    return (
        detected_student,
        all_students,
        len(encodings)
    )