from src.database.config import supabase
import bcrypt
from datetime import datetime



def hash_pass(pwd):
    return bcrypt.hashpw(pwd.encode(), bcrypt.gensalt()).decode()

def check_pass(pwd, hashed):
    return bcrypt.checkpw(pwd.encode(), hashed.encode())


def check_teacher_exists(employee_id):
    response = (
        supabase
        .table("teachers")
        .select("teacher_id")
        .eq("employee_id", employee_id)
        .execute()
    )

    return len(response.data) > 0


def create_teacher(employee_id, password, name):

    data = {
        "employee_id": employee_id,
        "password": hash_pass(password),
        "name": name
    }

    response = (
        supabase
        .table("teachers")
        .insert(data)
        .execute()
    )

    return response.data


def teacher_login(employee_id, password):

    response = (
        supabase
        .table("teachers")
        .select("*")
        .eq("employee_id", employee_id)
        .execute()
    )

    if response.data:

        teacher = response.data[0]

        if check_pass(password, teacher["password"]):
            return teacher

    return None


def get_all_students():
    response = supabase.table('students').select("*").execute()
    return response.data

def create_student(
    registration_number,
    new_name,
    password,
    face_embedding=None
):
    data = {
        "registration_number": registration_number,
        "name": new_name,
        "password": hash_pass(password),
        "face_embedding": face_embedding
    }

    response = supabase.table("students").insert(data).execute()
    return response.data

def student_login(registration_number, password):

    response = (
        supabase
        .table("students")
        .select("*")
        .eq("registration_number", registration_number)
        .execute()
    )

    if response.data:

        student = response.data[0]

        if student.get("password") and check_pass(
            password,
            student["password"]
        ):
            return student

    return None

def check_student_exists(registration_number):

    response = (
        supabase
        .table("students")
        .select("student_id")
        .eq("registration_number", registration_number)
        .execute()
    )

    return len(response.data) > 0

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

def create_subject(subject_code, name, slot, teacher_id):

    # Get the teacher's Employee ID
    teacher_response = (
        supabase
        .table("teachers")
        .select("employee_id")
        .eq("teacher_id", teacher_id)
        .single()
        .execute()
    )

    employee_id = teacher_response.data["employee_id"]

    # Generate Joining Code
    joining_code = f"{employee_id}-{subject_code}-{slot}".upper()

    data = {
        "subject_code": subject_code,
        "name": name,
        "slot": slot,
        "teacher_id": teacher_id,
        "joining_code": joining_code
    }

    response = (
        supabase
        .table("subjects")
        .insert(data)
        .execute()
    )

    return response.data

def get_teacher_subjects(teacher_id):

    response = (
        supabase
        .table('subjects')
        .select(
            "*, subject_students(count), "
            "attendance_logs(attendance_date)"
        )
        .eq("teacher_id", teacher_id)
        .execute()
    )

    subjects = response.data

    for sub in subjects:

        # Total enrolled students
        sub['total_students'] = (
            sub.get("subject_students", [{}])[0].get("count", 0)
            if sub.get("subject_students")
            else 0
        )

        # Total unique attendance dates
        attendance = sub.get("attendance_logs", [])

        unique_dates = set(
            log['attendance_date']
            for log in attendance
            if log.get('attendance_date')
        )

        sub['total_classes'] = len(unique_dates)

        # Remove unnecessary nested data
        sub.pop('subject_students', None)
        sub.pop('attendance_logs', None)

    return subjects


def  enroll_student_to_subject(student_id, subject_id):
    data = {'student_id': student_id, "subject_id": subject_id}
    response= supabase.table('subject_students').insert(data).execute()
    backfill_attendance_for_student_subject(
        student_id,
        subject_id
    )
    return response.data


def  unenroll_student_to_subject(student_id, subject_id):
    response= supabase.table('subject_students').delete().eq('student_id', student_id).eq('subject_id', subject_id).execute()
    return response.data


def get_student_by_registration_number(
    registration_number
):

    response = (
        supabase
        .table("students")
        .select("student_id, name, registration_number")
        .eq(
            "registration_number",
            registration_number.strip()
        )
        .execute()
    )

    if response.data:
        return response.data[0]

    return None


def get_subject_students(subject_id):
    response = (
        supabase
        .table("subject_students")
        .select(
            "student_id, students("
            "student_id, name, registration_number"
            ")"
        )
        .eq("subject_id", subject_id)
        .execute()
    )

    return response.data


def get_subject_attendance_summary(subject_id):
    response = (
        supabase
        .table("attendance_logs")
        .select("student_id, is_present")
        .eq("subject_id", subject_id)
        .execute()
    )

    stats_map = {}

    for log in response.data:
        student_id = log["student_id"]

        if student_id not in stats_map:
            stats_map[student_id] = {
                "total": 0,
                "present": 0
            }

        stats_map[student_id]["total"] += 1

        if log.get("is_present"):
            stats_map[student_id]["present"] += 1

    return stats_map


def backfill_attendance_for_student_subject(
    student_id,
    subject_id
):

    attendance_dates_response = (
        supabase
        .table("attendance_logs")
        .select("attendance_date")
        .eq("subject_id", subject_id)
        .execute()
    )

    attendance_dates = {
        log["attendance_date"]
        for log in attendance_dates_response.data
        if log.get("attendance_date")
    }

    attendance_dates.add(
        datetime.now().date().isoformat()
    )

    existing_logs_response = (
        supabase
        .table("attendance_logs")
        .select("attendance_date")
        .eq("subject_id", subject_id)
        .eq("student_id", student_id)
        .execute()
    )

    existing_dates = {
        log["attendance_date"]
        for log in existing_logs_response.data
        if log.get("attendance_date")
    }

    missing_logs = [
        {
            "student_id": student_id,
            "subject_id": subject_id,
            "attendance_date": attendance_date,
            "is_present": False
        }
        for attendance_date in sorted(attendance_dates)
        if attendance_date not in existing_dates
    ]

    if missing_logs:
        create_attendance(missing_logs)

    return missing_logs



def get_student_subjects(student_id):
    response = supabase.table('subject_students').select('*, subjects(*)').eq('student_id', student_id).execute()
    return response.data


def get_student_attendance(student_id):
    response = (
        supabase
        .table('attendance_logs')
        .select('*, subjects(*)')
        .eq('student_id', student_id)
        .order('attendance_date', desc=True)
        .execute()
    )

    return response.data


def create_attendance(logs):
    """
    Create attendance records.

    Each record must contain:
        student_id
        subject_id
        attendance_date
        is_present
    """

    response = (
        supabase
        .table('attendance_logs')
        .insert(logs)
        .execute()
    )

    return response.data


def get_attendance_for_teacher(teacher_id):
    response = (
        supabase
        .table('attendance_logs')
        .select("*, subjects!inner(*)")
        .eq('subjects.teacher_id', teacher_id)
        .order('attendance_date', desc=True)
        .execute()
    )

    return response.data

def get_attendance_for_date(subject_id, attendance_date):
    response = (
        supabase
        .table('attendance_logs')
        .select('*')
        .eq('subject_id', subject_id)
        .eq('attendance_date', attendance_date)
        .execute()
    )

    return response.data


def update_attendance(attendance_id, is_present):
    response = (
        supabase
        .table('attendance_logs')
        .update({
            'is_present': is_present
        })
        .eq('id', attendance_id)
        .execute()
    )

    return response.data
