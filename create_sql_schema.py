"""
create_sql_schema.py
Nada Ali Yaqoob · 202507955 · Polytechnic of Bahrain

PURPOSE
-------
Creates all required tables in Azure SQL Database for the
Agentic RAG academic advising system.

This script must be run ONCE before loading any student data.
It creates:
  - Lookup tables (GenderLookup, ProgrammeLookup, ScholarshipLookup)
  - Core tables (students, courses, enrollments, grades, degree_plans)

Run: py create_sql_schema.py
"""

import os
import pyodbc
from dotenv import load_dotenv

load_dotenv()

CONN_STR = os.getenv("AZURE_SQL_CONNECTION_STRING")


def get_connection():
    return pyodbc.connect(CONN_STR)


def create_all_tables(cursor):

    # ── Lookup: Gender ─────────────────────────────────────────────
    cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='GenderLookup' AND xtype='U')
        CREATE TABLE GenderLookup (
            gender_id   INT PRIMARY KEY,
            gender_name NVARCHAR(20) NOT NULL
        )
    """)

    # ── Lookup: Programme ──────────────────────────────────────────
    cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='ProgrammeLookup' AND xtype='U')
        CREATE TABLE ProgrammeLookup (
            programme_id   INT PRIMARY KEY,
            programme_name NVARCHAR(200) NOT NULL,
            programme_type NVARCHAR(50),
            faculty        NVARCHAR(100)
        )
    """)

    # ── Lookup: Scholarship ────────────────────────────────────────
    cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='ScholarshipLookup' AND xtype='U')
        CREATE TABLE ScholarshipLookup (
            scholarship_id   INT PRIMARY KEY,
            scholarship_name NVARCHAR(100) NOT NULL
        )
    """)

    # ── Core: Students ─────────────────────────────────────────────
    cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='students' AND xtype='U')
        CREATE TABLE students (
            student_id          NVARCHAR(20)  PRIMARY KEY,
            first_name          NVARCHAR(50)  NOT NULL,
            last_name           NVARCHAR(50)  NOT NULL,
            gender_id           INT           REFERENCES GenderLookup(gender_id),
            programme_id        INT           REFERENCES ProgrammeLookup(programme_id),
            scholarship_id      INT           REFERENCES ScholarshipLookup(scholarship_id),
            gpa                 DECIMAL(3,2),
            credits_completed   INT           DEFAULT 0,
            academic_standing   NVARCHAR(30),
            enrolment_year      INT,
            current_semester    INT
        )
    """)

    # ── Core: Courses ──────────────────────────────────────────────
    cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='courses' AND xtype='U')
        CREATE TABLE courses (
            course_code    NVARCHAR(20)  PRIMARY KEY,
            course_title   NVARCHAR(200) NOT NULL,
            credits        INT           NOT NULL,
            level          INT,
            faculty        NVARCHAR(100)
        )
    """)

    # ── Core: Enrollments ──────────────────────────────────────────
    cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='enrollments' AND xtype='U')
        CREATE TABLE enrollments (
            enrollment_id  INT           PRIMARY KEY IDENTITY(1,1),
            student_id     NVARCHAR(20)  REFERENCES students(student_id),
            course_code    NVARCHAR(20)  REFERENCES courses(course_code),
            semester       INT,
            year           INT,
            status         NVARCHAR(20)  DEFAULT 'Enrolled'
        )
    """)

    # ── Core: Grades ───────────────────────────────────────────────
    cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='grades' AND xtype='U')
        CREATE TABLE grades (
            grade_id      INT           PRIMARY KEY IDENTITY(1,1),
            student_id    NVARCHAR(20)  REFERENCES students(student_id),
            course_code   NVARCHAR(20)  REFERENCES courses(course_code),
            grade         NVARCHAR(5),
            grade_points  DECIMAL(3,2),
            semester      INT,
            year          INT
        )
    """)

    # ── Core: Degree Plans ─────────────────────────────────────────
    cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='degree_plans' AND xtype='U')
        CREATE TABLE degree_plans (
            plan_id            INT           PRIMARY KEY IDENTITY(1,1),
            programme_id       INT           REFERENCES ProgrammeLookup(programme_id),
            course_code        NVARCHAR(20)  REFERENCES courses(course_code),
            semester_sequence  INT,
            is_compulsory      BIT           DEFAULT 1
        )
    """)

    print("  ✅ All tables created (or already exist)")


def seed_lookup_tables(cursor):
    """Insert standard lookup values."""

    # Gender
    cursor.execute("IF NOT EXISTS (SELECT 1 FROM GenderLookup) INSERT INTO GenderLookup VALUES (1,'Male'),(2,'Female'),(3,'Other')")

    # Scholarship types
    cursor.execute("""
        IF NOT EXISTS (SELECT 1 FROM ScholarshipLookup)
        INSERT INTO ScholarshipLookup VALUES
        (1,'Full Scholarship'),(2,'Partial Scholarship'),(3,'No Scholarship')
    """)

    # Sample programmes matching Polytechnic of Bahrain offerings
    cursor.execute("""
        IF NOT EXISTS (SELECT 1 FROM ProgrammeLookup)
        INSERT INTO ProgrammeLookup VALUES
        (1,  'Bachelor of Business: Marketing',              'Bachelor', 'School of Business'),
        (2,  'Bachelor of Business: Accounting',             'Bachelor', 'School of Business'),
        (3,  'Bachelor of Business: Human Resource Management','Bachelor','School of Business'),
        (4,  'Bachelor of ICT: Management Information Systems','Bachelor','Faculty of ICT'),
        (5,  'Bachelor of ICT: Networking',                  'Bachelor', 'Faculty of ICT'),
        (6,  'Bachelor of ICT: Programming',                 'Bachelor', 'Faculty of ICT'),
        (7,  'Bachelor of Science in Artificial Intelligence and Robotics Software','Bachelor','Faculty of ICT'),
        (8,  'Bachelor of Science in Cloud Computing and Information Systems','Bachelor','Faculty of ICT'),
        (9,  'Bachelor of Science in Data Analytics',        'Bachelor', 'Faculty of ICT'),
        (10, 'Diploma in Cloud Computing',                   'Diploma',  'Faculty of ICT'),
        (11, 'Diploma in Web Development',                   'Diploma',  'Faculty of ICT'),
        (12, 'Master of Science in Artificial Intelligence', 'Masters',  'Faculty of ICT'),
        (13, 'Master of Science in Supply Chain Management', 'Masters',  'School of Business')
    """)

    print("  ✅ Lookup tables seeded")


def run():
    print("=" * 50)
    print("Azure SQL Schema Creation")
    print("Database: academic-advising-db")
    print("=" * 50)

    if not CONN_STR:
        print("[ERROR] AZURE_SQL_CONNECTION_STRING not found in .env")
        return

    conn   = get_connection()
    cursor = conn.cursor()

    print("\n[Step 1] Creating tables...")
    create_all_tables(cursor)

    print("[Step 2] Seeding lookup tables...")
    seed_lookup_tables(cursor)

    conn.commit()
    conn.close()

    print("\n[Done] Schema ready — run load_student_data.py next")


if __name__ == "__main__":
    run()
