"""
load_student_data.py
Nada Ali Yaqoob · 202507955 · Polytechnic of Bahrain

PURPOSE
-------
Loads the synthetic student dataset CSV into Azure SQL Database.
Handles both text and numeric column formats automatically.

Place student_dataset.csv in the same folder before running.

Run: py load_student_data.py
"""

import os
import pyodbc
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

load_dotenv()

CONN_STR = os.getenv("AZURE_SQL_CONNECTION_STRING")
CSV_FILE = "student_dataset.csv"


def get_engine():
    connection_url = URL.create(
        "mssql+pyodbc",
        query={"odbc_connect": CONN_STR}
    )
    return create_engine(connection_url, fast_executemany=True)


def load_csv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, encoding="utf-8")
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Standardise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    print(f"  Columns found: {list(df.columns)}")

    # ── Generate student IDs ───────────────────────────────────────
    if "student_id" not in df.columns:
        df["student_id"] = [f"S{10000 + i}" for i in range(len(df))]

    # ── Gender mapping ─────────────────────────────────────────────
    gender_text_map = {"male": 1, "m": 1, "female": 2, "f": 2, "other": 3}
    gender_col = next((c for c in df.columns if "gender" in c), None)
    if gender_col:
        if df[gender_col].dtype == object:
            df["gender_id"] = (
                df[gender_col].astype(str).str.strip().str.lower()
                .map(gender_text_map).fillna(3).astype(int)
            )
        else:
            # Numeric: 1=Male, 0=Female
            df["gender_id"] = df[gender_col].map({1: 1, 0: 2}).fillna(3).astype(int)
    else:
        df["gender_id"] = 3

    # ── Scholarship mapping ────────────────────────────────────────
    schol_col = next((c for c in df.columns if "scholarship" in c), None)
    if schol_col:
        if df[schol_col].dtype == object:
            schol_map = {"full": 1, "full scholarship": 1,
                         "partial": 2, "partial scholarship": 2,
                         "none": 3, "no scholarship": 3, "0": 3, "1": 1}
            df["scholarship_id"] = (
                df[schol_col].astype(str).str.strip().str.lower()
                .map(schol_map).fillna(3).astype(int)
            )
        else:
            df["scholarship_id"] = df[schol_col].map({1: 1, 0: 3}).fillna(3).astype(int)
    else:
        df["scholarship_id"] = 3

    # ── Programme mapping ──────────────────────────────────────────
    prog_col = next((c for c in df.columns if "course" in c or "programme" in c), None)
    if prog_col and df[prog_col].dtype != object:
        # Numeric course codes — map to programme IDs
        unique_courses = df[prog_col].dropna().unique()
        course_to_prog = {code: min(int(i) % 13 + 1, 13)
                          for i, code in enumerate(sorted(unique_courses))}
        df["programme_id"] = df[prog_col].map(course_to_prog).fillna(4).astype(int)
    else:
        df["programme_id"] = 4

    # ── GPA from grade columns ─────────────────────────────────────
    grade_cols = [c for c in df.columns if "grade" in c]
    if grade_cols:
        df["gpa"] = df[grade_cols].apply(
            pd.to_numeric, errors="coerce"
        ).mean(axis=1).fillna(2.50)
        # Normalise to 0-4 scale if values > 4
        if df["gpa"].max() > 4:
            df["gpa"] = (df["gpa"] / df["gpa"].max() * 4).round(2)
        df["gpa"] = df["gpa"].clip(0, 4.0).round(2)
    else:
        df["gpa"] = 2.50

    # ── Credits completed ──────────────────────────────────────────
    approved_cols = [c for c in df.columns if "approved" in c]
    if approved_cols:
        df["credits_completed"] = df[approved_cols].apply(
            pd.to_numeric, errors="coerce"
        ).sum(axis=1).fillna(0).astype(int) * 5
    else:
        df["credits_completed"] = 0

    # ── Academic standing from Target column ───────────────────────
    target_col = next((c for c in df.columns if c == "target"), None)
    if target_col:
        standing_map = {"graduate": "Good Standing",
                        "enrolled": "Good Standing",
                        "dropout": "Academic Warning"}
        if df[target_col].dtype == object:
            df["academic_standing"] = (
                df[target_col].str.lower().map(standing_map).fillna("Good Standing")
            )
        else:
            df["academic_standing"] = df[target_col].map(
                {1: "Good Standing", 0: "Academic Warning"}
            ).fillna("Good Standing")
    else:
        df["academic_standing"] = "Good Standing"

    # ── First/last name from index ─────────────────────────────────
    if "first_name" not in df.columns:
        df["first_name"] = "Student"
    if "last_name" not in df.columns:
        df["last_name"] = df["student_id"].astype(str)

    # ── Other required fields ──────────────────────────────────────
    if "enrolment_year" not in df.columns:
        df["enrolment_year"] = 2024
    if "current_semester" not in df.columns:
        df["current_semester"] = 2

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["student_id"])
    if len(df) < before:
        print(f"  Removed {before - len(df)} duplicate student IDs")

    print(f"  Preprocessing complete — {len(df)} students ready")
    return df


def load_to_sql(df: pd.DataFrame, engine):
    sql_columns = [
        "student_id", "first_name", "last_name",
        "gender_id", "programme_id", "scholarship_id",
        "gpa", "credits_completed", "academic_standing",
        "enrolment_year", "current_semester"
    ]
    available = [c for c in sql_columns if c in df.columns]
    df_to_load = df[available]

    df_to_load.to_sql(
        name="students",
        con=engine,
        if_exists="append",
        index=False,
        chunksize=500
    )
    print(f"  ✅ {len(df_to_load)} student records loaded into Azure SQL")


def verify_load(engine):
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT TOP 5
                s.student_id,
                s.first_name + ' ' + s.last_name AS name,
                p.programme_name,
                s.gpa,
                s.credits_completed,
                s.academic_standing
            FROM students s
            LEFT JOIN ProgrammeLookup p ON s.programme_id = p.programme_id
            ORDER BY s.student_id
        """))
        rows = result.fetchall()

    print("\n[Verification] First 5 students:")
    print(f"  {'ID':<12} {'Name':<20} {'Programme':<35} {'GPA':<6} {'Credits'}")
    print("  " + "─" * 80)
    for row in rows:
        print(f"  {row[0]:<12} {row[1]:<20} {str(row[2]):<35} {row[3]:<6} {row[4]}")


def run():
    print("=" * 55)
    print("Load Synthetic Student Dataset → Azure SQL")
    print("=" * 55)

    if not CONN_STR:
        print("[ERROR] AZURE_SQL_CONNECTION_STRING not found in .env")
        return

    if not os.path.exists(CSV_FILE):
        print(f"[ERROR] {CSV_FILE} not found in project folder")
        return

    print(f"\n[Step 1] Loading CSV: {CSV_FILE}")
    df = load_csv(CSV_FILE)

    print("\n[Step 2] Preprocessing...")
    df = preprocess(df)

    print("\n[Step 3] Connecting to Azure SQL...")
    engine = get_engine()
    print("  ✅ Connected")

    print("\n[Step 4] Loading into students table...")
    load_to_sql(df, engine)

    print("\n[Step 5] Verifying...")
    verify_load(engine)

    print("\n[Done] Run: py test_sql_connection.py")


if __name__ == "__main__":
    run()
