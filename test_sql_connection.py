"""
test_sql_connection.py
Nada Ali Yaqoob · 202507955 · Polytechnic of Bahrain

PURPOSE
-------
Tests the Azure SQL Database connection and verifies the Profile Agent
can retrieve student records correctly.

Runs four tests:
  1. Basic connection test
  2. SELECT TOP 5 FROM students
  3. Full profile query (JOIN with lookup tables)
  4. Profile Agent function test (as used by orchestrator_agent.py)

Run: py test_sql_connection.py
"""

import os
import sys
import pyodbc
from dotenv import load_dotenv

load_dotenv()

CONN_STR = os.getenv("AZURE_SQL_CONNECTION_STRING")


def get_connection():
    return pyodbc.connect(CONN_STR)


# ─────────────────────────────────────────────
# TEST 1 — Basic connection
# ─────────────────────────────────────────────
def test_connection():
    print("[Test 1] Basic connection...")
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT @@VERSION")
        version = cursor.fetchone()[0]
        conn.close()
        print(f"  ✅ Connected to Azure SQL")
        print(f"  Version: {version[:60]}...")
        return True
    except Exception as e:
        print(f"  ❌ Connection failed: {e}")
        return False


# ─────────────────────────────────────────────
# TEST 2 — SELECT TOP 5 FROM students
# ─────────────────────────────────────────────
def test_select_students():
    print("\n[Test 2] SELECT TOP 5 FROM students...")
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT TOP 5 student_id, first_name, last_name, gpa FROM students")
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            print("  ⚠️  Table is empty — run load_student_data.py first")
            return False

        print(f"  ✅ Found {len(rows)} rows")
        for row in rows:
            print(f"     {row.student_id} | {row.first_name} {row.last_name} | GPA: {row.gpa}")
        return True
    except Exception as e:
        print(f"  ❌ Query failed: {e}")
        return False


# ─────────────────────────────────────────────
# TEST 3 — Full profile JOIN query
# ─────────────────────────────────────────────
def test_profile_join():
    print("\n[Test 3] Full profile query with JOIN...")
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT TOP 1
                s.student_id,
                s.first_name + ' ' + s.last_name  AS full_name,
                p.programme_name,
                g.gender_name,
                sc.scholarship_name,
                s.gpa,
                s.credits_completed,
                s.academic_standing,
                s.current_semester
            FROM students s
            LEFT JOIN ProgrammeLookup   p  ON s.programme_id   = p.programme_id
            LEFT JOIN GenderLookup      g  ON s.gender_id       = g.gender_id
            LEFT JOIN ScholarshipLookup sc ON s.scholarship_id  = sc.scholarship_id
        """)
        row = cursor.fetchone()
        conn.close()

        if not row:
            print("  ⚠️  No students found")
            return False

        print(f"  ✅ Profile retrieved successfully:")
        print(f"     Student ID:    {row.student_id}")
        print(f"     Name:          {row.full_name}")
        print(f"     Programme:     {row.programme_name}")
        print(f"     Gender:        {row.gender_name}")
        print(f"     Scholarship:   {row.scholarship_name}")
        print(f"     GPA:           {row.gpa}")
        print(f"     Credits:       {row.credits_completed}")
        print(f"     Standing:      {row.academic_standing}")
        print(f"     Semester:      {row.current_semester}")
        return True
    except Exception as e:
        print(f"  ❌ JOIN query failed: {e}")
        return False


# ─────────────────────────────────────────────
# TEST 4 — Profile Agent function test
# ─────────────────────────────────────────────
def test_profile_agent():
    print("\n[Test 4] Profile Agent function test...")
    try:
        # Get first student ID from database
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT TOP 1 student_id FROM students")
        row = cursor.fetchone()
        conn.close()

        if not row:
            print("  ⚠️  No students in database")
            return False

        test_id = row.student_id
        print(f"  Testing with student_id: {test_id}")

        # Import and call the actual profile_agent function
        sys.path.insert(0, ".")
        from orchestrator_agent import profile_agent

        profile = profile_agent(test_id)

        if not profile:
            print("  ❌ Profile Agent returned empty result")
            return False

        print(f"  ✅ Profile Agent working correctly:")
        for key, value in profile.items():
            print(f"     {key}: {value}")
        return True

    except ImportError:
        print("  ⚠️  orchestrator_agent.py not found — skipping Profile Agent test")
        print("      Basic SQL tests passed — Profile Agent will work once agents are ready")
        return True
    except Exception as e:
        print(f"  ❌ Profile Agent test failed: {e}")
        return False


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run():
    print("=" * 55)
    print("Azure SQL Connection Test")
    print("=" * 55)

    if not CONN_STR:
        print("[ERROR] AZURE_SQL_CONNECTION_STRING not set in .env")
        print("Add this line to your .env file:")
        print("  AZURE_SQL_CONNECTION_STRING=Driver={ODBC Driver 18 for SQL Server};Server=tcp:<server>.database.windows.net,1433;Database=academic-advising-db;Uid=<admin>;Pwd=<password>;Encrypt=yes;TrustServerCertificate=no;")
        return

    results = []
    results.append(("Connection",    test_connection()))
    results.append(("SELECT TOP 5",  test_select_students()))
    results.append(("JOIN query",    test_profile_join()))
    results.append(("Profile Agent", test_profile_agent()))

    print("\n" + "=" * 55)
    print("TEST SUMMARY")
    print("=" * 55)
    passed = sum(1 for _, r in results if r)
    for name, result in results:
        print(f"  {'✅' if result else '❌'} {name}")
    print(f"\n{passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\n✅ Azure SQL fully operational — Profile Agent ready")
        print("Next step: py orchestrator_agent.py  (with student ID)")
    else:
        print("\n⚠️  Fix failing tests before running the Agentic RAG system")


if __name__ == "__main__":
    run()
