"""
Master Test Runner for ECG Diagnostic Platform
Runs API endpoints, multi-modal file ingestion, and browser workflow validation.
"""

import os
import sys
import subprocess

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)

test_scripts = [
    ("REST API Endpoints", os.path.join(TESTS_DIR, "test_api_endpoints.py")),
    ("Multi-Modal File Ingestion", os.path.join(TESTS_DIR, "test_multimodal_upload.py")),
    ("End-to-End Browser Workflow", os.path.join(TESTS_DIR, "test_browser_workflow.py"))
]

def main():
    print("=" * 80)
    print("  ECG MEDICAL AI PLATFORM — MASTER VERIFICATION TEST RUNNER")
    print("=" * 80)
    
    python_exe = sys.executable
    all_passed = True
    results = []

    for name, script_path in test_scripts:
        print(f"\n>>> Running: {name} ...")
        proc = subprocess.run([python_exe, script_path], cwd=PROJECT_ROOT, capture_output=True, text=True)
        if proc.returncode == 0:
            print(proc.stdout)
            results.append((name, "PASSED"))
        else:
            print(proc.stdout)
            print(proc.stderr)
            results.append((name, "FAILED"))
            all_passed = False

    print("\n" + "=" * 80)
    print("  TEST SUITE EXECUTION SUMMARY")
    print("=" * 80)
    for name, status in results:
        print(f"  * {name:<35}: [{status}]")
    print("=" * 80)

    if not all_passed:
        sys.exit(1)

if __name__ == '__main__':
    main()
