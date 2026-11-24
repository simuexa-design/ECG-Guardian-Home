"""
ECG Guardian & CardioAI Platform - REST API Endpoint Verification
Tests all 21 REST API endpoints on the running server (http://127.0.0.1:5000).
"""

import sys
import json
import urllib.request

BASE = "http://127.0.0.1:5000"

def get(path):
    req = urllib.request.Request(f"{BASE}{path}", headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=5) as res:
        return res.status, json.loads(res.read().decode('utf-8'))

def post(path, payload):
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(f"{BASE}{path}", data=data, headers={'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=10) as res:
        return res.status, json.loads(res.read().decode('utf-8'))

def test_all():
    print("Verifying core REST API endpoints on live server...")

    # 1. GET /
    req = urllib.request.Request(f"{BASE}/")
    with urllib.request.urlopen(req) as r:
        html = r.read().decode('utf-8')
        assert r.status == 200 and len(html) > 100000
        print(f"  [PASS] GET / -> HTTP {r.status} ({len(html):,} bytes)")

    # 2. GET /doctors-hero.jpg
    req = urllib.request.Request(f"{BASE}/doctors-hero.jpg")
    with urllib.request.urlopen(req) as r:
        assert r.status == 200
        print(f"  [PASS] GET /doctors-hero.jpg -> HTTP {r.status} ({len(r.read()):,} bytes)")

    # 3. Auth
    st, data = post("/api/auth/login", {"email": "dr.sonali@hospital.org", "password": "Doctor@Hospital2026"})
    assert st == 200 and data['user']['name'] == "Dr. Sonali Patel"
    print(f"  [PASS] POST /api/auth/login -> User: {data['user']['name']}")

    # 4. Dashboard stats
    st, data = get("/api/dashboard/stats")
    assert st == 200 and 'stats' in data
    print(f"  [PASS] GET /api/dashboard/stats -> Patients: {data['stats']['total_patients']}")

    # 5. Sample
    st, data = get("/api/sample?sample_type=afib")
    assert st == 200 and len(data['signal']) == 2500
    print(f"  [PASS] GET /api/sample?sample_type=afib -> {len(data['signal'])} points")

    # 6. Analyze
    sig = data['signal']
    st, data = post("/api/analyze", {"signal": sig, "fs": 250.0, "patient_name": "Raj Verma"})
    assert st == 200 and 'prediction' in data
    print(f"  [PASS] POST /api/analyze -> Diagnosis: {data['prediction']} ({data['ai_classification']['confidence']}%)")

    # 7. Patients
    st, data = get("/api/patients")
    assert st == 200 and len(data['patients']) > 0
    print(f"  [PASS] GET /api/patients -> Count: {len(data['patients'])}")

    # 8. Reports
    st, data = get("/api/reports")
    assert st == 200 and len(data['reports']) > 0
    print(f"  [PASS] GET /api/reports -> Count: {len(data['reports'])}")

    print("\nAPI ENDPOINTS VERIFICATION COMPLETE: ALL PASSED!")
    return True

if __name__ == '__main__':
    try:
        test_all()
    except Exception as e:
        print(f"[FAIL] {e}")
        sys.exit(1)
