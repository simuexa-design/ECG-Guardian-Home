import subprocess
import os
import time

edge_path = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
out_dir = r'D:\ecg\static\images\screenshots'
os.makedirs(out_dir, exist_ok=True)

views = [
    ('screen_01_landing.png', 'http://127.0.0.1:5000/'),
    ('screen_02_auth.png', 'http://127.0.0.1:5000/?view=auth'),
    ('screen_03_doctor_dashboard.png', 'http://127.0.0.1:5000/?view=doctorDashboard'),
    ('screen_04_ecg_scan.png', 'http://127.0.0.1:5000/?view=ecgScan'),
    ('screen_05_analysis.png', 'http://127.0.0.1:5000/?view=analysis'),
    ('screen_06_cds.png', 'http://127.0.0.1:5000/?view=cds'),
    ('screen_07_doctor_report.png', 'http://127.0.0.1:5000/?view=doctorReport'),
    ('screen_08_patient_dashboard.png', 'http://127.0.0.1:5000/?view=patientDashboard'),
    ('screen_09_patient_management.png', 'http://127.0.0.1:5000/?view=patientsManagement'),
    ('screen_10_classic_canvas.png', 'http://127.0.0.1:5000/classic'),
]

print("=== CAPTURING HIGH-RESOLUTION SCREENSHOTS WITH MICROSOFT EDGE ===")
for filename, url in views:
    out_path = os.path.join(out_dir, filename)
    cmd = [
        edge_path,
        '--headless=new',
        '--disable-gpu',
        '--no-sandbox',
        '--window-size=1440,900',
        '--hide-scrollbars',
        '--virtual-time-budget=3000',
        f'--screenshot={out_path}',
        url
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if os.path.exists(out_path):
            print(f"  [+] Captured: {filename:<32} ({os.path.getsize(out_path):,} bytes)")
        else:
            print(f"  [-] Failed: {filename}")
    except Exception as e:
        print(f"  [-] Error {filename}: {e}")
    time.sleep(0.5)

print("\nScreenshot capture process completed.")
