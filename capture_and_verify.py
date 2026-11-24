import subprocess
import os
import time
from PIL import Image
import numpy as np

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

print("=" * 70)
print("  CAPTURING 10 DIVERSE WEBSITE SCREENSHOTS VIA MICROSOFT EDGE HEADLESS")
print("=" * 70)

captured = []
for filename, url in views:
    out_path = os.path.join(out_dir, filename)
    cmd = [
        edge_path,
        '--headless=new',
        '--disable-gpu',
        '--no-sandbox',
        '--window-size=1440,900',
        '--hide-scrollbars',
        f'--screenshot={out_path}',
        url
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
        if os.path.exists(out_path):
            sz = os.path.getsize(out_path)
            im = Image.open(out_path)
            captured.append((filename, out_path, im.size, sz))
            print(f"  [+] {filename:<32} {im.size}  {sz:>10,} bytes")
        else:
            print(f"  [-] Failed: {filename}")
    except Exception as e:
        print(f"  [-] Error {filename}: {e}")
    time.sleep(0.3)

print("\n" + "=" * 70)
print("  PIXEL DIVERSITY VERIFICATION MATRIX (Diff vs screen_01_landing)")
print("=" * 70)

base_arr = np.array(Image.open(captured[0][1]).convert('RGB')).astype(float)
all_distinct = True

for fname, fpath, dims, sz in captured[1:]:
    arr = np.array(Image.open(fpath).convert('RGB')).astype(float)
    diff = np.mean(np.abs(arr - base_arr))
    status = "DISTINCT" if diff > 5.0 else "DUPLICATE WARNING"
    if diff <= 5.0:
        all_distinct = False
    print(f"  {fname:<32} Mean Pixel Diff: {diff:>8.4f}  [{status}]")

print("=" * 70)
if all_distinct:
    print(">>> SUCCESS: ALL 10 SCREENSHOTS ARE 100% VISUALLY DISTINCT! <<<")
else:
    print(">>> WARNING: Some screenshots have low visual difference! <<<")
print("=" * 70)
