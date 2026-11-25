"""
Export ECG Guardian Presentation Deck to Publication-Grade PDF
Uses Playwright with Microsoft Edge to render all 11 slides with exact 16:9 layout.
"""

import os
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

def export_deck_to_pdf():
    script_dir = Path(__file__).resolve().parent
    html_path = script_dir / "index.html"
    output_pdf = script_dir / "ECG_Guardian_Presentation_Deck.pdf"

    if not html_path.exists():
        print(f"[-] Error: {html_path} does not exist!")
        sys.exit(1)

    file_url = html_path.as_uri()
    print("=" * 70)
    print("  ECG GUARDIAN PRESENTATION DECK -> PDF EXPORTER")
    print(f"  Target URL: {file_url}")
    print(f"  Output PDF: {output_pdf}")
    print("=" * 70)

    with sync_playwright() as p:
        print("[*] Launching Headless Microsoft Edge via Playwright...")
        browser = p.chromium.launch(
            channel="msedge",
            headless=True,
            args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"]
        )

        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            device_scale_factor=2
        )

        page = context.new_page()
        print("[*] Loading presentation deck...")
        page.goto(file_url, wait_until="networkidle")

        # Give web fonts and images a brief moment to stabilize
        time.sleep(1.5)

        # Count total slides in DOM
        slide_count = page.evaluate("() => document.querySelectorAll('.slide').length")
        print(f"[+] Verified {slide_count} slides loaded in DOM.")

        print("[*] Generating 16:9 presentation PDF with zero margins...")
        page.emulate_media(media="print")

        page.pdf(
            path=str(output_pdf),
            width="16in",
            height="9in",
            print_background=True,
            margin={"top": "0px", "right": "0px", "bottom": "0px", "left": "0px"},
            prefer_css_page_size=True
        )

        browser.close()

    if output_pdf.exists():
        size_kb = output_pdf.stat().st_size / 1024
        print(f"[✓] SUCCESS! PDF successfully created: {output_pdf}")
        print(f"    File Size: {size_kb:.1f} KB")
        return str(output_pdf)
    else:
        print("[-] PDF export failed.")
        sys.exit(1)

if __name__ == "__main__":
    export_deck_to_pdf()
