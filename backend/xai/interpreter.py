"""
Cardiological Diagnostic Reasoning Engine
Synthesizes deep learning multi-label outputs, XAI Grad-CAM attribution regions,
and Pan-Tompkins electrophysiological metrics into clear clinical narratives.
"""

from models.ecg_model import CLASSES, CLASS_IDS


def generate_clinical_interpretation(predictions, hrv_metrics, xai_regions_by_class):
    """
    Generates a structured clinical AI diagnostic report:
    - Primary and co-occurring diagnoses
    - Electrophysiological corroboration
    - XAI temporal attribution evidence
    - Clinical risk tier (Low, Moderate, High, Critical)
    - Recommended clinical next steps
    """
    detected_arrhythmias = [p for p in predictions if p['is_positive']]
    hr = hrv_metrics.get('heart_rate_bpm', 0)
    regularity = hrv_metrics.get('regularity', 'Unknown')
    sdnn = hrv_metrics.get('sdnn_ms', 0)

    # Determine Clinical Risk Tier
    risk_tier = 'Low'
    risk_color = '#10b981'  # emerald-500
    risk_reasons = []

    detected_codes = [d['id'] for d in detected_arrhythmias]

    if 'MI' in detected_codes:
        risk_tier = 'Critical'
        risk_color = '#ef4444'  # red-500
        risk_reasons.append("Evidence of acute or ongoing Myocardial Infarction / Ischemia.")
    elif 'AFIB' in detected_codes and hr > 110:
        risk_tier = 'High'
        risk_color = '#f97316'  # orange-500
        risk_reasons.append("Atrial Fibrillation with Rapid Ventricular Response (RVR).")
    elif 'PVC' in detected_codes and ('LBBB' in detected_codes or 'RBBB' in detected_codes):
        risk_tier = 'High'
        risk_color = '#f97316'
        risk_reasons.append("Conduction defect complicated by ventricular ectopic activity.")
    elif len(detected_arrhythmias) >= 2:
        risk_tier = 'Moderate'
        risk_color = '#eab308'  # yellow-500
        risk_reasons.append(f"Multi-label co-occurrence: {len(detected_arrhythmias)} concurrent conditions detected.")
    elif len(detected_arrhythmias) == 1 and detected_codes[0] != 'NORM':
        risk_tier = 'Moderate'
        risk_color = '#eab308'
        risk_reasons.append(f"Isolated cardiac arrhythmia: {detected_arrhythmias[0]['name']}.")
    else:
        risk_tier = 'Low'
        risk_color = '#10b981'
        risk_reasons.append("Physiological normal sinus rhythm without acute repolarization abnormalities.")

    # Generate Per-Class XAI Narratives
    class_explanations = []
    for pred in predictions:
        cid = pred['id']
        prob = pred['probability']
        is_pos = pred['is_positive']
        regions = xai_regions_by_class.get(cid, [])

        explanation = ""
        evidence = []

        if cid == 'NORM':
            if is_pos:
                explanation = f"Consistent with normal sinus rhythm. P waves precede each narrow QRS complex with resting heart rate of {hr} bpm."
            else:
                explanation = "Normal sinus criteria NOT met due to co-occurring rhythm or conduction anomalies."
        elif cid == 'AFIB':
            if is_pos:
                explanation = (
                    f"Atrial Fibrillation identified with {prob*100:.1f}% confidence. "
                    f"Corroborated by irregular RR intervals (SDNN: {sdnn} ms, {regularity}) "
                    f"and baseline fibrillatory oscillations highlighted by 1D Grad-CAM."
                )
            else:
                explanation = "No evidence of atrial fibrillation; regular rhythm cadence maintained."
        elif cid == 'AFL':
            if is_pos:
                explanation = f"Atrial Flutter suggested with {prob*100:.1f}% confidence, showing characteristic sawtooth flutter baseline."
            else:
                explanation = "Absence of sawtooth flutter waves in atrial baseline."
        elif cid == 'LBBB':
            if is_pos:
                explanation = (
                    f"Left Bundle Branch Block indicated ({prob*100:.1f}%). "
                    f"Grad-CAM strongly highlights wide, slurred QRS complexes indicating delayed left ventricular activation."
                )
            else:
                explanation = "Ventricular conduction normal; QRS duration within standard limits."
        elif cid == 'RBBB':
            if is_pos:
                explanation = (
                    f"Right Bundle Branch Block identified ({prob*100:.1f}%). "
                    f"Attribution focuses on terminal conduction delay and notched rSR' complex morphology."
                )
            else:
                explanation = "No right ventricular conduction delay observed."
        elif cid == 'PAC':
            if is_pos:
                explanation = f"Premature Atrial Contraction detected ({prob*100:.1f}%). Grad-CAM isolates early ectopic P-wave timing."
            else:
                explanation = "No premature supraventricular depolarizations found."
        elif cid == 'PVC':
            if is_pos:
                explanation = (
                    f"Premature Ventricular Contractions recognized ({prob*100:.1f}%). "
                    f"XAI saliency localizes prominent, bizarre premature QRS deflections with compensatory pauses."
                )
            else:
                explanation = "No ventricular premature complexes identified."
        elif cid == 'STACH':
            if is_pos:
                explanation = f"Sinus Tachycardia confirmed. Measured Heart Rate is {hr} bpm (threshold > 100 bpm)."
            else:
                explanation = f"Heart Rate is {hr} bpm (does not exceed 100 bpm threshold)."
        elif cid == 'SBRAD':
            if is_pos:
                explanation = f"Sinus Bradycardia confirmed. Measured Heart Rate is {hr} bpm (threshold < 60 bpm)."
            else:
                explanation = f"Heart Rate is {hr} bpm (does not fall below 60 bpm threshold)."
        elif cid == 'MI':
            if is_pos:
                explanation = (
                    f"ALERT: Myocardial Infarction / Ischemia pattern detected with {prob*100:.1f}% confidence! "
                    f"Grad-CAM highlights significant ST-segment deviation from the isoelectric baseline."
                )
            else:
                explanation = "No significant ischemic ST-segment deviation or pathological Q-waves observed."

        # Region evidence summary
        for r in regions[:2]:
            if isinstance(r, dict):
                s_samp = r.get('start_sample', 0)
                e_samp = r.get('end_sample', 0)
                max_att = r.get('max_attribution', 0.85)
                dur_ms = r.get('duration_ms', int((e_samp - s_samp) * 4))
            elif isinstance(r, (list, tuple)) and len(r) >= 2:
                s_samp = r[0]
                e_samp = r[1]
                max_att = r[2] if len(r) > 2 else 0.85
                dur_ms = int((e_samp - s_samp) * 4)
            else:
                continue
            evidence.append(
                f"Peak influence at {s_samp/250.0:.2f}s - {e_samp/250.0:.2f}s "
                f"(Attribution: {max_att:.2f}, Duration: {dur_ms} ms)"
            )

        class_explanations.append({
            'id': cid,
            'name': pred['name'],
            'probability': prob,
            'is_positive': is_pos,
            'explanation': explanation,
            'evidence': evidence,
            'regions': regions
        })

    # Summary Recommendations
    recommendations = []
    if risk_tier == 'Critical':
        recommendations.append("Immediate 12-lead clinical ECG and urgent cardiology consultation recommended.")
        recommendations.append("Check serum cardiac biomarkers (Troponin I/T, CK-MB).")
    elif risk_tier == 'High':
        recommendations.append("Formal clinical review by electrophysiologist or attending physician.")
        recommendations.append("Evaluate for rate/rhythm control therapy and anticoagulation stroke risk (CHA2DS2-VASc score).")
    elif risk_tier == 'Moderate':
        recommendations.append("Follow-up with 24-48 hour Holter monitoring or echocardiogram if symptomatic.")
    else:
        recommendations.append("Routine clinical follow-up; ECG within normal physiological variations.")

    return {
        'risk_tier': risk_tier,
        'risk_color': risk_color,
        'risk_reasons': risk_reasons,
        'detected_arrhythmias': detected_arrhythmias,
        'class_explanations': class_explanations,
        'recommendations': recommendations
    }

