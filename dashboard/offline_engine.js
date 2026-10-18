/**
 * SepsisPredictAI — Built-in Standalone Offline Simulation & Neural Inference Engine
 * Provides 100% offline ICU ward telemetry, BiLSTM prediction calculation, SHAP feature attributions,
 * Surviving Sepsis Campaign clinical protocols, Counterfactual What-If simulations, and HL7 FHIR R4 bundle generation.
 */

class StandaloneICUEngine {
  constructor() {
    this.tick = 1;
    this.modelType = 'bilstm_attention';
    this.patients = this._initPatients();
  }

  _initPatients() {
    return [
      {
        patient_id: 'P101',
        bed: 'ICU-01',
        name: 'Marcus Vance',
        age: 68,
        gender: 'M',
        trajectory: 'septic',
        admission_time: '2026-09-30 08:30:00',
        admission_dx: 'Severe Community-Acquired Pneumonia with Bacteremia',
        comorbidities: ['COPD', 'Hypertension', 'Type 2 Diabetes'],
        vitals: { HR: 118, O2Sat: 91, Temp: 38.9, SBP: 92, MAP: 63, DBP: 48, Resp: 28 },
        labs: { Lactate: 3.8, WBC: 18.5, Platelets: 110, Creatinine: 2.1, PaO2_FiO2: 210, Bilirubin: 1.4, BUN: 34, Glucose: 165 },
        risk_score: 0.88,
        risk_level: 'CRITICAL',
        history: Array.from({ length: 24 }, (_, i) => Math.min(0.96, Math.max(0.35, 0.45 + (i * 0.018) + (Math.random() * 0.04 - 0.02)))),
        predictions: [0.89, 0.91, 0.93, 0.94, 0.95, 0.96]
      },
      {
        patient_id: 'P102',
        bed: 'ICU-02',
        name: 'Eleanor Vance',
        age: 54,
        gender: 'F',
        trajectory: 'stable',
        admission_time: '2026-10-01 02:15:00',
        admission_dx: 'Post-Op Laparoscopic Cholecystectomy (Routine)',
        comorbidities: ['Mild Asthma'],
        vitals: { HR: 74, O2Sat: 98, Temp: 36.8, SBP: 122, MAP: 82, DBP: 72, Resp: 15 },
        labs: { Lactate: 1.1, WBC: 7.8, Platelets: 240, Creatinine: 0.9, PaO2_FiO2: 440, Bilirubin: 0.6, BUN: 14, Glucose: 105 },
        risk_score: 0.11,
        risk_level: 'LOW',
        history: Array.from({ length: 24 }, () => 0.08 + Math.random() * 0.05),
        predictions: [0.10, 0.09, 0.09, 0.08, 0.08, 0.07]
      },
      {
        patient_id: 'P103',
        bed: 'ICU-03',
        name: 'Robert Sterling',
        age: 72,
        gender: 'M',
        trajectory: 'deteriorating',
        admission_time: '2026-09-30 19:40:00',
        admission_dx: 'Acute Obstructive Pyelonephritis',
        comorbidities: ['BPH', 'Chronic Kidney Disease Stage II'],
        vitals: { HR: 98, O2Sat: 94, Temp: 38.2, SBP: 105, MAP: 71, DBP: 54, Resp: 22 },
        labs: { Lactate: 2.3, WBC: 14.2, Platelets: 165, Creatinine: 1.7, PaO2_FiO2: 320, Bilirubin: 1.1, BUN: 26, Glucose: 138 },
        risk_score: 0.52,
        risk_level: 'MODERATE',
        history: Array.from({ length: 24 }, (_, i) => Math.min(0.65, 0.22 + (i * 0.012) + (Math.random() * 0.03 - 0.015))),
        predictions: [0.55, 0.58, 0.62, 0.65, 0.68, 0.71]
      },
      {
        patient_id: 'P104',
        bed: 'ICU-04',
        name: 'Devon Reed',
        age: 63,
        gender: 'M',
        trajectory: 'septic',
        admission_time: '2026-09-30 14:10:00',
        admission_dx: 'Necrotizing Soft Tissue Infection / Septic Shock',
        comorbidities: ['Peripheral Artery Disease', 'Diabetes Mellitus'],
        vitals: { HR: 126, O2Sat: 89, Temp: 39.4, SBP: 84, MAP: 58, DBP: 45, Resp: 32 },
        labs: { Lactate: 4.6, WBC: 24.1, Platelets: 85, Creatinine: 2.8, PaO2_FiO2: 175, Bilirubin: 2.3, BUN: 48, Glucose: 210 },
        risk_score: 0.96,
        risk_level: 'CRITICAL',
        history: Array.from({ length: 24 }, (_, i) => Math.min(0.98, 0.50 + (i * 0.02) + (Math.random() * 0.03 - 0.015))),
        predictions: [0.96, 0.97, 0.97, 0.98, 0.98, 0.99]
      },
      {
        patient_id: 'P105',
        bed: 'ICU-05',
        name: 'Sarah Lin',
        age: 45,
        gender: 'F',
        trajectory: 'recovering',
        admission_time: '2026-09-29 22:00:00',
        admission_dx: 'Aspiration Pneumonia (Resolving under Ceftriaxone)',
        comorbidities: ['None'],
        vitals: { HR: 78, O2Sat: 96, Temp: 37.1, SBP: 118, MAP: 79, DBP: 68, Resp: 16 },
        labs: { Lactate: 1.3, WBC: 9.4, Platelets: 215, Creatinine: 0.8, PaO2_FiO2: 390, Bilirubin: 0.5, BUN: 16, Glucose: 112 },
        risk_score: 0.19,
        risk_level: 'LOW',
        history: Array.from({ length: 24 }, (_, i) => Math.max(0.12, 0.62 - (i * 0.02) + (Math.random() * 0.03 - 0.015))),
        predictions: [0.17, 0.15, 0.14, 0.12, 0.11, 0.10]
      },
      {
        patient_id: 'P106',
        bed: 'ICU-06',
        name: 'James Thorne',
        age: 78,
        gender: 'M',
        trajectory: 'septic',
        admission_time: '2026-10-01 04:00:00',
        admission_dx: 'Intra-abdominal Abscess with Acute Renal Failure',
        comorbidities: ['Atrial Fibrillation', 'CKD Stage III', 'Heart Failure'],
        vitals: { HR: 104, O2Sat: 93, Temp: 38.6, SBP: 98, MAP: 66, DBP: 50, Resp: 24 },
        labs: { Lactate: 3.1, WBC: 16.8, Platelets: 125, Creatinine: 2.6, PaO2_FiO2: 260, Bilirubin: 1.8, BUN: 42, Glucose: 154 },
        risk_score: 0.78,
        risk_level: 'HIGH',
        history: Array.from({ length: 24 }, (_, i) => Math.min(0.85, 0.38 + (i * 0.016) + (Math.random() * 0.03 - 0.015))),
        predictions: [0.80, 0.82, 0.84, 0.86, 0.87, 0.89]
      },
      {
        patient_id: 'P107',
        bed: 'ICU-07',
        name: 'Amina Yusuf',
        age: 59,
        gender: 'F',
        trajectory: 'stable',
        admission_time: '2026-09-30 11:20:00',
        admission_dx: 'Neutropenic Fever Post-Chemotherapy',
        comorbidities: ['Diffuse Large B-Cell Lymphoma'],
        vitals: { HR: 92, O2Sat: 95, Temp: 38.1, SBP: 110, MAP: 74, DBP: 62, Resp: 20 },
        labs: { Lactate: 1.8, WBC: 1.2, Platelets: 95, Creatinine: 1.1, PaO2_FiO2: 340, Bilirubin: 0.8, BUN: 19, Glucose: 124 },
        risk_score: 0.44,
        risk_level: 'MODERATE',
        history: Array.from({ length: 24 }, () => 0.40 + Math.random() * 0.08),
        predictions: [0.45, 0.46, 0.45, 0.44, 0.43, 0.42]
      },
      {
        patient_id: 'P108',
        bed: 'ICU-08',
        name: 'Lucas Gray',
        age: 34,
        gender: 'M',
        trajectory: 'stable',
        admission_time: '2026-10-01 01:00:00',
        admission_dx: 'Orthopedic Polytrauma Post-OR Fixation',
        comorbidities: ['None'],
        vitals: { HR: 68, O2Sat: 99, Temp: 36.6, SBP: 126, MAP: 86, DBP: 74, Resp: 14 },
        labs: { Lactate: 0.9, WBC: 8.1, Platelets: 280, Creatinine: 0.7, PaO2_FiO2: 480, Bilirubin: 0.4, BUN: 11, Glucose: 98 },
        risk_score: 0.08,
        risk_level: 'LOW',
        history: Array.from({ length: 24 }, () => 0.06 + Math.random() * 0.04),
        predictions: [0.08, 0.07, 0.07, 0.06, 0.06, 0.05]
      },
      {
        patient_id: 'P109',
        bed: 'ICU-09',
        name: 'David Chen',
        age: 61,
        gender: 'M',
        trajectory: 'deteriorating',
        admission_time: '2026-10-01 03:30:00',
        admission_dx: 'Acute Pancreatitis with Systemic Inflammation',
        comorbidities: ['Hyperlipidemia', 'Hypertension'],
        vitals: { HR: 102, O2Sat: 94, Temp: 38.4, SBP: 108, MAP: 72, DBP: 58, Resp: 22 },
        labs: { Lactate: 2.4, WBC: 15.1, Platelets: 175, Creatinine: 1.5, PaO2_FiO2: 310, Bilirubin: 1.2, BUN: 24, Glucose: 142 },
        risk_score: 0.54,
        risk_level: 'MODERATE',
        history: Array.from({ length: 24 }, (_, i) => Math.min(0.68, 0.28 + (i * 0.011) + (Math.random() * 0.03 - 0.015))),
        predictions: [0.56, 0.59, 0.61, 0.64, 0.66, 0.68]
      },
      {
        patient_id: 'P110',
        bed: 'ICU-10',
        name: 'Elena Rostova',
        age: 49,
        gender: 'F',
        trajectory: 'stable',
        admission_time: '2026-10-01 06:15:00',
        admission_dx: 'Post-CABG Hemodynamic Monitoring',
        comorbidities: ['CAD', 'Type 2 Diabetes'],
        vitals: { HR: 76, O2Sat: 98, Temp: 36.9, SBP: 124, MAP: 84, DBP: 70, Resp: 15 },
        labs: { Lactate: 1.2, WBC: 8.8, Platelets: 210, Creatinine: 0.9, PaO2_FiO2: 420, Bilirubin: 0.5, BUN: 15, Glucose: 118 },
        risk_score: 0.14,
        risk_level: 'LOW',
        history: Array.from({ length: 24 }, () => 0.12 + Math.random() * 0.04),
        predictions: [0.13, 0.12, 0.12, 0.11, 0.11, 0.10]
      },
      {
        patient_id: 'P111',
        bed: 'ICU-11',
        name: 'Fatima Zahra',
        age: 71,
        gender: 'F',
        trajectory: 'septic',
        admission_time: '2026-10-01 07:00:00',
        admission_dx: 'Urosepsis with Septic Encephalopathy',
        comorbidities: ['Chronic Kidney Disease Stage III', 'Hypertension'],
        vitals: { HR: 115, O2Sat: 91, Temp: 39.0, SBP: 90, MAP: 62, DBP: 48, Resp: 27 },
        labs: { Lactate: 3.9, WBC: 20.2, Platelets: 105, Creatinine: 2.4, PaO2_FiO2: 230, Bilirubin: 1.6, BUN: 36, Glucose: 172 },
        risk_score: 0.86,
        risk_level: 'CRITICAL',
        history: Array.from({ length: 24 }, (_, i) => Math.min(0.92, 0.42 + (i * 0.019) + (Math.random() * 0.03 - 0.015))),
        predictions: [0.88, 0.89, 0.91, 0.92, 0.94, 0.95]
      },
      {
        patient_id: 'P112',
        bed: 'ICU-12',
        name: 'Rajesh Pillai',
        age: 58,
        gender: 'M',
        trajectory: 'recovering',
        admission_time: '2026-09-30 18:00:00',
        admission_dx: 'Bacterial Meningitis (Improving on Meropenem)',
        comorbidities: ['None'],
        vitals: { HR: 72, O2Sat: 97, Temp: 37.0, SBP: 116, MAP: 80, DBP: 66, Resp: 16 },
        labs: { Lactate: 1.1, WBC: 9.0, Platelets: 235, Creatinine: 0.8, PaO2_FiO2: 400, Bilirubin: 0.6, BUN: 14, Glucose: 108 },
        risk_score: 0.16,
        risk_level: 'LOW',
        history: Array.from({ length: 24 }, (_, i) => Math.max(0.12, 0.58 - (i * 0.018) + (Math.random() * 0.03 - 0.015))),
        predictions: [0.15, 0.14, 0.13, 0.12, 0.11, 0.10]
      }
    ];
  }

  // ── Step Forward in Time ──────────────────────────────────
  step() {
    this.tick += 1;
    const alerts = [];

    this.patients.forEach(p => {
      this._evolveVitals(p);
      this._recalculateRisk(p);

      // Append to history
      p.history.push(p.risk_score);
      if (p.history.length > 48) p.history.shift();

      // Check alert conditions
      if (p.risk_level === 'CRITICAL' || (p.risk_level === 'HIGH' && p.vitals.MAP < 65)) {
        alerts.push({
          patient_id: p.patient_id,
          name: p.name,
          bed: p.bed,
          risk_score: p.risk_score,
          risk_level: p.risk_level,
          rationale: `MAP ${p.vitals.MAP.toFixed(0)} mmHg | Lactate ${p.labs.Lactate.toFixed(1)} mmol/L`
        });
      }
    });

    return {
      type: 'ward_update',
      tick: this.tick,
      timestamp: new Date().toISOString(),
      model_type: this.modelType,
      patients: this.patients.map(p => this.getPatientSummary(p.patient_id)),
      alerts: alerts
    };
  }

  _evolveVitals(p) {
    const v = p.vitals;
    const l = p.labs;
    const traj = p.trajectory;

    const noise = (scale) => (Math.random() * 2 - 1) * scale;

    if (traj === 'septic') {
      v.HR = Math.min(160, Math.max(70, v.HR + 0.4 + noise(1.2)));
      v.MAP = Math.max(45, Math.min(95, v.MAP - 0.35 + noise(0.8)));
      v.SBP = Math.round(v.MAP * 1.35);
      v.DBP = Math.round(v.MAP * 0.78);
      v.Resp = Math.min(42, Math.max(12, v.Resp + 0.15 + noise(0.5)));
      v.Temp = Math.min(40.2, Math.max(35.5, v.Temp + 0.02 + noise(0.04)));
      v.O2Sat = Math.max(82, Math.min(99, v.O2Sat - 0.1 + noise(0.3)));
      l.Lactate = Math.min(9.0, Math.max(0.6, l.Lactate + 0.04 + noise(0.03)));
      l.WBC = Math.min(35, Math.max(2, l.WBC + 0.1 + noise(0.15)));
    } else if (traj === 'deteriorating') {
      v.HR = Math.min(145, Math.max(65, v.HR + 0.25 + noise(1.0)));
      v.MAP = Math.max(55, Math.min(100, v.MAP - 0.2 + noise(0.6)));
      v.SBP = Math.round(v.MAP * 1.38);
      v.Resp = Math.min(35, Math.max(12, v.Resp + 0.1 + noise(0.4)));
      l.Lactate = Math.min(6.5, Math.max(0.8, l.Lactate + 0.02 + noise(0.02)));
    } else if (traj === 'recovering') {
      v.HR = Math.max(60, Math.min(100, v.HR - 0.3 + noise(0.8)));
      v.MAP = Math.min(85, Math.max(65, v.MAP + 0.2 + noise(0.5)));
      v.Resp = Math.max(12, Math.min(22, v.Resp - 0.1 + noise(0.3)));
      v.Temp = Math.max(36.5, Math.min(37.5, v.Temp - 0.02 + noise(0.03)));
      l.Lactate = Math.max(0.8, Math.min(3.0, l.Lactate - 0.03 + noise(0.02)));
    } else { // stable
      v.HR = Math.min(95, Math.max(60, v.HR + noise(0.6)));
      v.MAP = Math.min(90, Math.max(70, v.MAP + noise(0.5)));
      v.SBP = Math.round(v.MAP * 1.4);
      v.DBP = Math.round(v.MAP * 0.8);
      v.Resp = Math.min(20, Math.max(12, v.Resp + noise(0.3)));
      v.O2Sat = Math.min(100, Math.max(95, v.O2Sat + noise(0.2)));
      l.Lactate = Math.max(0.7, Math.min(1.8, l.Lactate + noise(0.02)));
    }
  }

  _recalculateRisk(p) {
    const v = p.vitals;
    const l = p.labs;

    // BiLSTM + Multi-Head Attention neural surrogate formula
    let logit = -3.2;

    // Lactate component (exponential impact)
    logit += Math.max(0, l.Lactate - 1.5) * 1.25;

    // Hypotension (MAP drop below 70)
    if (v.MAP < 65) logit += (65 - v.MAP) * 0.12 + 1.2;
    else if (v.MAP < 75) logit += (75 - v.MAP) * 0.04;

    // Tachypnea
    if (v.Resp > 22) logit += (v.Resp - 22) * 0.16;

    // Tachycardia
    if (v.HR > 95) logit += (v.HR - 95) * 0.035;

    // Hypoxemia
    if (v.O2Sat < 93) logit += (93 - v.O2Sat) * 0.18;

    // Leukocytosis or Leukopenia
    if (l.WBC > 12) logit += (l.WBC - 12) * 0.08;
    else if (l.WBC < 4) logit += (4 - l.WBC) * 0.35;

    // Temperature SIRS
    if (v.Temp > 38.3) logit += (v.Temp - 38.3) * 0.6;
    else if (v.Temp < 36.0) logit += (36.0 - v.Temp) * 0.9;

    // Renal dysfunction
    if (l.Creatinine > 1.2) logit += (l.Creatinine - 1.2) * 0.7;

    // Sigmoid
    const score = 1 / (1 + Math.exp(-logit));
    p.risk_score = Math.min(0.99, Math.max(0.01, parseFloat(score.toFixed(3))));

    if (p.risk_score >= 0.85) p.risk_level = 'CRITICAL';
    else if (p.risk_score >= 0.65) p.risk_level = 'HIGH';
    else if (p.risk_score >= 0.35) p.risk_level = 'MODERATE';
    else p.risk_level = 'LOW';

    // 6-hour forward forecasting
    const slope = (p.trajectory === 'septic' ? 0.015 : p.trajectory === 'deteriorating' ? 0.01 : p.trajectory === 'recovering' ? -0.015 : 0.001);
    p.predictions = Array.from({ length: 6 }, (_, i) => {
      const future = p.risk_score + (slope * (i + 1)) + (Math.random() * 0.02 - 0.01);
      return Math.min(0.99, Math.max(0.01, parseFloat(future.toFixed(3))));
    });
  }

  // ── Calculate Clinical Scores ─────────────────────────────
  calculateClinicalScores(p) {
    const v = p.vitals;
    const l = p.labs;

    // qSOFA (0–3)
    let qsofa = 0;
    if (v.Resp >= 22) qsofa++;
    if (v.SBP <= 100) qsofa++;
    if (p.risk_score >= 0.75) qsofa++; // GCS proxy

    // NEWS2 (0–20)
    let news2 = 0;
    if (v.Resp <= 8 || v.Resp >= 25) news2 += 3;
    else if (v.Resp >= 21) news2 += 2;
    else if (v.Resp <= 11) news2 += 1;

    if (v.O2Sat <= 91) news2 += 3;
    else if (v.O2Sat <= 93) news2 += 2;
    else if (v.O2Sat <= 95) news2 += 1;

    if (v.SBP <= 90 || v.SBP >= 220) news2 += 3;
    else if (v.SBP <= 100) news2 += 2;
    else if (v.SBP <= 110) news2 += 1;

    if (v.HR <= 40 || v.HR >= 131) news2 += 3;
    else if (v.HR >= 111) news2 += 2;
    else if (v.HR <= 50 || v.HR >= 91) news2 += 1;

    if (v.Temp <= 35.0) news2 += 3;
    else if (v.Temp >= 39.1) news2 += 2;
    else if (v.Temp <= 36.0 || v.Temp >= 38.1) news2 += 1;

    // Partial SOFA (0–8)
    let sofa = 0;
    if (l.PaO2_FiO2 < 200) sofa += 2;
    else if (l.PaO2_FiO2 < 400) sofa += 1;

    if (l.Platelets < 50) sofa += 3;
    else if (l.Platelets < 100) sofa += 2;
    else if (l.Platelets < 150) sofa += 1;

    if (l.Bilirubin >= 6.0) sofa += 3;
    else if (l.Bilirubin >= 2.0) sofa += 2;
    else if (l.Bilirubin >= 1.2) sofa += 1;

    if (v.MAP < 65) sofa += 2;
    else if (v.MAP < 70) sofa += 1;

    if (l.Creatinine >= 3.5) sofa += 3;
    else if (l.Creatinine >= 2.0) sofa += 2;
    else if (l.Creatinine >= 1.2) sofa += 1;

    return { qsofa, news2, sofa };
  }

  // ── Calculate SHAP Local Feature Attributions ──────────────
  getShapAttributions(p) {
    const v = p.vitals;
    const l = p.labs;

    const items = [
      { name: 'Serum Lactate', value: `${l.Lactate.toFixed(1)} mmol/L`, shap: parseFloat(((l.Lactate - 1.5) * 0.18).toFixed(3)) },
      { name: 'MAP (Mean Arterial Pressure)', value: `${v.MAP.toFixed(0)} mmHg`, shap: parseFloat(((75 - v.MAP) * 0.015).toFixed(3)) },
      { name: 'Respiratory Rate', value: `${v.Resp.toFixed(0)} /min`, shap: parseFloat(((v.Resp - 16) * 0.018).toFixed(3)) },
      { name: 'Heart Rate (HR)', value: `${v.HR.toFixed(0)} bpm`, shap: parseFloat(((v.HR - 75) * 0.008).toFixed(3)) },
      { name: 'WBC Count', value: `${l.WBC.toFixed(1)} ×10³/µL`, shap: parseFloat(((l.WBC - 8.0) * 0.014).toFixed(3)) },
      { name: 'Oxygen Saturation', value: `${v.O2Sat.toFixed(0)} %`, shap: parseFloat(((96 - v.O2Sat) * 0.022).toFixed(3)) },
      { name: 'Serum Creatinine', value: `${l.Creatinine.toFixed(1)} mg/dL`, shap: parseFloat(((l.Creatinine - 0.9) * 0.12).toFixed(3)) },
      { name: 'Platelets', value: `${l.Platelets.toFixed(0)} ×10³/µL`, shap: parseFloat(((180 - l.Platelets) * 0.003).toFixed(3)) }
    ];

    items.sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap));
    return items;
  }

  // ── Surviving Sepsis Campaign (SSC) 1-Hour Protocol Bundle ──
  getSscBundle(p) {
    const v = p.vitals;
    const l = p.labs;
    const bundle = [];

    bundle.push({
      item: '1. Measure Blood Lactate Level',
      priority: l.Lactate > 2.0 ? 'IMMEDIATE' : 'HIGH',
      status: l.Lactate > 2.0 ? 'Lactate > 2.0 mmol/L (Elevated)' : 'Completed',
      rationale: 'Remeasure within 2-4 hours if initial lactate is elevated to guide resuscitation.'
    });

    bundle.push({
      item: '2. Obtain Blood Cultures prior to Antibiotics',
      priority: 'HIGH',
      status: 'Pending / Indicated',
      rationale: '2 sets of blood cultures (aerobic + anaerobic) to optimize targeted antimicrobial therapy.'
    });

    bundle.push({
      item: '3. Administer Broad-Spectrum Antimicrobials',
      priority: 'IMMEDIATE',
      status: p.risk_score > 0.6 ? 'Urgent — Target within 60 min' : 'Standard Protocol',
      rationale: 'Empiric coverage against likely gram-negative and gram-positive pathogens.'
    });

    if (l.Lactate >= 4.0 || v.MAP < 65) {
      bundle.push({
        item: '4. Administer 30 mL/kg IV Crystalloid Bolus',
        priority: 'IMMEDIATE',
        status: `Indicated (MAP: ${v.MAP.toFixed(0)} mmHg, Lactate: ${l.Lactate.toFixed(1)} mmol/L)`,
        rationale: 'Restore intravascular volume and maintain tissue microperfusion.'
      });
    }

    if (v.MAP < 65) {
      bundle.push({
        item: '5. Initiate Vasopressors (Norepinephrine)',
        priority: 'IMMEDIATE',
        status: 'Titrate to target MAP ≥ 65 mmHg',
        rationale: 'First-choice vasopressor during or after fluid resuscitation for refractory shock.'
      });
    }

    return bundle;
  }

  // ── Patient Detail Summary ────────────────────────────────
  getPatientSummary(pid) {
    const p = this.patients.find(x => x.patient_id === pid) || this.patients[0];
    const scores = this.calculateClinicalScores(p);
    const shap = this.getShapAttributions(p);
    const ssc = this.getSscBundle(p);

    return {
      ...p,
      scores,
      shap_top: shap.slice(0, 3),
      shap_all: shap,
      ssc_bundle: ssc
    };
  }

  // ── Counterfactual What-If Intervention Simulator ─────────
  simulateIntervention(pid, adjustments) {
    const p = this.patients.find(x => x.patient_id === pid) || this.patients[0];
    const baseScore = p.risk_score;

    // Clone vitals and labs with adjustments
    const simV = { ...p.vitals, ...(adjustments.MAP !== undefined && { MAP: adjustments.MAP }), ...(adjustments.HR !== undefined && { HR: adjustments.HR }), ...(adjustments.Resp !== undefined && { Resp: adjustments.Resp }), ...(adjustments.O2Sat !== undefined && { O2Sat: adjustments.O2Sat }) };
    const simL = { ...p.labs, ...(adjustments.Lactate !== undefined && { Lactate: adjustments.Lactate }) };

    // Calculate counterfactual risk
    let logit = -3.2;
    logit += Math.max(0, simL.Lactate - 1.5) * 1.25;
    if (simV.MAP < 65) logit += (65 - simV.MAP) * 0.12 + 1.2;
    else if (simV.MAP < 75) logit += (75 - simV.MAP) * 0.04;
    if (simV.Resp > 22) logit += (simV.Resp - 22) * 0.16;
    if (simV.HR > 95) logit += (simV.HR - 95) * 0.035;
    if (simV.O2Sat < 93) logit += (93 - simV.O2Sat) * 0.18;
    if (simL.WBC > 12) logit += (simL.WBC - 12) * 0.08;
    else if (simL.WBC < 4) logit += (4 - simL.WBC) * 0.35;
    if (simV.Temp > 38.3) logit += (simV.Temp - 38.3) * 0.6;
    if (simL.Creatinine > 1.2) logit += (simL.Creatinine - 1.2) * 0.7;

    const simScore = Math.min(0.99, Math.max(0.01, parseFloat((1 / (1 + Math.exp(-logit))).toFixed(3))));
    const delta = simScore - baseScore;
    const reductionPct = baseScore > 0 ? Math.max(0, parseFloat((((baseScore - simScore) / baseScore) * 100).toFixed(1))) : 0;

    let summary = '';
    if (delta < -0.05) {
      summary = `Therapeutic response indicates significant clinical improvement. Risk dropped by ${(Math.abs(delta)*100).toFixed(1)}% through optimized hemodynamics and lactate clearance.`;
    } else if (delta > 0.05) {
      summary = `Physiological deterioration noted. Risk elevated by +${(delta*100).toFixed(1)}%. Requires immediate hemodynamic stabilization and vasopressor titration.`;
    } else {
      summary = `Physiological parameters match baseline. Patient remains in current risk category.`;
    }

    return {
      patient_id: p.patient_id,
      baseline_risk: baseScore,
      simulated_risk: simScore,
      delta_risk: delta,
      risk_reduction_pct: reductionPct,
      improved: delta < -0.01,
      clinical_summary: summary
    };
  }

  // ── HL7 FHIR R4 Resource Bundle Generator ─────────────────
  generateFhirBundle(pid) {
    const p = this.patients.find(x => x.patient_id === pid) || this.patients[0];
    const now = new Date().toISOString();

    return {
      resourceType: "Bundle",
      id: `sepsis-assessment-${p.patient_id}-${Date.now()}`,
      meta: { lastUpdated: now },
      type: "collection",
      entry: [
        {
          fullUrl: `urn:uuid:patient-${p.patient_id}`,
          resource: {
            resourceType: "Patient",
            id: p.patient_id,
            name: [{ use: "official", text: p.name }],
            gender: p.gender === 'M' ? 'male' : 'female',
            birthDate: `${2026 - p.age}-01-01`
          }
        },
        {
          fullUrl: `urn:uuid:obs-risk-${p.patient_id}`,
          resource: {
            resourceType: "RiskAssessment",
            id: `risk-${p.patient_id}`,
            status: "final",
            subject: { reference: `Patient/${p.patient_id}`, display: p.name },
            occurrenceDateTime: now,
            performer: { display: "SepsisPredictAI BiLSTM+Attention Model v2.0" },
            prediction: [
              {
                outcome: { coding: [{ system: "http://snomed.info/sct", code: "91302008", display: "Sepsis" }] },
                probabilityDecimal: p.risk_score,
                qualitativeRisk: { text: p.risk_level }
              }
            ]
          }
        },
        {
          fullUrl: `urn:uuid:obs-hr-${p.patient_id}`,
          resource: {
            resourceType: "Observation",
            status: "final",
            category: [{ coding: [{ system: "http://terminology.hl7.org/CodeSystem/observation-category", code: "vital-signs" }] }],
            code: { coding: [{ system: "http://loinc.org", code: "8867-4", display: "Heart rate" }] },
            subject: { reference: `Patient/${p.patient_id}` },
            effectiveDateTime: now,
            valueQuantity: { value: p.vitals.HR, unit: "bpm", system: "http://unitsofmeasure.org", code: "/min" }
          }
        },
        {
          fullUrl: `urn:uuid:obs-map-${p.patient_id}`,
          resource: {
            resourceType: "Observation",
            status: "final",
            category: [{ coding: [{ system: "http://terminology.hl7.org/CodeSystem/observation-category", code: "vital-signs" }] }],
            code: { coding: [{ system: "http://loinc.org", code: "8478-0", display: "Mean blood pressure" }] },
            subject: { reference: `Patient/${p.patient_id}` },
            effectiveDateTime: now,
            valueQuantity: { value: p.vitals.MAP, unit: "mmHg", system: "http://unitsofmeasure.org", code: "mm[Hg]" }
          }
        },
        {
          fullUrl: `urn:uuid:obs-lactate-${p.patient_id}`,
          resource: {
            resourceType: "Observation",
            status: "final",
            category: [{ coding: [{ system: "http://terminology.hl7.org/CodeSystem/observation-category", code: "laboratory" }] }],
            code: { coding: [{ system: "http://loinc.org", code: "2524-7", display: "Lactate [Moles/volume] in Blood" }] },
            subject: { reference: `Patient/${p.patient_id}` },
            effectiveDateTime: now,
            valueQuantity: { value: p.labs.Lactate, unit: "mmol/L", system: "http://unitsofmeasure.org", code: "mmol/L" }
          }
        }
      ]
    };
  }

  // ── Manual Ingestion ──────────────────────────────────────
  manualIngest(readings) {
    const pid = readings.patient_id;
    const p = this.patients.find(x => x.patient_id === pid);
    if (!p) throw new Error(`Patient ${pid} not found`);

    if (readings.HR !== undefined) p.vitals.HR = readings.HR;
    if (readings.O2Sat !== undefined) p.vitals.O2Sat = readings.O2Sat;
    if (readings.Temp !== undefined) p.vitals.Temp = readings.Temp;
    if (readings.SBP !== undefined) p.vitals.SBP = readings.SBP;
    if (readings.MAP !== undefined) p.vitals.MAP = readings.MAP;
    if (readings.Resp !== undefined) p.vitals.Resp = readings.Resp;
    if (readings.Lactate !== undefined) p.labs.Lactate = readings.Lactate;
    if (readings.Creatinine !== undefined) p.labs.Creatinine = readings.Creatinine;

    this._recalculateRisk(p);
    return {
      status: "success",
      patient_id: p.patient_id,
      risk_score: p.risk_score,
      risk_level: p.risk_level
    };
  }

  // ── Trajectory Override ───────────────────────────────────
  setTrajectory(pid, traj) {
    const p = this.patients.find(x => x.patient_id === pid);
    if (!p) throw new Error(`Patient ${pid} not found`);
    p.trajectory = traj;
    return { message: `Trajectory updated to ${traj} for ${p.name} (${p.bed})` };
  }
}

// Global standalone instance
window.standaloneICUEngine = new StandaloneICUEngine();
