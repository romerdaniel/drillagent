# 🤖 DrillAgent — Autonomous ReAct Drilling Copilot (Multi-Tool + Offset DDR Vector RAG)

**Project 4 (Capstone) of the Drilling ML Progression Series** — An Autonomous Single-Agent AI System (**Observe → Think → Act → Synthesize**) that orchestrates all prior physics-informed ML models (`Axial Flare`, `DrillSense`, `DrillGuard`, `DrillRisk`) alongside **Vector Retrieval-Augmented Generation (RAG)** over historical **Equinor Volve Field Daily Drilling Reports (DDRs)**—with **$0 API keys required**.

---

## 🧠 Why Standalone ML Models Aren't Enough on the Rig Floor
Over Projects 1 through 3C, we built specialized ML models on the **Equinor Volve Well F-9A** dataset (`3,774` depth intervals):
1. **Tool 1 (`predict_baseline_rop`)** — *Axial Flare* Gradient Boosting Regressor (`R² = 0.858`)
2. **Tool 2 (`classify_dysfunction_state`)** — *DrillSense* 4-State Classifier (`96.6%` Accuracy)
3. **Tool 3 (`detect_unsupervised_anomaly`)** — *DrillGuard* Hybrid Autoencoder + Isolation Forest (`0.997` ROC-AUC)
4. **Tool 4 (`evaluate_stand_ahead_npt_economics`)** — *DrillRisk* Cost-Sensitive NPT Optimizer (`19%` Optimal Threshold)
5. **Tool 5 (`retrieve_offset_ddr_sop`)** — TF-IDF Cosine Vector Search over Volve Offset Well Daily Drilling Reports (`15/9-F-9A`, `F-12`, `F-14`, `F-15`)

In real operations, a Drilling Engineer doesn't want to check 5 separate dashboards during a tight-hole or stick-slip event. **DrillAgent** acts as an autonomous **ReAct (Reasoning + Acting) Copilot** that inspects the live WITSML telemetry stream, invokes all 5 tools deterministically, verifies safety guardrails, and outputs a **grounded, zero-hallucination engineering prescription**.

---

## 📊 Agent Architecture Benchmark (200 Volve Test Scenarios)
| Architecture | Diagnosis Accuracy | Action Prescription | Economic Decision | Overall Task Success |
|---|---|---|---|---|
| **Zero-Shot Prompting (No Tools, No RAG)** | `61.5%` | `54.0%` | `52.5%` | `56.0%` |
| **RAG-Only Copilot (Offset DDRs Only)** | `74.0%` | `82.5%` | `66.0%` | `74.2%` |
| **Tool-Only Copilot (ML Models, No RAG)** | `95.5%` | `78.0%` | `93.0%` | `88.8%` |
| **DrillAgent ReAct Loop (5 Tools + Vector RAG)** | **`97.5%`** | **`96.0%`** | **`96.5%`** | **`96.7%`** |

---

## 🚀 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
