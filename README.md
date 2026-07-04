# OlfactoSense Studio

**A Molecular-to-Systems AI Platform for Digital Olfaction**

OlfactoSense Studio is a Streamlit app that demonstrates how artificial olfaction can connect molecular VOC structure, mixture chemistry, sensor-array signals, biological meaning, AI representation learning, uncertainty, robot plume mapping, and health, food, and environmental decisions.

## Tabs
1. Programme Thesis
2. Molecular VOC Atlas
3. Mixture Representation Map
4. Sensor Array Simulator
5. Biological Meaning Model
6. AI Prediction and Uncertainty
7. OlfactoBot Plume Mapping
8. Health, Food, Environment Use Cases
9. Creator Proposal Evaluator

## Run locally

```bash
cd olfactosense_studio
conda activate base
pip install -r requirements.txt --no-cache-dir
streamlit run app.py
```

If your base environment already has the dependencies, try `streamlit run app.py` first.

---

## Added next-generation modules

The app now includes nine additional modules extending the original molecular-to-systems artificial olfaction workflow.

### 10. GC-MS Raw Signal Viewer

Interactive chromatogram and mass-spectrum visualisation for simulated analytical chemistry data. This provides the high-resolution reference layer that can anchor deployable sensor arrays and representation-learning workflows.

### 11. VOC Database Connector

Prototype connector layer for mapping VOCs to public resources such as HMDB, mVOC, FooDB, Cancer Odor Database, and OlfactionBase. The current version uses embedded simulated evidence tables for portability.

### 12. Graph Neural Network Molecular Encoder

Lightweight molecular-graph demonstrator showing how SMILES-like structures and molecular features can be encoded into a learned olfactory embedding.

### 13. Mixture Decoder

AI-style mixture comparison module for predicting mixture similarity, discriminative chemical dimensions, and potential cross-domain transfer.

### 14. Sensor Drift Laboratory

Dedicated module for simulating sensor drift, baseline correction, calibration transfer, and stability gain.

### 15. Open Benchmark Builder

Tool for designing benchmark tasks and challenge datasets for digital olfaction, including domains, data sources, metrics, sample size, difficulty, and expected field utility.

### 16. Dataset Card Generator

Automated metadata and governance documentation generator for olfactory datasets.

### 17. Model Card Generator

Automated model documentation for intended use, inputs, outputs, validation metrics, explainability, uncertainty, limitations, and failure modes.

### 18. Hardware Readiness Assessment

Structured scoring tool for olfactory sensing platforms across sensitivity, selectivity, portability, cost, throughput, drift resistance, and chemical insight.

