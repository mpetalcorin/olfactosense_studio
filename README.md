# OlfactoSense Studio

**A Molecular-to-Systems AI Platform for Digital Olfaction**

OlfactoSense Studio is an interactive Streamlit application for demonstrating how artificial olfaction can be framed as a full molecular-to-systems AI problem. It connects volatile organic compounds, mixture chemistry, sensor-array responses, environmental confounders, biological meaning, machine-learning prediction, uncertainty, robot plume mapping, and cross-domain use cases in health, food, and the environment.
<img width="1536" height="1024" alt="OlfactoSense" src="https://github.com/user-attachments/assets/e71c8150-6b07-411a-b5a5-eee0f7867fae" />
The central idea is:

> A smell is a dynamic, context-dependent chemical signal. To make it useful for AI, we need to capture the molecule, the mixture, the sensor response, the environment, the biological meaning, and the decision.

This app is designed as a demonstration for **AI for Chemistry**, **digital olfaction**, **representation learning**, **sensor intelligence**, and **biological systems modelling**.

---

## Relevance

Artificial intelligence can process images, text, sound, protein structures, and multimodal data, but it still lacks a general-purpose artificial sense of smell. Digital olfaction remains fragmented because most systems are built as narrow point solutions, often with application-specific sensors, small datasets, limited calibration, and weak transfer across domains.

OlfactoSense Studio demonstrates a different design philosophy:

- build from molecular chemistry upward,
- represent real smells as mixtures rather than single molecules,
- include sensor behaviour and environmental confounders,
- add biological meaning, not only human odour labels,
- estimate uncertainty,
- connect predictions to deployable actions,
- evaluate whether a project contributes to general-purpose olfactory perception.

The app is inspired by the idea that digital olfaction needs the equivalent of an **RGB camera plus ImageNet** for smell: standardised sensing, open datasets, shared benchmarks, and learned representations.

---

## Key features

### 1. A visual overview of the full artificial olfaction stack:

```text
VOC molecule → mixture → sensor response → AI representation → biological meaning → decision → robot action
```

This tab explains why a smell should be treated as a dynamic, context-dependent chemical signal rather than a single molecule.

### 2. Molecular VOC Atlas

An interactive atlas of representative volatile organic compounds, including acetone, ethanol, ammonia, hydrogen sulfide, benzaldehyde, 2-nonanone, dimethyl sulfide, and limonene.

For each VOC, the app displays chemical class, formula, molecular weight, volatility, polarity, functional groups, likely microbial, metabolic, food, or environmental origin, biological context, and sensor affinity.

The tab also includes a molecular property-space plot and a VOC-to-biological-meaning network.

### 3. Mixture Representation Map

Real smells are mixtures. This tab simulates volatile chemical mixtures and displays them in learned representation space using PCA.

It shows how samples from different functional contexts cluster in a shared olfactory space:

- safe background,
- food spoilage,
- infection-like VOCs,
- environmental hazard,
- fermentation quality.

The purpose is to illustrate how AI could learn transferable olfactory representations across health, food, and environmental domains.

### 4. Sensor Array Simulator

This module simulates a cross-reactive electronic nose with multiple sensor channels:

- MOS-like sensors,
- electrochemical ammonia sensor,
- electrochemical hydrogen sulfide sensor,
- QCM-like sensor,
- PID-like sensor,
- optical-like sensor.

The simulator includes sensor response amplitude, dynamic rise and recovery traces, humidity effects, temperature effects, drift, and noise.

### 5. Biological Meaning Model

This module adds biological interpretation to volatile chemistry.

Instead of treating smell only as a human descriptor such as “sweet”, “rotten”, or “fruity”, it models functional biological labels such as attraction, avoidance, stress, metabolic shift, and danger or hazard.

The concept is inspired by biological olfaction, including simple model systems such as *Caenorhabditis elegans*, where volatile chemicals are converted into behavioural decisions.

### 6. AI Prediction and Uncertainty

This tab trains and evaluates a machine-learning model on simulated olfactory data.

The app uses a Random Forest classifier to predict the functional chemical context from sensor responses, molecular feature intensities, environmental metadata, and biological meaning scores.

It displays predicted class, confidence, uncertainty, class probabilities, confusion matrix, feature importance, and validation metrics.

### 7. OlfactoBot Plume Mapping

A systems-level simulation of an autonomous robot moving through a chemical plume.

The robot can search for a plume, reacquire a weak chemical signal, avoid excessive exposure, map the plume boundary, and maintain safe standoff from a chemical source.

Visual outputs include 2D robot path, plume-strength colour mapping, source localisation, action labels, and exposure timeline.

### 8. Health, Food, Environment Use Cases

A cross-domain map of potential applications:

#### Health

- breath VOC monitoring,
- skin or sebum VOCs,
- metabolic stress,
- infection-like VOCs,
- longitudinal disease-flare monitoring.

#### Food

- spoilage detection,
- freshness assessment,
- fermentation monitoring,
- food safety.

#### Environment

- indoor air quality,
- mould and microbial VOCs,
- industrial emissions,
- chemical hazard detection,
- exposome mapping.

### 9. Proposal Evaluator

A reviewer-style scoring tool for evaluating whether a digital olfaction project contributes to a general-purpose platform or remains a narrow point solution.

The evaluator scores proposals across dataset quality, sensor robustness, calibration strategy, representation learning value, cross-domain transfer, biological relevance, deployability, uncertainty handling, and milestone credibility.

It generates an overall portfolio-fit score, radar chart, and suggested reviewer questions.

---

## Molecular-to-systems framework

### Molecular level

The app represents:

- VOC structure,
- functional groups,
- volatility,
- polarity,
- microbial or metabolic origin,
- chemical class,
- expected sensor response.

### Mixture level

The app models VOC combinations, health-associated patterns, food spoilage patterns, microbial VOC signatures, and environmental hazard signatures.

### Sensor level

The app simulates cross-reactive sensor arrays, response amplitude, drift, noise, humidity, temperature, and dynamic rise and recovery.

### Biological level

The app adds attraction, avoidance, stress, metabolic meaning, and functional chemical interpretation.

### AI level

The app demonstrates representation learning, supervised classification, confidence scoring, uncertainty estimation, explainability, and feature importance.

### Systems level

The app connects predictions to robot movement, plume mapping, health decisions, food decisions, environmental decisions, and proposal evaluation.

---


## Requirements

```text
streamlit
numpy
pandas
plotly
scikit-learn
networkx
matplotlib
```

---

## Suggested repository structure

```text
olfactosense_studio/
├── app.py
├── README.md
├── requirements.txt
├── runtime.txt
├── run_app.sh
└── .gitignore
```

---

## Scientific interpretation

OlfactoSense Studio is not a diagnostic, safety, food-quality, or regulatory product. It is a research demonstration showing how digital olfaction could be integrated across chemistry, analytical measurement, sensor design, biological systems modelling, machine learning, uncertainty estimation, robotics, and translational decision-making.

The simulated datasets are intended for demonstration, teaching, technical design, and discussion.

---

## Example discussion

> OlfactoSense Studio frames artificial olfaction as an AI challenge that spans from molecules to whole systems. A volatile molecule alone does not define a smell. Smell emerges from a chemical mixture, the way sensors respond to it, the surrounding environmental conditions, its biological meaning, and the action or decision that follows. The app brings these layers together using simulated VOC chemistry, sensor-array signals, olfactory representation maps, biological meaning labels, uncertainty-aware AI predictions, robot plume mapping, and cross-domain examples across health, food, and environmental monitoring.


---

## Questions supported by the app


- Does the dataset support cross-domain transfer?
- Are sensor drift and environmental confounders explicitly modelled?
- Is there analytical chemistry ground truth?
- Are there functional or biological labels beyond subjective odour descriptors?
- Does the model generalise across sample types, laboratories, and use cases?
- Does the system estimate uncertainty?
- Can the sensing platform be deployed without rebuilding hardware for each new application?
- Are the milestones ambitious but measurable?
- What result would falsify the central technical claim?

---

## Limitations

This app uses simulated data. It does not claim to provide validated VOC biomarkers, real medical diagnosis, real food safety certification, real environmental hazard detection, physical sensor calibration, or clinical or regulatory performance.

Future versions could integrate real GC-MS or PTR-MS data, public VOC databases, breathomics datasets, microbial VOC libraries, sensor-array hardware streams, active-learning sample selection, graph neural networks for molecular representation, robot hardware interfaces, model cards, and dataset cards.

---

## Development roadmap

Additional modules included:

1. **GC-MS Raw Signal Viewer**  
   Interactive chromatogram and mass-spectrum visualisation.

2. **VOC Database Connector**  
   Integration with HMDB, mVOC, FooDB, Cancer Odor Database, and OlfactionBase.

3. **Graph Neural Network Molecular Encoder**  
   Molecular representation learning from SMILES or molecular graphs.

4. **Mixture Decoder**  
   AI model for predicting mixture similarity and functional olfactory meaning.

5. **Sensor Drift Laboratory**  
   Dedicated module for drift correction, calibration transfer, and batch-effect analysis.

6. **Open Benchmark Builder**  
   Tool for designing olfaction benchmark tasks and challenge datasets.

7. **Dataset Card Generator**  
   Automated metadata and governance documentation for olfactory datasets.

8. **Model Card Generator**  
   Explainability, intended-use, uncertainty, and limitation reporting for olfaction models.

9. **Hardware Readiness Assessment**  
   Structured scoring of sensor platforms against sensitivity, selectivity, portability, cost, throughput, drift, and chemical insight.

---

## Author

**Mark I. R. Petalcorin**  
Biomedical, biochemical, and AI-for-science researcher  
Founder, aAidea Ltd

Educational focus:

- molecular biology,
- biochemistry,
- DNA repair,
- cancer metabolism,
- mitochondrial bioenergetics,
- analytical biochemistry,
- AI for chemistry,
- machine learning for biomedical discovery,
- scientific software and decision-support systems.

---

## License

This project is intended as a research, education, and demonstration. 

```text
MIT License
```

---

## Citation

If referencing this prototype, you may cite it as:

```text
Petalcorin, M. I. R. (2026). OlfactoSense Studio: A Molecular-to-Systems AI Platform for Digital Olfaction. GitHub repository: https://github.com/mpetalcorin/olfactosense_studio
```

---

## Status

Current version: **v1.0 prototype**

This version includes simulated VOC atlas, simulated mixture representation learning, simulated sensor-array dynamics, biological meaning model, random forest prediction, uncertainty score, plume-mapping robot simulation, use-case map, and evaluator.

