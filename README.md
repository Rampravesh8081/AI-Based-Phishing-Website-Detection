# AI-Based Phishing Website Detection System Using Machine Learning

**IBM SkillsBuild Cyber Security BootCamp 2026 — BharatCares**
**Developed by: Rampravesh**

---

## 📋 Project Description

A machine learning-powered cybersecurity application that analyses URL structural features to classify websites as **Legitimate** or **Phishing**. The system extracts 16 URL-based features, trains three classifiers (Logistic Regression, Decision Tree, Random Forest), selects the best model, and serves predictions through an interactive Streamlit dashboard.

---

## ✨ Main Features

| Feature | Description |
|---|---|
| URL Input | Enter any URL for instant analysis |
| Feature Extraction | 16 structural URL features extracted automatically |
| ML Classification | Three models trained and compared |
| Confidence Score | Probability estimate with limitation disclaimer |
| Feature Breakdown | Per-feature risk assessment table |
| Performance Charts | Confusion matrix, ROC curves, feature importance |
| Safety Tips | 10 cybersecurity awareness tips |
| Offline Operation | No external API or internet connection required |

---

## 🛠️ Technologies Used

| Category | Libraries / Tools |
|---|---|
| Language | Python 3.9+ |
| Machine Learning | scikit-learn 1.3+ |
| Data Processing | pandas, numpy, scipy |
| Visualisation | matplotlib, seaborn |
| Web Dashboard | Streamlit |
| Model Persistence | joblib |
| Notebook | Jupyter |
| Version Control | Git / GitHub |

---

## 📊 Dataset Source

**Primary:** UCI Phishing Websites Dataset
- URL: https://archive.ics.uci.edu/dataset/327/phishing+websites
- Instances: 11,055
- Features: 30
- Licence: CC BY 4.0 — free to use with attribution

**Fallback:** A reproducible synthetic dataset based on URL-pattern statistics from published phishing research is used automatically when the UCI file is not present.

To use the UCI dataset:
1. Download from https://archive.ics.uci.edu/dataset/327/phishing+websites
2. Extract the `.arff` file and rename it to `phishing.arff`
3. Place it in the `dataset/` folder

---

## ⚙️ Installation Instructions

### Prerequisites
- Python 3.9 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/Phishing-Website-Detection.git
cd Phishing-Website-Detection

# 2. (Recommended) Create a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1 — Train the Model (required once)
```bash
python train_model.py
```
This will:
- Load the dataset (UCI or synthetic)
- Train and evaluate all three models
- Save the best model to `models/model.pkl` and `model.pkl`
- Generate performance charts in `reports/`

### Step 2 — Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at http://localhost:8501

### Step 3 — Run the Jupyter Notebook (optional)
```bash
jupyter notebook Rampravesh_PhishingDetection.ipynb
```

---

## 📁 Project Folder Structure

```
Phishing-Website-Detection/
│
├── app.py                              # Streamlit web dashboard
├── train_model.py                      # Training pipeline
├── feature_extraction.py               # URL feature extractor
├── Rampravesh_PhishingDetection.ipynb  # Jupyter notebook
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── model.pkl                           # Trained model (generated)
│
├── dataset/
│   ├── dataset_info.txt               # Dataset description & download instructions
│   └── phishing.arff                  # UCI dataset (place here after download)
│
├── models/
│   ├── model.pkl                      # Best trained model
│   └── model_metadata.json           # Model metrics & info
│
├── reports/
│   ├── evaluation.txt                 # Full text evaluation report
│   ├── confusion_matrix.png          # Confusion matrix chart
│   ├── roc_curve.png                 # ROC curves chart
│   ├── feature_importance.png        # Feature importance chart
│   └── model_comparison.png         # Model comparison chart
│
└── images/
    └── (dashboard screenshots)
```

---

## 🤖 Model Information

| Metric | Value (varies by dataset) |
|---|---|
| Algorithm | Random Forest (typically best) |
| Features | 16 URL structural features |
| Train/Test Split | 80% / 20% |
| Cross-validation | 5-fold stratified CV |
| Evaluation Metrics | Accuracy, Precision, Recall, F1, AUC-ROC |

### Features Extracted

| Feature | Description |
|---|---|
| url_length | Total character count |
| hostname_length | Hostname length |
| path_length | URL path length |
| num_dots | Count of dots |
| num_hyphens | Count of hyphens |
| num_at | Count of @ symbols |
| num_question_marks | Count of ? symbols |
| num_equal | Count of = symbols |
| num_slashes | Count of / symbols |
| num_percent | Count of % (URL encoding) |
| has_https | 1 if HTTPS, else 0 |
| has_ip | 1 if IP address used as hostname |
| has_suspicious_tld | 1 if TLD is known-suspicious |
| num_subdomains | Number of subdomain levels |
| has_phishing_keyword | 1 if URL contains phishing keywords |
| double_slash_in_path | 1 if // appears after scheme |

---

## ⚠️ Limitations

- Analyses URL structure only — does **not** visit or render the website
- A structurally normal URL can still host malicious content
- Confidence score reflects model certainty, not guaranteed website safety
- Performance on the synthetic dataset may not generalise to all real-world URLs
- Always use multiple layers of security for sensitive browsing

---

## 👤 Author

**Rampravesh**
IBM SkillsBuild Cyber Security BootCamp 2026
Conducted by BharatCares

---

## 📄 Licence

This project is intended for educational purposes as part of the IBM SkillsBuild
Cyber Security BootCamp 2026. All third-party datasets and libraries retain their
original licences.
