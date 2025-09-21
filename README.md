# Prediction Model for Pharmaceutical Formulation

This repository contains the full code for building **machine-learning and deep-learning models** to predict the **disintegration time of fast-disintegrating tablets (FDTs)** from formulation and process variables.

The project explores multiple algorithms and compares their performance to identify the most accurate predictor of tablet disintegration time.

---

## 📰 Related Publications
The methods and findings are described in these peer-reviewed articles:

* Momeni M. *et al.* **“A prediction model based on artificial intelligence techniques for disintegration time and hardness of fast disintegrating tablets in pre-formulation tests”**  
  [BMC Medical Informatics and Decision Making (2024)](https://link.springer.com/article/10.1186/s12911-024-02485-4)

* Momeni M. *et al.* **“Dataset development of pre-formulation tests on fast disintegrating tablets (FDT): data aggregation”**  
  [BMC Research Notes (2023)](https://link.springer.com/article/10.1186/s13104-023-06416-w)

Please cite these works if you use this code or data.

---

## 📊 Dataset
The formulation dataset is openly available on Zenodo:

[➡️ Zenodo Record 15525554](https://zenodo.org/records/15525554)

It includes excipient composition, processing parameters, and experimentally measured disintegration times.



## 🔑 Key Features of the Project
* **Hybrid Modeling:** Traditional ML (Random Forest, XGBoost, SVM, etc.) and deep learning models.
* **Output Transformation:** Continuous disintegration times were **categorized** into discrete classes to enable classification models alongside regression.
* **Model Selection:** Extensive comparison to determine the best-performing predictive model.


## 🧩 Project Structure
Prediction-Model-for-Pharmaceutical-Formulation/
├─ data/ # data handling and preprocessing scripts
├─ src/ # core Python modules (training, evaluation)
├─ requirements.txt # Python dependencies
└─ README.md



## ⚙️ Installation
```bash
git clone https://github.com/mehri-momeny/Prediction-Model-for-Pharmaceutical-Formulation.git
cd Prediction-Model-for-Pharmaceutical-Formulation
pip install -r requirements.txt
🚀 Usage
Download the dataset from the Zenodo link and place it in the data/ directory (or update paths accordingly).

Train/Evaluate models using scripts in src/ to compare regression vs. classification approaches.

📝 Citation
If you use this repository, please cite:


Momeni, M., Afkanpour, M., Rakhshani, S. et al. A prediction model based on artificial intelligence techniques for disintegration time and hardness of fast disintegrating tablets in pre-formulation tests. BMC Med Inform Decis Mak 24, 88 (2024).
https://doi.org/10.1186/s12911-024-02485-4
and


Momeni, M., Rakhshani, S., Abbaspour, M. et al. Dataset development of pre-formulation tests on fast disintegrating tablets (FDT): data aggregation. BMC Res Notes 16, 131 (2023).
https://doi.org/10.1186/s13104-023-06416-w



