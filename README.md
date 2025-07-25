# AutoMLite  
*A lightweight and modular AutoML pipeline for tabular data*  

AutoMLite automates the **end-to-end machine learning workflow**—from data preprocessing to model explainability—with minimal code. It is designed to be **resource-efficient** and highly **extensible**, making it perfect for laptops and small servers.  

---

## **Key Features**  
✔ **Automated Data Preprocessing**  
- Handles missing values, scaling, and categorical encoding automatically.  

✔ **Feature Engineering**  
- Polynomial feature generation, feature selection, and more.  

✔ **Model Selection & Training**  
- Supports multiple models:  
  - Logistic Regression  
  - Random Forest  
  - SVM  
  - Multi-Layer Perceptron (MLP)  

✔ **Hyperparameter Optimization**  
- Efficient tuning with **Optuna**.  

✔ **Evaluation & Reporting**  
- Standard ML metrics (Accuracy, Precision, Recall, F1-score)  
- Confusion matrix  

✔ **Explainability**  
- **SHAP-based feature importance** and model interpretation.  

✔ **Resource-Efficient**  
- Optimized for systems with **8GB RAM or more**.  

✔ **Extensible & Modular**  
- Easily add new models or preprocessing steps.  

---

## **Installation**  

```bash
# 1. Clone the repository
git clone https://github.com/siddharth23k/AutoMLite.git

# 2. Navigate into the project folder
cd AutoMLite

# 3. Create a virtual environment
python3 -m venv venv

# 4. Activate the virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 5. Install dependencies
pip install -r requirements.txt
