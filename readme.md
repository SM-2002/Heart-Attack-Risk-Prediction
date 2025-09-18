# 🩺 Heart Attack Risk Prediction using Machine Learning

This project predicts the risk level of heart disease in patients using machine learning models trained on medical diagnostic data.

---

## 📊 Dataset

- Contains patient medical features:
  - `age`: Age of the patient  
  - `sex`: Gender (1 = male, 0 = female)  
  - `cp`: Chest pain type  
  - `trestbps`: Resting blood pressure  
  - `chol`: Serum cholesterol  
  - `fbs`: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)  
  - `restecg`: Resting electrocardiographic results  
  - `thalach`: Maximum heart rate achieved  
  - `exang`: Exercise induced angina (1 = yes, 0 = no)  
  - `oldpeak`: ST depression induced by exercise  
  - `slope`: Slope of the ST segment  
  - `ca`: Number of major vessels (0–4)  
  - `thal`: Thalassemia  
  - `num`: Target variable — heart disease severity (0–4)

---

## ⚙️ Steps Performed

1. **Data Preprocessing**
   - Handled missing values
   - Encoded categorical features
   - Performed feature selection using **:contentReference[oaicite:0]{index=0}** → selected top 6 features
   - Balanced dataset using oversampling (via **:contentReference[oaicite:1]{index=1}**) → `x_res` and `y_res`

2. **Exploratory Data Analysis**
   - Correlation matrix
   - Pairplots using **:contentReference[oaicite:2]{index=2}** for selected features

3. **Model Training**
   - Gaussian Naive Bayes   
|  - Random Forest Classifier    
|  - ANN (TensorFlow/Keras)      
|  - Support Vector Classifier   
|  - K-Nearest Neighbors

4. **Evaluation**
   - Accuracy
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - Matthews Correlation Coefficient (MCC)
   - Classification report (Precision, Recall, F1-score)

---

## 📈 Results

| Model                      | Accuracy |
|-----------------------------|-----------|
| Gaussian Naive Bayes        | 0.63      |
| Random Forest Classifier    | 0.91      |
| ANN (TensorFlow/Keras)      | 0.47      |
| Support Vector Classifier   | 0.65      |
| K-Nearest Neighbors          | 0.86      |

---

## 💻 Tech Stack

- **Language**: :contentReference[oaicite:9]{index=9}  
- **Libraries**: :contentReference[oaicite:10]{index=10}, :contentReference[oaicite:11]{index=11}, :contentReference[oaicite:12]{index=12}, :contentReference[oaicite:13]{index=13}  
- **ML Frameworks**: :contentReference[oaicite:14]{index=14}, :contentReference[oaicite:15]{index=15}, :contentReference[oaicite:16]{index=16} / :contentReference[oaicite:17]{index=17}

---

## 📌 Future Improvements

- Hyperparameter tuning
- Trying more advanced models 
- Use cross-validation for robust evaluation
- Build a web interface to predict heart attack risk

---



