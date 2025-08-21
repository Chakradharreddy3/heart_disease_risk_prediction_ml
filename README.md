Heart Disease Risk Prediction App:

This project predicts the risk of heart disease (High Risk / Low Risk) using a Random Forest Machine Learning model. Users can input patient details via a web app built with Streamlit, and the app returns a risk prediction based on the trained model.

1.Project Overview

  Heart disease is one of the leading causes of death worldwide. Early detection can save lives. This project uses a dataset of heart health parameters to train several machine learning models and evaluate their performance.
  Models Tested:
    1.Logistic Regression
    2.Support Vector Machine (SVM)
    3.Random Forest
    4.K-Nearest Neighbors (KNN)
    After comparing accuracy, precision, recall, and F1-score for both High Risk and Low Risk classes, the Random Forest model was selected as the best-performing algorithm and is used in this app.

2.Technologies Used
  Python
  Pandas, NumPy
  Scikit-learn (Machine Learning)
  Streamlit (Web App Deployment)
  Pickle (Model Serialization)

3.Model Performance
  Best Model: Random Forest Classifier
  Evaluated on metrics: Accuracy, Precision, Recall, F1-Score (both classes)
  Correctly classifies High Risk and Low Risk patients based on input features.

4.Installation
  4.1: Clone the repository:
       git clone https://github.com/yourusername/heart-disease-risk-app.git
  
  4.2: Navigate to the project folder:
       cd heart-disease-risk-app


  4.3:Install required packages:
      pip install -r requirements.txt

4.4: Run the Streamlit app:
    streamlit run app.py

5.How to Use the App
    Open the app in your browser (after running streamlit run app.py).
    Enter patient details using sliders and dropdowns.
    Click Predict Risk.
    The app will display whether the patient is High Risk or Low Risk.
6.Dataset
    The dataset contains health parameters and target labels (0 = Normal / Low Risk, 1 = Heart Disease / High Risk).
    Data preprocessing includes scaling numeric features and using categorical values as-is.
    
7.Files in Repository
  app.py → Streamlit web app
  best_heart_model_final.pkl → Trained Random Forest model
  scaler.pkl → StandardScaler used for numerical features
  requirements.txt → Python dependencies

NOTE:This project is for educational purposes only. It should not replace professional medical advice.



