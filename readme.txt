Consumer Complaint Text Classification
Company Assignment: Kaiburr - Data Science Position
Date: October 2025
📌 Project Overview
This project implements a multi-class text classification system to automatically categorize consumer complaints into four categories:

0: Credit reporting, repair, or other
1: Debt collection
2: Consumer Loan
3: Mortgage

🎯 Objective
Build and evaluate machine learning models to classify consumer complaints from the Consumer Complaint Database with high accuracy and reliability.
📊 Dataset

Source: U.S. Consumer Financial Protection Bureau
Link: https://catalog.data.gov/dataset/consumer-complaint-database
Features Used: Consumer complaint narrative text and Product category

🚀 Project Workflow
1. Exploratory Data Analysis (EDA)

Class distribution analysis
Text length and word count statistics
Visualization of data patterns
Missing value analysis

2. Text Preprocessing

Lowercasing and tokenization
Removal of URLs, emails, special characters
Stopword removal
Lemmatization
Feature extraction using TF-IDF

3. Model Selection
Five classification models evaluated:

Logistic Regression
Multinomial Naive Bayes
Linear SVM
Random Forest Classifier
XGBoost Classifier

4. Model Evaluation

Accuracy, Precision, Recall, F1-Score
Confusion Matrix
Classification Report
Cross-validation

5. Model Optimization

Hyperparameter tuning using GridSearchCV
Feature importance analysis
Best model selection

6. Deployment Ready

Prediction function for new complaints
Model and vectorizer saved for production use

🛠️ Installation & Setup
Prerequisites
bashPython 3.8+
Install Dependencies
bashpip install pandas numpy matplotlib seaborn scikit-learn nltk xgboost
Download NLTK Data
The script automatically downloads required NLTK data, but you can also do it manually:
pythonimport nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
📁 Project Structure
consumer-complaint-classification/
│
├── complaint_classifier.py          # Main Python script
├── consumer_complaints.csv           # Dataset (download separately)
├── README.md                         # Project documentation
│
├── Output Files (Generated):
│   ├── eda_visualizations.png        # EDA plots
│   ├── model_comparison.png          # Model performance comparison
│   ├── confusion_matrix.png          # Confusion matrix heatmap
│   ├── feature_importance.png        # Feature importance plot
│   ├── complaint_classifier_model.pkl # Trained model
│   ├── tfidf_vectorizer.pkl          # TF-IDF vectorizer
│   └── model_performance_report.txt  # Detailed metrics report
▶️ How to Run

Download the dataset:

Visit: https://catalog.data.gov/dataset/consumer-complaint-database
Download the CSV file
Save as consumer_complaints.csv in project folder


Update file path in script:

python   df = pd.read_csv('consumer_complaints.csv', low_memory=False)

Run the script:

bash   python complaint_classifier.py

Check outputs:

View generated visualizations (PNG files)
Review performance report (TXT file)
Use saved model for predictions



📈 Results
The model achieves:

Best Model: [Model name will be determined after running]
F1-Score: ~0.XX (varies based on data)
Accuracy: ~XX%

Detailed results available in model_performance_report.txt
🔮 Making Predictions
Use the trained model to classify new complaints:
pythonimport pickle

# Load model and vectorizer
with open('complaint_classifier_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Predict
new_complaint = "Your complaint text here..."
# Apply preprocessing and predict
# (See predict_complaint_category function in script)
📊 Visualizations Generated

EDA Visualizations - Class distribution, text statistics
Model Comparison - Performance metrics across all models
Confusion Matrix - Prediction accuracy per class
Feature Analysis - Important words/features

🔑 Key Features

✅ Complete data science pipeline
✅ Multiple model comparison
✅ Advanced text preprocessing
✅ Hyperparameter optimization
✅ Production-ready code
✅ Comprehensive documentation
✅ Saved models for deployment

📝 Technical Stack

Language: Python 3.8+
ML Libraries: scikit-learn, XGBoost
NLP: NLTK
Data Processing: Pandas, NumPy
Visualization: Matplotlib, Seaborn

🎓 Learning Outcomes

Text preprocessing and feature engineering
Multi-class classification techniques
Model evaluation and comparison
Hyperparameter tuning
Production deployment preparation

👤 Author
SANATHRAJ S
sanathrajs2002@gmail.com

📄 License
This project is created for educational and assignment purposes.
🙏 Acknowledgments

Dataset: U.S. Consumer Financial Protection Bureau
Company: Kaiburr
Purpose: Placement Assignment



Note: Ensure you have downloaded the dataset before running the script. The dataset is publicly available but not included in this repository due to size constraints.

