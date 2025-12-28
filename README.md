# Student Performance Prediction System
## 🚀 End-to-End Machine Learning Pipeline

This project is a complete machine learning system designed to predict student performance (Pass/Fail) based on academic metrics. It features a robust Python backend, a polished Streamlit frontend, and a production-grade directory structure.

## 🌟 Key Features
*   **Modern UI**: High-contrast, dark-themed dashboard with interactive charts (Plotly).
*   **End-to-End Pipeline**: Includes data ingestion, cleaning, feature engineering, training, and inference.
*   **Live History**: Sidebar tracks your recent prediction sessions in real-time.
*   **Explainable AI**: Provides human-readable explanations for every prediction.

## 🛠️ Tech Stack
*   **Language**: Python 3.9+
*   **Frontend**: Streamlit
*   **Data Processing**: Pandas, NumPy
*   **Machine Learning**: Scikit-Learn
*   **Visualization**: Plotly, Matplotlib, Seaborn

## 📂 Project Structure
```
student-performance-system/
├── app/
│   └── app.py              # Main dashboard application
├── src/
│   ├── data/               # Data loading & preprocessing scripts
│   ├── features/           # Feature engineering logic
│   └── models/             # Model training and prediction logic
├── data/                   # Raw and processed datasets
├── models/                 # Serialized model files (.pkl)
├── requirements.txt        # Python dependencies
└── README.md
```

## 🚀 How to Run Locally

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/student-performance-system.git
    cd student-performance-system
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Launch the App**
    ```bash
    streamlit run app/app.py
    ```