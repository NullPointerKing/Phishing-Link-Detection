# Phishing Link Detection System 🚀

## 📌 Overview
The **Phishing Link Detection System** is a machine learning-based application designed to classify URLs as **phishing** or **legitimate**. It aims to enhance cybersecurity defenses by identifying malicious links in real-time, reducing the risk of phishing attacks.

## ⚙️ Key Features
- **🔍 Feature Extraction:** URL length, number of subdomains, special characters, HTTPS usage, and phishing keywords.
- **🤖 Machine Learning Models:** Utilizes **Random Forest** and **Gradient Boosting** for accurate classification.
- **⚖️ Class Imbalance Handling:** Oversampling techniques and weighted models for balanced performance.
- **📊 Analytics:** Descriptive & predictive analytics for uncovering phishing trends.
- **🌐 Browser Extension (Upcoming):** Real-time detection while browsing.
- **🖥️ Web Interface:** Flask-based web application for easy interaction.

## 🚀 Project Architecture
1. **Input:** User submits a URL.
2. **Feature Extraction:** Extracts relevant URL characteristics.
3. **Model Prediction:** Trained ML model classifies the URL.
4. **Output:** Displays whether the URL is phishing or legitimate.

## 🧰 Tech Stack
- **Backend:** Python, Flask, scikit-learn
- **Frontend:** HTML, CSS
- **Database:** MySQL

---

## 📥 Installation
```bash
# Clone the repository
git clone https://github.com/your-username/phishing-link-detection.git

# Navigate to the project directory
cd phishing-link-detection

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🖥️ Usage
1. **Start the Flask server:**
```bash
python app.py
```
2. **Access the Web Interface:**  
Open your browser and go to: `http://127.0.0.1:5000/`

3. **Submit a URL:**  
Enter the URL you want to check and click **"Detect"** to see the results.

## 🗃️ Dataset
- **Phishing URLs:** Collected from OpenPhish, PhishTank, etc.
- **Legitimate URLs:** Sourced from Alexa Top Sites.

## 📊 Model Training
The Random Forest model is trained with:
- **Accuracy:** 80% on test data
- **Cross-Validation:** Stratified K-Fold for balanced evaluation

## 🔐 Security Measures
- Handles real-time URL validation.
- Reduces dependency on static blacklists.

## 📈 Future Enhancements
- Integrate browser extension for real-time protection.
- Deploy as a cloud-based API for scalability.

## 🤝 Contributing
Contributions are welcome! Feel free to fork the repository and submit pull requests.

## 📜 License
This project is licensed under the [MIT License](LICENSE).

## 🙌 Acknowledgments
- Datasets from OpenPhish, PhishTank, and Alexa.
- Libraries: scikit-learn, Flask, MySQL, Seaborn, Matplotlib.

