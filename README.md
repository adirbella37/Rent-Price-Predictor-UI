# Rent Price Predictor UI

A simple web application for predicting **monthly apartment rent prices in Tel Aviv**.  
The project is based on a **cleaned dataset of rental listings**, and uses an **Elastic Net regression model** trained with scikit-learn.  
Users can enter apartment details (size, rooms, address, etc.) into a web form and instantly get a predicted rent price.

---

## 🚀 Features
- Interactive **HTML form** for entering apartment details
- Backend built with **Flask (Python)**
- **Elastic Net regression model** trained on Tel Aviv rental data
- Preprocessing pipeline for handling new input data
- Prediction returned instantly on the web page

---

## 📂 Project Structure


| File/Folder              | Description                        |
|--------------------------|------------------------------------|
| `api.py`                 | Flask app (routes + prediction API)|
| `assets_data_prep.py`    | Data preparation functions         |
| `model_training.py`      | Elastic Net model training script  |
| `neigh_dist_medians.pkl` | Helper file for distances          |
| `trained_model.pkl`      | Trained Elastic Net model          |
| `requirements.txt`       | Python dependencies                |
| `templates/index.html`   | HTML UI form                       |
| `README.md`              | Project documentation              |



---

## 🛠 Installation & Setup

1. **Clone this repository:**
   ```bash
   git clone https://github.com/adirbella37/Rent-Price-Predictor-UI.git
   cd Rent-Price-Predictor-UI
    ```
   
2. **Create and activate a virtual environment:**
   
   python -m venv

 - On Windows: venv\Scripts\activate
     
 - On Mac/Linux: source venv/bin/activate

4. **Install dependencies:**
   
   pip install -r requirements.txt

5. **Run the Flask app:**
   
   python api.py

6. **Open the app in your browser:**
   
   http://127.0.0.1:5000

## 📸 Demo

## 📜 License
This project is licensed under the MIT License.
