Smart Agriculture: Plant Disease Detection Using Deep Learning
🚀 Project Overview
This project addresses the critical challenge of agricultural yield loss due to plant diseases. We have developed an automated AI solution that identifies 38 different categories of healthy and diseased plant leaves with high precision, enabling farmers to take timely action.

🔗 Repository Link
GitHub Repo: https://github.com/abdulrafayza01/Plant-Disease-Detection-RTX3090.git

📊 Key Performance Metrics
Validation Accuracy: 92.85% (Approx. 93%)

Weighted Average F1-Score: 0.93

Dataset Size: 54,305 images from the PlantVillage Dataset

Training Hardware: Optimized using NVIDIA GeForce RTX 3090 GPU for high-performance deep learning

🏗️ Methodology & Architecture
The system utilizes a custom Convolutional Neural Network (CNN) architecture designed specifically for plant pathology image recognition.

Input Layer: 256x256 RGB leaf images

Core: Multi-layer CNN for feature extraction (patterns, spots, and textures)

Output: Softmax prediction across 38 distinct classes

💻 Tech Stack
Deep Learning Framework: TensorFlow / Keras

Web Interface: Streamlit (Real-time Dashboard)

Language: Python 3.10

📈 Results & Visualizations
The model successfully identifies complex diseases like Tomato Late Blight. Detailed metrics including Accuracy/Loss curves and Confusion Matrices are documented in the plant_disease.ipynb notebook.

🛠️ Installation & Quick Start

Clone the repository:
Bash
git clone https://github.com/abdulrafayza01/Plant-Disease-Detection-RTX3090.git

Install Dependencies:
Bash
pip install -r requirements.txt

Launch the Application:
Bash
streamlit run app.py