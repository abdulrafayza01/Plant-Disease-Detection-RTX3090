Smart Agriculture: Plant Disease Detection Using Deep Learning

🚀 Project Overview
This project addresses the critical challenge of agricultural yield loss due to plant diseases. We have developed an automated AI solution that identifies 38 different categories of healthy and diseased plant leaves with high precision (93%), enabling farmers to take timely action.

🔗 Repository Link
GitHub Repo: https://github.com/abdulrafayza01/Plant-Disease-Detection-RTX3090.git

📊 Key Performance Metrics

Validation Accuracy: 92.85% (Approx. 93%)

Weighted Average F1-Score: 0.93

Dataset Size: 54,305 images from the PlantVillage Dataset

Training Hardware: Optimized using NVIDIA GeForce RTX 3090 GPU.

🏗️ Methodology & Architecture

The system utilizes a custom Convolutional Neural Network (CNN) architecture designed specifically for plant pathology.

Input Layer: 256x256 RGB leaf images.

Core: Multi-layer CNN for feature extraction of patterns, spots, and textures.

Output: Softmax prediction across 38 distinct classes.

💻 Tech Stack

Deep Learning Framework: TensorFlow / Keras.

Web Interface: Streamlit (Real-time Dashboard).

Optimization: oneDNN optimized for hardware acceleration.

⚡ Easy Execution (Gradio-Style)

We have optimized the execution process for developers. You no longer need to type complex commands.

VS Code Debugger: Simply open app.py and press F5. The project is pre-configured with launch.json to automatically start the Streamlit server.

Direct Script Run: You can also run it like a standard Python script: python app.py.

📈 Results & Visualizations
The model successfully identifies complex diseases like Tomato Late Blight. Detailed training history (Accuracy/Loss curves) is documented in the plant_disease.ipynb notebook.

🛠️ Installation

Clone the repository:
Bash
git clone https://github.com/abdulrafayza01/Plant-Disease-Detection-RTX3090.git

Install Dependencies:
Bash
pip install -r requirements.txt