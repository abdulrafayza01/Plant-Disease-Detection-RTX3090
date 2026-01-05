import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

# --- 1. SETTINGS & MODEL LOAD ---
app = Flask(__name__, 
            template_folder="../templates", 
            static_folder="../static")

app.config['UPLOAD_FOLDER'] = '../static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Naya v2 model load karein
MODEL_PATH = r'D:\AI-ENV\Smart Agriculture Plant Disease Detection Using Leaf Images\notebooks\plant_disease_model_final_v1.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Alphabetical order list (verified)
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# --- 2. PREDICTION LOGIC (Top-2 Analysis) ---
def model_predict(img_path, model):
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # MobileNetV2 preprocessing (Crucial for 97%+ accuracy)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    preds = model.predict(img_array)[0]
    top_indices = preds.argsort()[-2:][::-1] # Sabse uchi 2 predictions
    
    main_pred = class_names[top_indices[0]].replace('___', ' ').replace('_', ' ')
    main_conf = preds[top_indices[0]] * 100
    
    second_pred = class_names[top_indices[1]].replace('___', ' ').replace('_', ' ')
    
    # Agar model 75% se kam sure ho, toh "Possible" option bhi dikhayen
    if main_conf < 75:
        display_text = f"{main_pred} (Likely) or {second_pred}"
    else:
        display_text = main_pred
        
    return display_text, f"{main_conf:.2f}%", main_pred

# --- 3. ROUTES ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'})
    
    file = request.files['file']
    if file.filename:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        prediction_text, confidence, raw_name = model_predict(file_path, model)

        return jsonify({
            'prediction': prediction_text,
            'confidence': confidence,
            'description': f"Our AI model analyzed the leaf patterns and identified it as {raw_name}.",
            'solution': "Ensure proper ventilation, remove infected leaves, and use targeted fungicides/pesticides."
        })
    return jsonify({'error': 'Upload failed'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)