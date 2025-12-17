import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page Configuration
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")

# Classes List
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

# Disease Information and Solutions Dictionary
disease_info = {
    'Apple___Apple_scab': {'description': 'Dark, olive-colored spots appear on leaves and fruit.', 'solution': 'Prune affected branches and use appropriate fungicides.'},
    'Apple___Black_rot': {'description': 'Circular brown spots on leaves; fruit turns black and rots.', 'solution': 'Remove dead wood and infected fruit; apply protective sprays.'},
    'Apple___Cedar_apple_rust': {'description': 'Bright orange spots on the upper surface of leaves.', 'solution': 'Remove nearby cedar trees or use rust-resistant varieties.'},
    'Apple___healthy': {'description': 'The apple tree is healthy and growing well.', 'solution': 'Continue regular monitoring and balanced fertilization.'},
    'Blueberry___healthy': {'description': 'No signs of disease or pests found on the blueberry leaves.', 'solution': 'Maintain acidic soil pH and keep the plant well-watered.'},
    'Cherry_(including_sour)___Powdery_mildew': {'description': 'White, powdery fungal growth on leaves and twigs.', 'solution': 'Improve air circulation and use sulfur-based fungicides.'},
    'Cherry_(including_sour)___healthy': {'description': 'The cherry plant is healthy.', 'solution': 'Keep the area clean and follow standard watering schedules.'},
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {'description': 'Long, rectangular grey lesions on the leaves.', 'solution': 'Use resistant hybrids and practice crop rotation.'},
    'Corn_(maize)___Common_rust_': {'description': 'Small, cinnamon-brown powdery pustules on leaf surfaces.', 'solution': 'Apply fungicides early if infection levels are high.'},
    'Corn_(maize)___Northern_Leaf_Blight': {'description': 'Large, cigar-shaped greyish-green lesions on leaves.', 'solution': 'Manage crop residue and use resistant corn varieties.'},
    'Corn_(maize)___healthy': {'description': 'The corn plant shows no signs of disease.', 'solution': 'Ensure proper nitrogen levels and weed control.'},
    'Grape___Black_rot': {'description': 'Small brown spots on leaves; berries shrivel and turn black.', 'solution': 'Pick and destroy infected berries; keep vines well-pruned.'},
    'Grape___Esca_(Black_Measles)': {'description': 'Dark stripes on leaves and small dark spots on fruit skin.', 'solution': 'Avoid large pruning wounds and remove infected wood.'},
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {'description': 'Irregular brown spots on leaves causing early leaf drop.', 'solution': 'Apply fungicides after harvest and clear fallen leaves.'},
    'Grape___healthy': {'description': 'The grapevine is healthy.', 'solution': 'Prune annually and provide proper trellis support.'},
    'Orange___Haunglongbing_(Citrus_greening)': {'description': 'Yellowing of veins and blotchy mottling of leaves.', 'solution': 'Control the psyllid insect and remove infected trees immediately.'},
    'Peach___Bacterial_spot': {'description': 'Small water-soaked spots that turn into holes (shot-holes).', 'solution': 'Apply copper sprays during dormancy and avoid excess nitrogen.'},
    'Peach___healthy': {'description': 'The peach tree is in good health.', 'solution': 'Thin out fruits for better size and monitor for pests.'},
    'Pepper,_bell___Bacterial_spot': {'description': 'Small, irregular raised spots on the underside of leaves.', 'solution': 'Use disease-free seeds and avoid overhead irrigation.'},
    'Pepper,_bell___healthy': {'description': 'The bell pepper plant is healthy.', 'solution': 'Provide consistent moisture and use mulch.'},
    'Potato___Early_blight': {'description': 'Dark spots with concentric rings on older leaves.', 'solution': 'Rotate crops and ensure plants have adequate nutrients.'},
    'Potato___Late_blight': {'description': 'Dark, water-soaked patches on leaves that turn brown.', 'solution': 'Destroy infected plants and use certified disease-free tubers.'},
    'Potato___healthy': {'description': 'The potato plant is healthy.', 'solution': 'Hilling up soil around plants and monitoring for beetles.'},
    'Raspberry___healthy': {'description': 'The raspberry plant is healthy.', 'solution': 'Prune old canes after harvest and keep weeds away.'},
    'Soybean___healthy': {'description': 'The soybean plant is healthy.', 'solution': 'Maintain proper spacing for air flow.'},
    'Squash___Powdery_mildew': {'description': 'White powdery spots on the surface of leaves and stems.', 'solution': 'Plant in full sun and use resistant varieties or neem oil.'},
    'Strawberry___Leaf_scorch': {'description': 'Purple to brown patches on leaves that eventually dry up.', 'solution': 'Remove old leaves and avoid over-crowding.'},
    'Strawberry___healthy': {'description': 'The strawberry plant is healthy.', 'solution': 'Use straw mulch to keep berries off the ground.'},
    'Tomato___Bacterial_spot': {'description': 'Small brown spots with yellow halos on leaves and fruit.', 'solution': 'Use copper-based sprays and avoid handling wet plants.'},
    'Tomato___Early_blight': {'description': 'Brown spots with rings on lower leaves, causing yellowing.', 'solution': 'Remove lower leaves and apply mulch to prevent soil splashing.'},
    'Tomato___Late_blight': {'description': 'Large grey-brown patches on leaves; spreads fast in wet weather.', 'solution': 'Improve drainage and apply fungicides immediately.'},
    'Tomato___Leaf_Mold': {'description': 'Pale green or yellow spots on top of leaves, mold below.', 'solution': 'Increase ventilation and reduce humidity.'},
    'Tomato___Septoria_leaf_spot': {'description': 'Small circular spots with grey centers and dark borders.', 'solution': 'Avoid overhead watering and rotate crops.'},
    'Tomato___Spider_mites Two-spotted_spider_mite': {'description': 'Fine yellow speckling on leaves and tiny webs.', 'solution': 'Increase humidity or use insecticidal soap/neem oil.'},
    'Tomato___Target_Spot': {'description': 'Small brown spots that develop concentric rings.', 'solution': 'Maintain good air circulation and use fungicides.'},
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {'description': 'Leaves curl upward and turn yellow; growth is stunted.', 'solution': 'Control whiteflies using reflective mulch or traps.'},
    'Tomato___Tomato_mosaic_virus': {'description': 'Mottled green and yellow patterns on leaves.', 'solution': 'Remove infected plants and practice good hygiene.'},
    'Tomato___healthy': {'description': 'The tomato plant is perfectly healthy.', 'solution': 'Continue consistent watering and provide support.'}
}

# Model Loading
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(r'E:\Rafay\Smart Agriculture Plant Disease Detection Using Leaf Images\plant_disease_model.h5')

model = load_model()

# UI
st.title("🌿 Plant Disease Detection System")
st.write("Upload an image of a plant leaf and the AI will predict the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with st.spinner('Analyzing...'):
        # Preprocessing
        img = image.resize((256, 256))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        
        # Prediction
        predictions = model.predict(img_array)
        result_index = np.argmax(predictions)
        result = class_names[result_index]
        confidence = np.max(predictions) * 100

    # Prediction results display
    st.success(f"**Prediction:** {result}")
    st.info(f"**Confidence:** {confidence:.2f}%")

    # Display Solution from Dictionary
    if result in disease_info:
        st.divider()
        st.subheader("🛠️ Recommendations")
        st.write(f"**Description:** {disease_info[result]['description']}")
        st.write(f"**Action Plan:** {disease_info[result]['solution']}")

# Gradio-style auto-run logic
if __name__ == "__main__":
    if "STREAMLIT_RUN" not in os.environ:
        os.environ["STREAMLIT_RUN"] = "true"
        os.system(f"streamlit run \"{__file__}\"")