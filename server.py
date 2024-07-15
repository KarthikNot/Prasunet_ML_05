import streamlit as st 
import cv2
import numpy as np
from keras.models import load_model #type:ignore
from live_video import capture_video
import matplotlib.pyplot as plt

st.set_page_config(page_icon='🍽️', page_title='Food Image Classification', layout="wide")
st.markdown('<div style="text-align:center;font-size:50px;">Food Images Classification 🍝</div>', unsafe_allow_html=True)

try:
    model = load_model('./models/best_model1.keras')
except Exception as e:
    st.error(f"Error loading the model:{e}")

classes = ['macarons', 'french_toast', 'lobster_bisque', 'prime_rib', 'pork_chop', 'guacamole', 'baby_back_ribs', 'mussels', 
           'beef_carpaccio', 'poutine', 'hot_and_sour_soup', 'seaweed_salad', 'foie_gras', 'dumplings', 'peking_duck', 
           'takoyaki', 'bibimbap', 'falafel', 'pulled_pork_sandwich', 'lobster_roll_sandwich', 'carrot_cake', 'beet_salad', 
           'panna_cotta', 'donuts', 'red_velvet_cake', 'grilled_cheese_sandwich', 'cannoli', 'spring_rolls', 'shrimp_and_grits',
           'clam_chowder', 'omelette', 'fried_calamari', 'caprese_salad', 'oysters', 'scallops', 'ramen', 'grilled_salmon', 
           'croque_madame', 'filet_mignon', 'hamburger', 'spaghetti_carbonara', 'miso_soup', 'bread_pudding', 'lasagna', 
           'crab_cakes', 'cheesecake', 'spaghetti_bolognese', 'cup_cakes', 'creme_brulee', 'waffles', 'fish_and_chips', 
           'paella', 'macaroni_and_cheese', 'chocolate_mousse', 'ravioli', 'chicken_curry', 'caesar_salad', 'nachos', 
           'tiramisu', 'frozen_yogurt', 'ice_cream', 'risotto', 'club_sandwich', 'strawberry_shortcake', 'steak', 'churros', 
           'garlic_bread', 'baklava', 'bruschetta', 'hummus', 'chicken_wings', 'greek_salad', 'tuna_tartare', 'chocolate_cake',
           'gyoza', 'eggs_benedict', 'deviled_eggs', 'samosa', 'sushi', 'breakfast_burrito', 'ceviche', 'beef_tartare', 
           'apple_pie', 'huevos_rancheros', 'beignets', 'pizza', 'edamame', 'french_onion_soup', 'hot_dog', 'tacos', 
           'chicken_quesadilla', 'pho', 'gnocchi', 'pancakes', 'fried_rice', 'cheese_plate', 'onion_rings', 'escargots', 
           'sashimi', 'pad_thai', 'french_fries']

def preprocess_image(image):
    try:
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image")
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        st.image(img)
        return img
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None
    

def classify_image(image):
    try:
        image = preprocess_image(image)
        if image is not None:
            prediction = model.predict(image)
            predicted_class = classes[np.argmax(prediction)]
            confidence = np.max(prediction)
            return predicted_class, confidence, prediction
        else:
            return None, None, None
    except Exception as e:
        st.error(f"Error classifying image: {e}")
        return None, None, None

def main():
    st.sidebar.title('Options')
    app_mode = st.sidebar.selectbox('Choose the app mode', ['Image Input', 'Webcam Input'])

    if app_mode == 'Image Input':
        st.sidebar.write('Upload an image for classification:')
        uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            st.image(uploaded_file, width = 450)
            if st.button('Classify Image'):
                predicted_class, confidence, prediction = classify_image(uploaded_file)
                if predicted_class is not None:
                    st.write("Prediction Probabilities:", prediction)
                    st.markdown(f'<h2>Predicted Class: {predicted_class}</h2>', unsafe_allow_html=True)
                    st.markdown(f'<h3>Confidence: {confidence:.2f}</h3>', unsafe_allow_html=True)
                else:
                    st.error("Failed to classify image. Please try another image.")

    elif app_mode == 'Webcam Input':
        st.sidebar.write('Click Start to begin webcam capture:')
        if st.button('Start'):
            st.write('To stop the live capture press Q on keyboard')
            capture_video()

if __name__ == '__main__':
    main()