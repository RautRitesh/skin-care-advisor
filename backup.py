# ===================================================================
# FILE: app.py (Flask Backend) - Corrected Version
# ===================================================================
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import cv2
import dlib
from PIL import Image
import io
from collections import Counter
import os

# --- Supress TensorFlow INFO and WARNING messages ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 1. INITIALIZE FLASK APP ---

app = Flask(__name__)

# --- Define Model Configuration (Matches your training data) ---
MODEL_IMG_SIZE = (224, 224) 
# IMPORTANT: Make sure this order matches the sub-directories in your training data
# (e.g., the order that TensorFlow inferred them in).
CLASS_LABELS = ['dry', 'normal', 'oily'] 

# ===================================================================
# --- MODEL LOADING ---
# ===================================================================

try:
    # The .keras format is recommended and saves the full model (architecture + weights).
    model = tf.keras.models.load_model("gray_scale_2.keras")
    print("✅ Model 'gray_scale.keras' loaded successfully.")
    
except Exception as e:
    # Corrected the error message to show the correct filename
    print(f"❌ FATAL: Could not load the model. Please ensure 'gray_scale.keras' is in the same directory. Error: {e}")
    model = None

# ===================================================================
# --- DLIB SETUP ---
# ===================================================================

try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    print("✅ Dlib face detector and predictor loaded successfully.")
except Exception as e:
    print(f"❌ FATAL: Could not load 'shape_predictor_68_face_landmarks.dat'. Error: {e}")
    detector = None
    predictor = None

# ===================================================================
# --- HELPER AND PREDICTION FUNCTIONS ---
# ===================================================================

def preprocess_upload_for_slicing(image_bytes):
    """Takes uploaded image bytes and prepares them for face slicing."""
    pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    cv_image = np.array(pil_image)
    # Convert RGB (from PIL) to BGR (for OpenCV)
    return cv_image[:, :, ::-1].copy()

def get_face_slices_for_prediction(cv_image):
    """Finds a face and returns cropped BGR image regions."""
    if detector is None or predictor is None:
        return {'error': 'Face detector models are not loaded.'}

    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    if not rects:
        return {'error': 'Face not found. Please use a photo with a clear, front-facing view.'}

    shape = predictor(gray, rects[0])
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    
    slices = {}
    try:
        # Robustly get cheek and forehead slices from the original color image
        x_left_cheek, y_left_cheek, w_left_cheek, h_left_cheek = cv2.boundingRect(landmarks[1:5])
        slices["left_cheek"] = cv_image[y_left_cheek:y_left_cheek+h_left_cheek, x_left_cheek:x_left_cheek+w_left_cheek]

        x_right_cheek, y_right_cheek, w_right_cheek, h_right_cheek = cv2.boundingRect(landmarks[12:16])
        slices["right_cheek"] = cv_image[y_right_cheek:y_right_cheek+h_right_cheek, x_right_cheek:x_right_cheek+w_right_cheek]

        eyebrow_top = min(landmarks[19][1], landmarks[24][1])
        forehead_height = int((rects[0].bottom() - rects[0].top()) * 0.25)
        y1, y2 = max(0, eyebrow_top - forehead_height), eyebrow_top
        x1, x2 = landmarks[17][0], landmarks[26][0]
        if y2 > y1 and x2 > x1:
            slices["forehead"] = cv_image[y1:y2, x1:x2]
    except Exception as e:
        print(f"⚠️ Warning: An exception occurred during slicing: {e}")

    if not slices:
        return {'error': 'Could not extract valid facial regions.'}
        
    return slices

def predict_skin_type(image_bytes):
    """Main prediction pipeline that processes an image and returns results."""
    if model is None: return {"error": "Model is not loaded. Please check server logs."}
    
    cv_image = preprocess_upload_for_slicing(image_bytes)
    slices = get_face_slices_for_prediction(cv_image)
    
    if 'error' in slices:
        return slices
    
    region_predictions, all_predictions = {}, []
    
    for region_name, region_img_bgr in slices.items():
        # --- PREDICTION PREPROCESSING PIPELINE ---
        # This now matches your training script EXACTLY.

        # 1. Convert the BGR slice to Grayscale
        gray_slice = cv2.cvtColor(region_img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 2. Resize the grayscale image
        img_resized = cv2.resize(gray_slice, MODEL_IMG_SIZE)
        
        # 3. Convert to a tensor and add a channel dimension -> (224, 224, 1)
        img_tensor = tf.convert_to_tensor(img_resized)
        img_tensor = tf.expand_dims(img_tensor, axis=-1)
        
        # 4. Duplicate the single gray channel to get 3 channels -> (224, 224, 3)
        img_rgb = tf.image.grayscale_to_rgb(img_tensor)
        
        # 5. Add the batch dimension -> (1, 224, 224, 3)
        img_batch = tf.expand_dims(img_rgb, axis=0)
        
        # --- FIX: Convert data type to float32 before preprocessing ---
        img_batch_float = tf.cast(img_batch, tf.float32)

        # 6. Apply the correct preprocessing for DenseNet
        img_preprocessed = tf.keras.applications.densenet.preprocess_input(img_batch_float)
        
        # 7. Make the prediction
        prediction_scores = model.predict(img_preprocessed, verbose=0)[0]
        predicted_index = np.argmax(prediction_scores)
        predicted_label = CLASS_LABELS[predicted_index]
        confidence = float(np.max(prediction_scores))
        
        region_predictions[region_name] = {"prediction": predicted_label, "confidence": round(confidence, 4)}
        all_predictions.append(predicted_label)

    if not all_predictions: 
        return {"error": "Analysis failed. No facial regions were successfully processed."}

    # Determine the overall prediction based on the majority vote
    prediction_counts = Counter(all_predictions)
    # Special case for combination skin
    if 'oily' in prediction_counts and ('dry' in prediction_counts or 'normal' in prediction_counts):
        overall_prediction = 'Combination'
    else:
        overall_prediction = prediction_counts.most_common(1)[0][0]

    return {
        "status": "success",
        "overall_prediction": overall_prediction,
        "region_predictions": region_predictions
    }

# ===================================================================
# --- FLASK ROUTES ---
# ===================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def handle_prediction():
    if 'file' not in request.files: return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No file selected for uploading"}), 400
    
    try:
        image_bytes = file.read()
        result = predict_skin_type(image_bytes)
        return jsonify(result)
    except Exception as e:
        print(f"❌ An error occurred during prediction: {e}")
        return jsonify({"error": "An unexpected server error occurred."}), 500

# --- RUN APP ---
if __name__ == '__main__':
    # Use host='0.0.0.0' to make it accessible on your network
    app.run(host='0.0.0.0', port=5000)
