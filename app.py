# ===================================================================
# FILE: app.py (Flask Backend) - Updated Version
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
CLASS_LABELS = ['dry', 'oily', 'normal'] # IMPORTANT: Must match the sub-directory order in your train/test folders

# ===================================================================
# --- MODEL LOADING ---
# ===================================================================

# --- Load the Model and Trained Weights ---
try:
    # When loading a full model from a .h5 file, you don't need to load the weights separately.
    # The .h5 file created by model.save() already contains the weights.
    # The custom_objects dictionary is needed to tell TensorFlow how to load the augmentation layers.
    custom_objects={
        'RandomFlip':tf.keras.layers.RandomFlip,
        'RandomRotation':tf.keras.layers.RandomRotation,
        'RandomZoom':tf.keras.layers.RandomZoom,
        'RandomContrast':tf.keras.layers.RandomContrast
    }
    model = tf.keras.models.load_model(
        "skin_type_classifier_finale.h5", # This file should contain the full model
        custom_objects=custom_objects,
        compile=False
    )
    print("✅ Full model and weights loaded successfully.")
    
except Exception as e:
    print(f"❌ FATAL: Could not build model or load weights. Error: {e}")
    model = None

# ===================================================================
# --- DLIB SETUP ---
# ===================================================================

# --- Load Dlib for face processing ---
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

def preprocess_image_for_prediction(image_bytes):
    """Takes image bytes, converts to a NumPy array for processing."""
    pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    cv_image = np.array(pil_image)
    # Convert RGB (from PIL) to BGR (for OpenCV)
    return cv_image[:, :, ::-1].copy()

# --- START: NEW, MORE ROBUST FACE SLICING FUNCTION ---
def get_face_slices_for_prediction(cv_image, debug=False):
    """
    Finds a face in an image and returns cropped regions. This version is more
    robust and less strict about finding every single region.
    """
    if detector is None or predictor is None:
        return {'error': 'Face detector models are not loaded.'}

    debug_image = cv_image.copy() if debug else None
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    if not rects:
        print("⚠️ Dlib face detector did not find a face.")
        return {'error': 'Face not found. Please use a photo with a clear, front-facing view.'}

    rect = rects[0]
    shape = predictor(gray, rect)
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    
    slices = {}

    def is_slice_valid(slice_img):
        return slice_img is not None and slice_img.size > 0 and slice_img.shape[0] > 10 and slice_img.shape[1] > 10

    try:
        # Left Cheek: Get bounding box around the key landmarks. This is more robust.
        cheek_pts = landmarks[1:5] # Points along the curve of the left cheek
        x, y, w, h = cv2.boundingRect(cheek_pts)
        if w > 10 and h > 10: # Ensure the box has a reasonable size
            left_cheek_crop = cv_image[y:y+h, x:x+w]
            slices["left_cheek"] = left_cheek_crop
            if debug: cv2.rectangle(debug_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Right Cheek: Use the same robust bounding box method.
        cheek_pts = landmarks[12:16] # Points along the curve of the right cheek
        x, y, w, h = cv2.boundingRect(cheek_pts)
        if w > 10 and h > 10:
            right_cheek_crop = cv_image[y:y+h, x:x+w]
            slices["right_cheek"] = right_cheek_crop
            if debug: cv2.rectangle(debug_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Forehead
        eyebrow_left = landmarks[17]
        eyebrow_right = landmarks[26]
        eyebrow_top = min(landmarks[19][1], landmarks[24][1])
        face_height = rect.bottom() - rect.top()
        forehead_height = int(face_height * 0.25)
        y1, y2 = max(0, eyebrow_top - forehead_height), eyebrow_top
        x1, x2 = eyebrow_left[0], eyebrow_right[0]
        if y2 > y1 and x2 > x1:
            forehead_crop = cv_image[y1:y2, x1:x2]
            if is_slice_valid(forehead_crop):
                slices["forehead"] = forehead_crop
                if debug: cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # T-Zone / Nose
        y1, y2 = landmarks[27][1], landmarks[33][1]
        x1, x2 = landmarks[31][0], landmarks[35][0]
        if y2 > y1 and x2 > x1:
            t_zone_crop = cv_image[y1:y2, x1:x2]
            if is_slice_valid(t_zone_crop):
                slices["t_zone"] = t_zone_crop
                if debug: cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    except Exception as e:
        print(f"⚠️ Warning: An exception occurred during slicing. Error: {e}")
        # Even if one part fails, we still return what we found.

    if debug:
        for (x, y) in landmarks:
            cv2.circle(debug_image, (x, y), 2, (0, 0, 255), -1)
        cv2.imwrite("debug_output.jpg", debug_image)
        print("ℹ️ Saved debug image to 'debug_output.jpg'")

    if not slices:
        return {'error': 'Could not extract any valid facial regions even though a face was detected.'}

    return slices
# --- END: NEW FACE SLICING FUNCTION ---


def predict_skin_type(image_bytes):
    """Main prediction pipeline that processes an image and returns results."""
    if model is None: return {"error": "Model is not loaded. Please check server logs."}
    
    cv_image = preprocess_image_for_prediction(image_bytes)
    
    # To debug a failing image, change debug=False to debug=True
    slices = get_face_slices_for_prediction(cv_image, debug=False)
    
    if 'error' in slices:
        return slices
    
    if not slices: 
        return {"error": "Could not detect any facial regions. Please use a clear, well-lit, front-facing photo."}

    region_predictions, all_predictions = {}, []
    
    for region_name, region_img in slices.items():
        img_resized = cv2.resize(region_img, MODEL_IMG_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

        prediction_scores = model.predict(img_array, verbose=0)[0]
        predicted_index = np.argmax(prediction_scores)
        predicted_label = CLASS_LABELS[predicted_index]
        confidence = float(np.max(prediction_scores))
        
        region_predictions[region_name] = {"prediction": predicted_label, "confidence": round(confidence, 4)}
        all_predictions.append(predicted_label)

    # Be more forgiving: require at least 2 regions to make a prediction
    if len(all_predictions) < 2: 
        return {"error": f"Analysis Incomplete. Only found {len(all_predictions)} region(s). Please use a clearer photo."}

    prediction_counts = Counter(all_predictions)
    
    if 'oily' in prediction_counts and 'dry' in prediction_counts:
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
    app.run(host='0.0.0.0', port=5000)
