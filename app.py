# ===================================================================
# FILE: app.py (Flask Backend) - Simplified Version (No Slicing)
# ===================================================================
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import cv2
import dlib
from PIL import Image
import io
import os
import base64 # For encoding the image to send back

# --- Supress TensorFlow INFO and WARNING messages ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 1. INITIALIZE FLASK APP ---
app = Flask(__name__)

# --- Define Model Configuration ---
MODEL_IMG_SIZE = (224, 224) 
CLASS_LABELS = ['dry', 'normal', 'oily'] 

# ===================================================================
# --- MODEL AND DLIB LOADING ---
# ===================================================================
model = None
try:
    # Ensure your model file is named correctly
    model = tf.keras.models.load_model("last_final_2.keras")
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ FATAL: Could not load model. Error: {e}")

detector = None
predictor = None
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    print("✅ Dlib components loaded successfully.")
except Exception as e:
    print(f"❌ FATAL: Could not load dlib components. Error: {e}")

# ===================================================================
# --- HELPER, ENHANCEMENT, AND PREDICTION FUNCTIONS ---
# ===================================================================

def preprocess_upload(image_bytes):
    """Takes uploaded image bytes and prepares them as a CV2 image."""
    pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    cv_image = np.array(pil_image)
    # Convert RGB (from PIL) to BGR (for OpenCV)
    return cv_image[:, :, ::-1].copy()

def enhance_full_face(cv_image, shape):
    """Applies cosmetic enhancement to the entire face region."""
    try:
        points = np.array([[p.x, p.y] for p in shape.parts()])
        face_hull = cv2.convexHull(points)
        
        mask = np.zeros(cv_image.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, face_hull, 255)
        
        # Dilate the mask to ensure the whole face is covered smoothly
        kernel_size = 21
        mask = cv2.dilate(mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
        
        # Blur the background
        blurred_background = cv2.GaussianBlur(cv_image, (kernel_size, kernel_size), 0)
        
        # Apply filters only to the face area for enhancement
        denoised_face = cv2.bilateralFilter(cv_image, d=9, sigmaColor=75, sigmaSpace=75)
        
        lab = cv2.cvtColor(denoised_face, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_l = clahe.apply(l)
        enhanced_lab = cv2.merge((clahe_l, a, b))
        contrast_enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        blurred_sharp = cv2.GaussianBlur(contrast_enhanced, (5, 5), 0)
        sharpened_face = cv2.addWeighted(contrast_enhanced, 1.5, blurred_sharp, -0.5, 0)
        
        # Combine the enhanced face with the blurred background
        final_image = np.where(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) > 0, sharpened_face, blurred_background)
        return final_image
    except Exception as e:
        print(f"⚠️ Warning: Could not apply enhancement. Error: {e}")
        return cv_image # Return original on failure

def predict_skin_type(image_bytes):
    """Main pipeline: validates face, enhances, and predicts."""
    if model is None: return {"error": "Model is not loaded."}
    if detector is None: return {"error": "Face detector is not loaded."}
    
    cv_image = preprocess_upload(image_bytes)
    
    # --- Step 1: Face Detection (Validation) ---
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if not rects:
        return {'error': 'Face not found. Please use a photo with a clear, front-facing view.'}
    shape = predictor(gray, rects[0])

    # --- Step 2: Enhance the full face ---
    enhanced_image = enhance_full_face(cv_image, shape)
    
    # --- Step 3: Preprocess the enhanced image for the model ---
    # This preprocessing pipeline should exactly match your training script
    gray_enhanced = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(gray_enhanced, MODEL_IMG_SIZE)
    
    img_tensor = tf.convert_to_tensor(img_resized)
    img_tensor = tf.expand_dims(img_tensor, axis=-1)
    img_rgb = tf.image.grayscale_to_rgb(img_tensor) # Duplicate gray channel to 3 channels
    
    img_batch = tf.expand_dims(img_rgb, axis=0)
    img_batch_float = tf.cast(img_batch, tf.float32)
    img_preprocessed = tf.keras.applications.densenet.preprocess_input(img_batch_float)
    
    # --- Step 4: Make the Prediction ---
    prediction_scores = model.predict(img_preprocessed, verbose=0)[0]
    predicted_index = np.argmax(prediction_scores)
    prediction_label = CLASS_LABELS[predicted_index]
    confidence = float(np.max(prediction_scores))
    
    # --- Step 5: Prepare final response ---
    # Encode the enhanced image to send back for display
    _, buffer = cv2.imencode('.jpg', enhanced_image)
    enhanced_image_b64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "status": "success",
        "prediction": prediction_label,
        "confidence": round(confidence, 4),
        "enhanced_image_b64": enhanced_image_b64
    }

# ===================================================================
# --- FLASK ROUTES ---
# ===================================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def handle_prediction():
    if 'file' not in request.files: return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No file selected"}), 400
    
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