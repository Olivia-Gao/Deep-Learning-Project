import base64
import cv2
import numpy as np
from flask import Blueprint, request, jsonify, current_app, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import os
from functools import wraps

functions_bp = Blueprint("functions", __name__, url_prefix="/functions")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'static/DLFile/upload_folder/'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def focal_loss_fixed(y_true, y_pred, gamma=2., alpha=0.75):
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.keras.backend.clip(y_pred, epsilon, 1. - epsilon)
    cross_entropy = -y_true * tf.math.log(y_pred)
    loss = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy
    return tf.reduce_sum(loss, axis=-1)

def get_img_array(img_path, size=(128, 128)):
    img = image.load_img(img_path, target_size=size)
    array = image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array / 255.0

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=1):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def superimpose_heatmap(original_path, heatmap, alpha=0.4, size=(256, 256)):
    img = cv2.imread(original_path)
    img = cv2.resize(img, size)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return img, heatmap_color, overlay

def encode_image_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def handle_errors(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            current_app.logger.error(f"预测失败: {str(e)}")
            return jsonify({'status': 'error', 'message': f"预测失败: {str(e)}"}), 500
    return wrapper

@functions_bp.route('/predict', methods=['GET', 'POST'])
@handle_errors
def predict_image():
    if request.method == 'GET':
        return render_template("result.html")
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': '未检测到图片文件'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'status': 'error', 'message': '请上传有效的图片（jpg/jpeg/png）'}), 400

    ensure_folder_exists(UPLOAD_FOLDER)
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    model_path = "model/best_model_split_70_20_10.h5"
    model = load_model(model_path, custom_objects={'focal_loss_fixed': focal_loss_fixed})
    last_conv_layer_name = "concatenate_5"

    img_array = get_img_array(file_path)
    preds = model.predict(img_array)
    pred_index = np.argmax(preds[0])
    pred_class = 'Defective' if pred_index == 1 else 'Non-Defective'

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index)
    img, heatmap_color, overlay = superimpose_heatmap(file_path, heatmap)

    return jsonify({
        'status': 'success',
        'prediction': pred_class,
        'original': encode_image_to_base64(img),
        'heatmap': encode_image_to_base64(heatmap_color),
        'overlay': encode_image_to_base64(overlay)
    })
