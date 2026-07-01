from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import json
import os
from PIL import Image
import io
import paho.mqtt.client as mqtt
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ==========================================
# LOAD MODEL & CLASS NAMES
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "class_names.json"), "r") as f:
    class_names = json.load(f)

interpreter = tf.lite.Interpreter(
    model_path=os.path.join(BASE_DIR, "plant_model.tflite")
)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = (128, 128)

# Read the model's actual quantization params at startup.
# Verified against the new plant_model.tflite:
#   Input : int8,  scale=1.00206,   zero_point=-128
#   Output: int8,  scale=0.00390625, zero_point=-128
# This model IS fully INT8 quantized (unlike the previous float32 model).
_input_dtype                    = input_details[0]["dtype"]
_input_scale, _input_zero_point = input_details[0]["quantization"]
_output_dtype                    = output_details[0]["dtype"]
_output_scale, _output_zero_point = output_details[0]["quantization"]
_is_quantized_input  = _input_scale  != 0.0
_is_quantized_output = _output_scale != 0.0

print(f"Model loaded — {len(class_names)} classes: {class_names}")
print(f"  Input  dtype={_input_dtype.__name__}, shape={input_details[0]['shape']}, "
      f"quantized={_is_quantized_input} (scale={_input_scale:.6f}, zp={_input_zero_point})")
print(f"  Output dtype={_output_dtype.__name__}, shape={output_details[0]['shape']}, "
      f"quantized={_is_quantized_output} (scale={_output_scale:.6f}, zp={_output_zero_point})")

# ==========================================
# MQTT SETUP
# ==========================================
MQTT_BROKER = "localhost"
MQTT_PORT   = 1883
MQTT_TOPIC  = "plant/alert"

mqtt_client = mqtt.Client()
mqtt_connected = False

def connect_mqtt():
    global mqtt_connected
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
        mqtt_connected = True
        print("MQTT broker connected")
    except Exception as e:
        print(f"MQTT broker not available ({e}) — alerts will be HTTP only")

connect_mqtt()

# Predictions whose top-1 vs top-2 margin is below this threshold are
# flagged as ambiguous — they won't trigger MQTT alerts.
AMBIGUITY_MARGIN = 15.0  # percentage points

# ==========================================
# INFERENCE
# ==========================================
def predict(image_bytes: bytes) -> dict:
    # 1. Decode and resize
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)   # raw 0-255

    # 2. This model has MobileNetV2 preprocess_input (x/127.5 - 1) baked
    #    into its own graph as the first two ops (MUL then SUB), confirmed
    #    by inspecting the TFLite op graph directly. Feed raw 0-255 pixels
    #    and let the model normalize internally — do NOT pre-normalize here.
    float_input = np.expand_dims(img_array, axis=0)  # (1, 128, 128, 3)

    # 3. Quantize float32 raw pixels -> int8 using the model's input params.
    #    Formula: q = round(real / scale) + zero_point, clamped to [-128, 127]
    #    Verified: scale=1.00206, zp=-128 produces a correct probability
    #    vector (sum=1.0) after dequantization.
    if _is_quantized_input:
        model_input = np.round(float_input / _input_scale) + _input_zero_point
        model_input = np.clip(model_input, -128, 127).astype(np.int8)
    else:
        model_input = float_input.astype(np.float32)

    # 4. Run inference
    interpreter.set_tensor(input_details[0]["index"], model_input)
    interpreter.invoke()
    raw_output = interpreter.get_tensor(output_details[0]["index"])[0]

    # 5. Dequantize int8 output -> real probabilities.
    #    Formula: real = (q - zero_point) * scale
    #    Output already represents a softmax probability vector — do NOT
    #    apply softmax again (that was the earlier double-softmax bug).
    if _is_quantized_output:
        probabilities = (raw_output.astype(np.float32) - _output_zero_point) * _output_scale
    else:
        probabilities = raw_output.astype(np.float32)

    # 6. Top-1 / top-2 + ambiguity check
    sorted_idx = np.argsort(probabilities)[::-1]
    top1_idx = int(sorted_idx[0])
    top2_idx = int(sorted_idx[1])
    top1_conf = float(probabilities[top1_idx]) * 100
    top2_conf = float(probabilities[top2_idx]) * 100
    margin    = top1_conf - top2_conf

    label       = class_names[top1_idx]
    is_diseased = "healthy" not in label.lower()
    is_ambiguous = margin < AMBIGUITY_MARGIN

    return {
        "label"      : label,
        "confidence" : round(top1_conf, 2),
        "runner_up"  : {
            "label"     : class_names[top2_idx],
            "confidence": round(top2_conf, 2),
        },
        "margin"      : round(margin, 2),
        "is_ambiguous": is_ambiguous,
        "is_diseased" : is_diseased,
        "status"      : "DISEASED" if is_diseased else "HEALTHY",
    }

# ==========================================
# ROUTES
# ==========================================
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status"          : "ok",
        "mqtt_connected"  : mqtt_connected,
        "num_classes"     : len(class_names),
        "classes"         : class_names,
        "input_dtype"     : _input_dtype.__name__,
        "input_quantized" : _is_quantized_input,
        "output_quantized": _is_quantized_output,
    })

@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        return jsonify({"error": "No image provided. POST as form-data with key 'image'"}), 400
    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        result = predict(image_file.read())
        result["timestamp"] = datetime.now().isoformat()

        if mqtt_connected and result["is_diseased"] and not result["is_ambiguous"]:
            payload = json.dumps({
                "disease"   : result["label"],
                "confidence": result["confidence"],
                "timestamp" : result["timestamp"],
            })
            mqtt_client.publish(MQTT_TOPIC, payload)
            print(f"MQTT alert: {result['label']}")

        print(f"Prediction: {result['label']} ({result['confidence']}%, "
              f"margin={result['margin']}, ambiguous={result['is_ambiguous']})")
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)