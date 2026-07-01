# 🌿 Plant Disease Detection — Flask Server

## Folder structure
```
flask_server/
├── app.py               ← main server
├── requirements.txt     ← dependencies
├── plant_model.tflite   ← copy here from Colab
└── class_names.json     ← copy here from Colab
```

## Setup

### 1. Copy your model files here
Put `plant_model.tflite` and `class_names.json` in this folder.

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the server
```bash
python app.py
```
Server starts at: http://localhost:5000

---

## API Endpoints

### GET /
Returns API info.

### GET /health
Returns server + MQTT connection status.

### POST /predict
Send a plant image and get the disease prediction back.

**Request:** `multipart/form-data` with key `image`

**Response:**
```json
{
  "label": "Tomato_Early_blight",
  "confidence": 94.3,
  "is_diseased": true,
  "status": "DISEASED",
  "timestamp": "2025-01-01T12:00:00"
}
```

---

## Test with curl
```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@/path/to/your/plant_photo.jpg"
```
