from flask import Flask, request, jsonify, render_template_string
import numpy as np
import cv2
import requests
import json
import os
from datetime import datetime

app = Flask(__name__)

# --------------------------------------------------
# ⚠️ CONFIGURATION DES ADRESSES IP
# --------------------------------------------------
PHONE_IP = "192.168.137.92:8080"  # IP du téléphone (IP Webcam)
ESP32_IP = "192.168.137.234"      # IP de l'ESP32-CAM
# --------------------------------------------------

# Chargement du modèle .tflite
try:
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path="plant_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ Le modèle TFLite a été chargé avec succès dans le PC !")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle : {e}")

# Liste des classes
classes = ["HIBISCUS_DISEASED", "HIBISCUS_HEALTHY", "Potato___Late_blight", "Potato___healthy", "Tomato_Late_blight", "Tomato_healthy"]

# 📚 BASE DE DONNÉES DES TRAITEMENTS ET CONSEILS (En Français)
TREATMENT_ADVISOR = {
    "hibiscus_healthy": {
        "title": "🎉 Hibiscus en parfaite santé !",
        "steps": [
            "☀️ **Lumière :** Placez-le dans un endroit très lumineux, avec au moins 6 heures de soleil direct par jour.",
            "💧 **Arrosage :** Arrosez régulièrement durant l'été, en laissant le sol sécher légèrement en surface entre deux arrosages.",
            "🌱 **Engrais :** Appliquez un engrais riche en potassium une fois par mois pour stimuler une belle floraison."
        ]
    },
    "hibiscus_diseased": {
        "title": "🚨 Hibiscus Malade (Infection ou Ravageurs) !",
        "steps": [
            "🔍 **Symptômes :** Présence de taches noires, feuilles jaunies ou petits insectes (pucerons) sous les feuilles.",
            "✂️ **Taille :** Coupez immédiatement les branches et feuilles gravement touchées avec un sécateur désinfecté.",
            "🧴 **Remède :** Vaporisez une solution de savon noir diluée dans de l'eau tiède ou un fongicide naturel pour soigner la plante."
        ]
    },
    "potato___healthy": {
        "title": "🎉 Plant de Pomme de Terre Vigoureux et Sain !",
        "steps": [
            "💧 **Arrosage :** Arrosez toujours au pied sans mouiller les feuilles pour éviter l'apparition de champignons.",
            "🥔 **Buttage :** Ramenez de la terre autour de la tige (butter) pour protéger les tubercules de la lumière du soleil.",
            "🌬️ **Espace :** Assurez-vous qu'il y a assez d'espace entre les plants pour une bonne circulation de l'air."
        ]
    },
    "potato___late_blight": {
        "title": "🚨 Mildiou de la Pomme de Terre détecté (Late Blight) !",
        "steps": [
            "🔍 **Symptômes :** Taches brun-noirâtres sur les feuilles qui se propagent vite, avec un duvet blanc humide dessous.",
            "✂️ **Action Vitale :** Coupez et brûlez/jetez immédiatement les feuilles infectées (ne jamais les mettre au compost).",
            "🧴 **Traitement :** Pulvérisez de la **Bouillie Bordelaise** (traitement à base de cuivre) pour stopper la propagation."
        ]
    },
    "tomato_healthy": {
        "title": "🎉 Plant de Tomate vigoureux et en pleine forme !",
        "steps": [
            "☀️ **Exposition :** Un maximum de soleil (6 à 8 heures par jour) pour aider les tomates à mûrir.",
            "✂️ **Taille :** Retirez régulièrement les 'gourmands' (jeunes pousses inutiles) pour concentrer l'énergie sur les fruits.",
            "💧 **Régularité :** Arrosez de manière constante sans excès pour éviter le fendillement des fruits."
        ]
    },
    "tomato_late_blight": {
        "title": "🚨 Mildiou de la Tomate détecté (Late Blight) !",
        "steps": [
            "🔍 **Symptômes :** Taches brunes d'aspect huileux sur les feuilles, tiges brunies et fruits se couvrant de taches dures.",
            "✂️ **Aération :** Enlevez les feuilles du bas pour qu'elles ne touchent pas le sol humide.",
            "🧴 **Traitement :** Appliquez immédiatement un fongicide biologique à base de cuivre et protégez la plante de la pluie si possible."
        ]
    }
}

# Fichier d'historique
HISTORY_FILE = "history_log.json"

def add_to_history(status, class_name, treatment_info):
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            try:
                history = json.load(f)
            except:
                pass
    
    new_entry = {
        "time": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "status": status,
        "class": class_name,
        "treatment": treatment_info
    }
    history.insert(0, new_entry)
    history = history[:10]
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f)

# Interface HTML Moderne (100% en Français)
HTML_INTERFACE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PlantGuard AI Pro - Tableau de bord</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --primary: #2e7d32;
            --primary-light: #4caf50;
            --danger: #d32f2f;
            --dark: #121212;
            --card-bg: rgba(255, 255, 255, 0.95);
            --text: #333;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
            color: var(--text);
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        header {
            text-align: center;
            margin-bottom: 30px;
        }

        header h1 {
            color: var(--primary);
            font-size: 2.5rem;
            margin: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
        }

        header p {
            color: #555;
            margin-top: 5px;
            font-weight: 500;
        }

        .grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 25px;
        }

        @media (min-width: 768px) {
            .grid {
                grid-template-columns: 1.3fr 1fr;
            }
        }

        .card {
            background: var(--card-bg);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.08);
            border: 1px solid rgba(255,255,255,0.5);
        }

        .card h2 {
            margin-top: 0;
            color: var(--primary);
            font-size: 1.4rem;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .stream-container {
            position: relative;
            border-radius: 12px;
            overflow: hidden;
            border: 3px solid var(--primary);
            height: 320px;
            background: #000;
        }

        .stream-view {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        .btn {
            background: var(--primary);
            color: white;
            border: none;
            padding: 14px 28px;
            font-size: 18px;
            border-radius: 30px;
            cursor: pointer;
            width: 100%;
            margin-top: 15px;
            font-weight: bold;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            box-shadow: 0 4px 15px rgba(46, 125, 50, 0.3);
            transition: all 0.3s ease;
        }

        .btn:hover {
            background: var(--primary-light);
        }

        #loading {
            display: none;
            text-align: center;
            margin-top: 15px;
            font-weight: bold;
            color: #ff9800;
        }

        .status-box {
            padding: 15px;
            border-radius: 10px;
            margin-top: 15px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
        }

        .status-healthy {
            background: #e8f5e9;
            color: var(--primary);
            border: 2px solid var(--primary);
        }

        .status-unhealthy {
            background: #ffebee;
            color: var(--danger);
            border: 2px solid var(--danger);
        }

        /* Boite de traitement */
        .treatment-box {
            margin-top: 15px;
            padding: 15px;
            background: #fdfdfd;
            border-left: 5px solid var(--primary);
            border-radius: 4px;
            display: none;
        }

        .treatment-box h3 {
            margin: 0 0 10px 0;
            color: #333;
        }

        .treatment-box ul {
            margin: 0;
            padding-left: 20px;
        }

        .treatment-box li {
            margin-bottom: 8px;
            line-height: 1.5;
        }

        /* Historique */
        .history-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }

        .history-table th, .history-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }

        .history-table th {
            background-color: rgba(0,0,0,0.05);
            color: var(--primary);
        }

        .badge {
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }

        .badge-healthy {
            background: #c8e6c9;
            color: #2e7d32;
        }

        .badge-unhealthy {
            background: #ffcdd2;
            color: #c62828;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1><i class="fa-solid fa-leaf"></i> PlantGuard AI Pro</h1>
            <p>Analyse en temps réel, Conseils d'experts & Alerte Flash par ESP32-CAM</p>
        </header>

        <div class="grid">
            <div class="card">
                <h2><i class="fa-solid fa-video"></i> Diagnostic de la plante</h2>
                <div class="stream-container">
                    <img id="cameraStream" class="stream-view" src="http://PHONE_IP_PLACEHOLDER/video" alt="Caméra non connectée">
                </div>
                <button class="btn" onclick="takePredict()">
                    <i class="fa-solid fa-expand"></i> Scanner la plante
                </button>
                
                <div id="loading">
                    <i class="fa-solid fa-spinner fa-spin"></i> Analyse d'image IA en cours...
                </div>

                <div id="statusResult" class="status-box" style="display: none;"></div>

                <div id="treatmentBox" class="treatment-box">
                    <h3 id="treatmentTitle"></h3>
                    <ul id="treatmentList"></ul>
                </div>
            </div>

            <div class="card">
                <h2><i class="fa-solid fa-clock-rotate-left"></i> Historique des analyses</h2>
                <div style="overflow-x: auto;">
                    <table class="history-table">
                        <thead>
                            <tr>
                                <th>Date/Heure</th>
                                <th>Classe détectée</th>
                                <th>Statut</th>
                            </tr>
                        </thead>
                        <tbody id="historyBody"></tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function loadHistory() {
            try {
                const response = await fetch('/get_history');
                const data = await response.json();
                const tbody = document.getElementById('historyBody');
                tbody.innerHTML = '';
                
                if (data.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="3" style="text-align:center;">Aucune analyse enregistrée.</td></tr>';
                    return;
                }

                data.forEach(item => {
                    const badgeClass = item.status === 'Sain' ? 'badge-healthy' : 'badge-unhealthy';
                    tbody.innerHTML += `
                        <tr>
                            <td>${item.time}</td>
                            <td><strong>${item.class}</strong></td>
                            <td><span class="badge ${badgeClass}">${item.status}</span></td>
                        </tr>
                    `;
                });
            } catch (err) {
                console.error("Erreur historique:", err);
            }
        }

        async function takePredict() {
            const loading = document.getElementById('loading');
            const statusResult = document.getElementById('statusResult');
            const treatmentBox = document.getElementById('treatmentBox');
            const treatmentTitle = document.getElementById('treatmentTitle');
            const treatmentList = document.getElementById('treatmentList');
            
            loading.style.display = 'block';
            statusResult.style.display = 'none';
            treatmentBox.style.display = 'none';

            try {
                const response = await fetch('/predict_direct', { method: 'POST' });
                const result = await response.json();
                
                loading.style.display = 'none';
                statusResult.style.display = 'block';
                treatmentBox.style.display = 'block';

                if (result.status === "Sain") {
                    statusResult.className = "status-box status-healthy";
                    statusResult.innerHTML = `<i class="fa-solid fa-heart-circle-check"></i> ${result.class} (${result.status})`;
                    treatmentBox.style.borderLeftColor = "var(--primary)";
                } else {
                    statusResult.className = "status-box status-unhealthy";
                    statusResult.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> ${result.class} (${result.status})`;
                    treatmentBox.style.borderLeftColor = "var(--danger)";
                }

                // Remplissage des conseils d'expert
                treatmentTitle.innerHTML = result.treatment_info.title;
                treatmentList.innerHTML = '';
                result.treatment_info.steps.forEach(step => {
                    const formattedStep = step.replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>');
                    treatmentList.innerHTML += `<li>${formattedStep}</li>`;
                });

                loadHistory();
            } catch (err) {
                loading.style.display = 'none';
                statusResult.style.display = 'block';
                statusResult.className = "status-box status-unhealthy";
                statusResult.innerHTML = `❌ Erreur : ${err}`;
            }
        }

        loadHistory();
    </script>
</body>
</html>
""".replace("PHONE_IP_PLACEHOLDER", PHONE_IP)

@app.route('/')
def home():
    return render_template_string(HTML_INTERFACE)

@app.route('/get_history')
def get_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return jsonify(json.load(f))
    return jsonify([])

@app.route('/predict_direct', methods=['POST'])
def predict_direct():
    try:
        # 1. Capture d'image
        phone_url = f"http://{PHONE_IP}/shot.jpg"
        img_resp = requests.get(phone_url, timeout=5)
        img_bytes = img_resp.content
        
        npimg = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        # 2. Prétraitement et Prédiction
        img_resized = cv2.resize(img, (128, 128))
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        input_scale, input_zero_point = input_details[0]['quantization']
        img_quantized = (img_resized.astype(np.float32) / input_scale) + input_zero_point
        img_quantized = np.clip(img_quantized, -128, 127)
        img_input = np.expand_dims(img_quantized, axis=0).astype(np.int8)
        
        interpreter.set_tensor(input_details[0]['index'], img_input)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        if output_details[0]['dtype'] == np.int8:
            output_scale, output_zero_point = output_details[0]['quantization']
            output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
        
        predicted_class_idx = np.argmax(output_data[0])
        class_name = classes[predicted_class_idx]
        
        # 3. Association des diagnostics et envoi de l'alerte à l'ESP32
        class_key = class_name.lower()
        
        if "healthy" in class_key:
            status = "Sain"
            treatment_info = TREATMENT_ADVISOR.get(class_key, TREATMENT_ADVISOR["hibiscus_healthy"])
            # Éteindre le flash de l'ESP32
            requests.get(f"http://{ESP32_IP}/alert?status=healthy", timeout=3)
        else:
            status = "Malade"
            treatment_info = TREATMENT_ADVISOR.get(class_key, TREATMENT_ADVISOR["hibiscus_diseased"])
            # Allumer le flash de l'ESP32
            requests.get(f"http://{ESP32_IP}/alert?status=unhealthy", timeout=3)
            
        add_to_history(status, class_name, treatment_info)
            
        return jsonify({
            "status": status, 
            "class": class_name,
            "treatment_info": treatment_info
        })
        
    except Exception as e:
        return jsonify({"status": "Erreur", "class": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)