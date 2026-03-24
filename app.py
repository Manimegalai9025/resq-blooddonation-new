from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import traceback
import firebase_admin
from firebase_admin import credentials, messaging

app = Flask(__name__)
CORS(app)

print("\n" + "="*60)
print("🩸 BLOOD DONOR ELIGIBILITY API")
print("="*60)

# ------------ INITIALIZE FIREBASE FROM ENVIRONMENT VARIABLES ------------
print("\n🔥 Initializing Firebase from environment variables...")
FIREBASE_LOADED = False

try:
    import json
    
    # Check if environment variables exist (Render)
    if os.environ.get('FIREBASE_PROJECT_ID'):
        print("✅ Found Firebase environment variables")
        
        # Handle private key formatting (important!)
        private_key = os.environ.get('FIREBASE_PRIVATE_KEY', '')
        # Replace literal \n with actual newlines if needed
        if private_key and '\\n' in private_key:
            private_key = private_key.replace('\\n', '\n')
        
        # Create credential dictionary from environment variables
        firebase_cred_dict = {
            "type": os.environ.get('FIREBASE_TYPE', 'service_account'),
            "project_id": os.environ.get('FIREBASE_PROJECT_ID'),
            "private_key_id": os.environ.get('FIREBASE_PRIVATE_KEY_ID'),
            "private_key": private_key,
            "client_email": os.environ.get('FIREBASE_CLIENT_EMAIL'),
            "client_id": os.environ.get('FIREBASE_CLIENT_ID'),
            "auth_uri": os.environ.get('FIREBASE_AUTH_URI', 'https://accounts.google.com/o/oauth2/auth'),
            "token_uri": os.environ.get('FIREBASE_TOKEN_URI', 'https://oauth2.googleapis.com/token')
        }
        
        # Convert dict to JSON string and create credentials
        cred = credentials.Certificate(json.dumps(firebase_cred_dict))
        firebase_admin.initialize_app(cred)
        FIREBASE_LOADED = True
        print("✅ Firebase initialized successfully from environment variables!")
    
    else:
        # Fallback to file method (for local development)
        print("⚠️ No Firebase env vars found, trying file method...")
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        firebase_key_path = os.path.join(BASE_DIR, "serviceAccountKey.json")
        
        if os.path.exists(firebase_key_path):
            cred = credentials.Certificate(firebase_key_path)
            firebase_admin.initialize_app(cred)
            FIREBASE_LOADED = True
            print("✅ Firebase initialized successfully from file (local mode)!")
        else:
            FIREBASE_LOADED = False
            print("⚠️ Firebase key not found. FCM notifications disabled.")
            
except Exception as e:
    FIREBASE_LOADED = False
    print(f"❌ Firebase initialization failed: {e}")

# ------------ LOAD MODEL & ENCODERS ------------
print("\n📦 Loading ML model and encoders...")

MODEL_ACCURACY = "95.2%"

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    model_path = os.path.join(BASE_DIR, "donor_eligibility_model.pkl")
    city_path = os.path.join(BASE_DIR, "city_encoder.pkl")
    blood_path = os.path.join(BASE_DIR, "blood_encoder.pkl")
    
    print(f"📂 Loading from: {BASE_DIR}")
    print(f"   Model file: {os.path.exists(model_path)}")
    print(f"   City encoder: {os.path.exists(city_path)}")
    print(f"   Blood encoder: {os.path.exists(blood_path)}")
    
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    
    with open(city_path, "rb") as f:
        city_encoder = joblib.load(f)
    
    with open(blood_path, "rb") as f:
        blood_encoder = joblib.load(f)
    
    print("✅ Model loaded successfully!")
    print("✅ City encoder loaded")
    print("✅ Blood group encoder loaded")
    print(f"📊 Model Accuracy: {MODEL_ACCURACY}")
    
    if hasattr(model, 'feature_names_in_'):
        print(f"\n📊 Model expects {len(model.feature_names_in_)} features")
        print(f"📋 Features: {list(model.feature_names_in_)}")
    
    MODEL_LOADED = True
    print("\n" + "="*60)
    print("✅ SERVER READY - Waiting for requests...")
    print("="*60 + "\n")
    
except FileNotFoundError as e:
    print(f"\n❌ ERROR: Model files not found!")
    print(f"   Missing file: {e}")
    MODEL_LOADED = False
    
except Exception as e:
    print(f"\n❌ ERROR loading model: {e}")
    traceback.print_exc()
    MODEL_LOADED = False

# ------------ IN-MEMORY STORAGE FOR FCM TOKENS ------------
fcm_tokens_db = {}

# ------------ HEALTH CHECK (ROOT) WITH ACCURACY ------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "message": "🩸 Blood Donor Eligibility API",
        "version": "2.0",
        "model_loaded": MODEL_LOADED,
        "model_accuracy": MODEL_ACCURACY,
        "firebase_loaded": FIREBASE_LOADED,
        "endpoints": {
            "/": "API status (GET)",
            "/predict": "Predict eligibility (POST)",
            "/health": "Health check (GET)",
            "/register_token": "Register FCM token (POST)",
            "/send_notification": "Send blood request notification (POST)"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": MODEL_LOADED,
        "model_accuracy": MODEL_ACCURACY,
        "firebase_loaded": FIREBASE_LOADED
    })

# ------------ REGISTER FCM TOKEN ------------
@app.route("/register_token", methods=["POST"])
def register_token():
    try:
        data = request.get_json()
        email = data.get("email")
        fcm_token = data.get("fcm_token")
        blood_group = data.get("blood_group")
        eligible = data.get("eligible", False)
        
        if not email or not fcm_token:
            return jsonify({"error": "Email and FCM token required"}), 400
        
        fcm_tokens_db[email] = {
            "token": fcm_token,
            "blood_group": blood_group,
            "eligible": eligible
        }
        
        print(f"✅ Registered token for {email} ({blood_group})")
        
        return jsonify({
            "success": True,
            "message": "FCM token registered"
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------ SEND BLOOD REQUEST NOTIFICATION (FIXED) ------------
@app.route("/send_notification", methods=["POST"])
def send_notification():
    if not FIREBASE_LOADED:
        return jsonify({
            "error": "Firebase not initialized"
        }), 500
    
    try:
        data = request.get_json()
        hospital = data.get("hospital")
        blood_group = data.get("blood_group")
        urgency = data.get("urgency", "Normal")
        location = data.get("location", "")
        
        print(f"📢 Sending notification for blood group: {blood_group}")
        print(f"   Hospital: {hospital}, Urgency: {urgency}")
        
        if not hospital or not blood_group:
            return jsonify({"error": "Hospital and blood group required"}), 400
        
        # Find eligible donors
        eligible_donors = []
        for email, info in fcm_tokens_db.items():
            if info.get("blood_group") == blood_group and info.get("eligible"):
                eligible_donors.append({
                    "email": email,
                    "token": info.get("token")
                })
        
        print(f"✅ Found {len(eligible_donors)} eligible donors for {blood_group}")
        
        if not eligible_donors:
            return jsonify({
                "success": True,
                "message": "No eligible donors found",
                "notified": 0
            }), 200
        
        # Send notifications one by one (FIXED - no send_multicast)
        sent_count = 0
        for donor in eligible_donors:
            token = donor.get("token")
            if token:
                try:
                    # Set title based on urgency
                    if urgency.lower() == 'critical':
                        title = f"🚨 CRITICAL: {blood_group} Blood Needed!"
                    elif urgency.lower() == 'urgent':
                        title = f"⚠️ URGENT: {blood_group} Blood Needed!"
                    else:
                        title = f"📢 {blood_group} Blood Request"
                    
                    # Create message
                    message = messaging.Message(
                        notification=messaging.Notification(
                            title=title,
                            body=f"{hospital} needs {blood_group} blood. Urgency: {urgency}"
                        ),
                        data={
                            "urgency": urgency,
                            "bloodType": blood_group,
                            "hospital": hospital,
                            "location": location
                        },
                        android=messaging.AndroidConfig(
                            priority='high'
                        ),
                        token=token
                    )
                    
                    # Send notification
                    response = messaging.send(message)
                    sent_count += 1
                    print(f"   ✅ Sent to {donor['email']}")
                    
                except Exception as e:
                    print(f"   ❌ Failed to send to {donor['email']}: {e}")
        
        print(f"✅ Sent {sent_count}/{len(eligible_donors)} notifications")
        
        return jsonify({
            "success": True,
            "message": f"Notified {sent_count} donors",
            "notified": sent_count
        }), 200
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ------------ PREDICT API WITH CONFIDENCE ------------
@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL_LOADED:
        return jsonify({
            "error": "Model not loaded. Server initialization failed."
        }), 500
    
    try:
        data = request.get_json()
        print(f"\n📥 Received data: {data}")

        required_fields = [
            "age", "gender", "blood_group", "medical_conditions",
            "months_since_last_donation", "weight", "city",
            "latitude", "longitude"
        ]
        
        missing = [f for f in required_fields if f not in data]
        if missing:
            error_msg = f"Missing fields: {', '.join(missing)}"
            print(f"❌ {error_msg}")
            return jsonify({"error": error_msg}), 400

        age = data["age"]
        gender_str = data["gender"]
        blood_group = data["blood_group"]
        medical_conditions_str = data["medical_conditions"]
        months_since_last_donation = data["months_since_last_donation"]
        weight = data["weight"]
        city = data["city"]
        latitude = data["latitude"]
        longitude = data["longitude"]
        
        # Convert gender string to int (Male=1, Female=0)
        gender_clean = gender_str.strip().lower()
        if gender_clean == "male":
            gender = 1
        elif gender_clean == "female":
            gender = 0
        else:
            return jsonify({"error": f"Invalid gender: {gender_str}"}), 400
        
        # Convert medical conditions string to int (Yes=1, No/None=0)
        medical_clean = medical_conditions_str.strip().lower()
        if medical_clean in ["yes", "1", "true"]:
            medical_conditions = 1
        elif medical_clean in ["no", "none", "0", "false"]:
            medical_conditions = 0
        else:
            return jsonify({"error": f"Invalid medical condition: {medical_conditions_str}"}), 400
        
        print(f"✓ Gender: {gender_str} → {gender}")
        print(f"✓ Medical: {medical_conditions_str} → {medical_conditions}")
        
        city_df_temp = pd.DataFrame([[city]], columns=["city"])
        blood_df_temp = pd.DataFrame([[blood_group]], columns=["blood_group"])
        
        city_encoded = city_encoder.transform(city_df_temp)
        blood_encoded = blood_encoder.transform(blood_df_temp)

        city_df = pd.DataFrame(
            city_encoded,
            columns=city_encoder.get_feature_names_out(["city"])
        )

        blood_df = pd.DataFrame(
            blood_encoded,
            columns=blood_encoder.get_feature_names_out(["blood_group"])
        )

        feature_dict = {
            "age": [age],
            "gender": [gender],
            "latitude": [latitude],
            "longitude": [longitude],
            "weight": [weight],
            "medical_conditions": [medical_conditions],
            "months_since_last_donation": [months_since_last_donation]
        }
        
        df = pd.DataFrame(feature_dict)
        
        for col in city_df.columns:
            df[col] = city_df[col].values
        
        for col in blood_df.columns:
            df[col] = blood_df[col].values

        print(f"📊 DataFrame shape: {df.shape}")
        print(f"📋 DataFrame columns: {list(df.columns)}")

        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
            df = df[expected_features]
            print(f"✓ Columns reordered successfully")

        pred = model.predict(df)[0]
        result = "Yes" if pred == 1 else "No"
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df)[0]
            confidence = float(max(probabilities) * 100)
            eligible_prob = float(probabilities[1] * 100) if len(probabilities) > 1 else confidence
            
            print(f"✅ Prediction: {result}")
            print(f"📊 Confidence: {confidence:.1f}%")
            print(f"📊 Eligible probability: {eligible_prob:.1f}%\n")
        else:
            confidence = 100.0 if result == "Yes" else 0.0
            eligible_prob = confidence
            print(f"✅ Prediction: {result}\n")

        # Auto-register eligible donor
        if "fcm_token" in data and result == "Yes":
            email = data.get("email", "unknown")
            fcm_tokens_db[email] = {
                "token": data["fcm_token"],
                "blood_group": blood_group,
                "eligible": True
            }
            print(f"✅ Auto-registered eligible donor: {email}")

        return jsonify({
            "eligible": result,
            "confidence": round(confidence, 1),
            "eligible_probability": round(eligible_prob, 1),
            "message": f"Prediction completed with {confidence:.1f}% confidence"
        })

    except KeyError as e:
        error_msg = f"Missing required field: {str(e)}"
        print(f"❌ {error_msg}\n")
        return jsonify({"error": error_msg}), 400
    
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error: {error_msg}")
        traceback.print_exc()
        return jsonify({"error": error_msg}), 500

# ------------ DEBUG: VIEW REGISTERED TOKENS ------------
@app.route("/tokens", methods=["GET"])
def view_tokens():
    return jsonify(fcm_tokens_db)
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)