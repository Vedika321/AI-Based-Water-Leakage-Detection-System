import math
import os
import random
from datetime import datetime
from flask import Flask, jsonify, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "leakage_model.pkl")

EVENTS = []

ALLOWED_MATERIALS = [
    "PVC", "CPVC", "HDPE", "Iron", "Steel",
    "Galvanized Iron", "Copper", "Concrete"
]

# Real-time simulation state
sim_state = {
    "risk": 14,
    "flow": 45.2,
    "pressure": 62.5,
    "devices": 6,
    "leak_simulated": False,
    "demo_paused": False
}


def analyze_pipe(brand, age, material, flow_rate, pressure):
    risk = 0
    reasons = []
    solutions = []

    # Age assessment
    if age >= 30:
        risk += 25
        reasons.append("The pipe is past its standard operating lifecycle.")
        solutions.append("Arrange an ultrasonic thickness test and assess section replacement.")
    elif age >= 15:
        risk += 18
        reasons.append("The pipe shows standard mid-to-late life degradation.")
        solutions.append("Inspect joints, couplings, and valves for micro-fissures.")
    elif age >= 5:
        risk += 8
        reasons.append("The pipe is within normal working service life.")
    else:
        reasons.append("The pipe is relatively new.")

    # Material assessment
    material_scores = {
        "PVC": 5, "CPVC": 5, "HDPE": 5, "Iron": 20,
        "Steel": 15, "Galvanized Iron": 20, "Copper": 5, "Concrete": 15
    }
    risk += material_scores.get(material, 10)

    if material in ["Iron", "Steel", "Galvanized Iron"]:
        reasons.append("Ferrous piping is susceptible to internal and external oxidation/tuberculation.")
        solutions.append("Conduct a corrosion assessment and cathodic protection check.")
    elif material == "Concrete":
        reasons.append("Rigid concrete is susceptible to shear fractures and joint displacement.")
        solutions.append("Perform acoustic leak detection or CCTV camera line crawler survey.")
    else:
        reasons.append("Thermoplastic/copper line with high corrosion resistance.")
        solutions.append("Check solvent welds, compression fittings, and thermal expansion joints.")

    # Flow rate assessment
    if flow_rate > 60:
        risk += 20
        reasons.append(f"Inlet flow ({flow_rate:.1f} L/s) exceeds normal network threshold (52 L/s).")
        solutions.append("Check downstream zone isolating valves for open bursts.")
    elif flow_rate < 10:
        risk += 15
        reasons.append(f"Abnormally low flow rate ({flow_rate:.1f} L/s).")
        solutions.append("Check for upstream suction throttling or sediment blockages.")

    # Pressure assessment
    if pressure < 50:
        risk += 25
        reasons.append(f"Line pressure ({pressure:.1f} psi) is below the minimum threshold (50 psi).")
        solutions.append("High probability of an active breach; deploy acoustic listening sticks.")
    elif pressure > 85:
        risk += 15
        reasons.append(f"Line pressure ({pressure:.1f} psi) is high, accelerating joint fatigue.")
        solutions.append("Inspect and re-calibrate the upstream pressure reducing valve (PRV).")

    # ML model prediction fallback
    model_note = "Heuristic rule-based assessment"
    if os.path.exists(MODEL_FILE):
        try:
            saved_data = joblib.load(MODEL_FILE)
            model = saved_data.get("model")
            model_columns = saved_data.get("columns", [])

            input_data = {
                "age": age,
                "flow_rate": flow_rate,
                "pressure": pressure
            }
            for col in model_columns:
                if col.startswith("material_"):
                    input_data[col] = int(col[len("material_"):] == material)

            input_df = pd.DataFrame(
                [[input_data.get(c, 0) for c in model_columns]],
                columns=model_columns
            )

            prediction = int(model.predict(input_df)[0])
            if hasattr(model, "predict_proba") and 1 in list(model.classes_):
                probs = model.predict_proba(input_df)[0]
                idx = list(model.classes_).index(1)
                risk = round(float(probs[idx]) * 100)
                model_note = "ML classification inference"

            if prediction == 1:
                reasons.insert(0, "Trained classification model flagged active risk profile.")
        except Exception:
            model_note = "Rule-based fallback (model load issue)"

    risk = max(0, min(risk, 100))

    if risk >= 60:
        risk_level = "High"
        action = "Dispatch an emergency field crew for immediate acoustic line tracing."
    elif risk >= 35:
        risk_level = "Medium"
        action = "Schedule secondary inspection and review pressure transient logs."
    else:
        risk_level = "Low"
        action = "Network operating nominally. Maintain scheduled preventative maintenance."

    if not solutions:
        solutions.append("Continue routine supervisory monitoring.")

    return {
        "brand": brand,
        "age": age,
        "material": material,
        "flow_rate": flow_rate,
        "pressure": pressure,
        "risk": risk,
        "risk_level": risk_level,
        "reasons": reasons,
        "solution": " ".join(dict.fromkeys(solutions)),
        "action": action,
        "model_note": model_note
    }


@app.route("/")
def home():
    return render_template(
        "index.html",
        events=EVENTS,
        result=None,
        error=None,
        show_events=False
    )


@app.route("/analyze-pipe", methods=["POST"])
def analyze():
    try:
        brand = request.form.get("brand", "").strip()
        material = request.form.get("material", "").strip()
        age_text = request.form.get("age", "").strip()

        # Optional flow and pressure; defaults to live simulation readings if not provided
        flow_text = request.form.get("flow_rate", "").strip()
        pressure_text = request.form.get("pressure", "").strip()

        if not brand or not material or not age_text:
            raise ValueError("Please provide the pipe brand, age, and material.")

        age = float(age_text)
        flow_rate = float(flow_text) if flow_text else sim_state["flow"]
        pressure = float(pressure_text) if pressure_text else sim_state["pressure"]

        if not all(math.isfinite(v) for v in [age, flow_rate, pressure]):
            raise ValueError("All inputs must be valid finite numbers.")

        if age < 0 or age > 200:
            raise ValueError("Pipe age must be between 0 and 200 years.")

        if material not in ALLOWED_MATERIALS:
            raise ValueError("Please choose a supported pipe material.")

        result = analyze_pipe(brand, age, material, flow_rate, pressure)

        EVENTS.insert(0, {
            "time": datetime.now().strftime("%d %b, %H:%M"),
            "brand": brand,
            "material": material,
            "risk": result["risk"],
            "risk_level": result["risk_level"],
            "flow_rate": flow_rate,
            "pressure": pressure
        })
        del EVENTS[20:]

        return render_template(
            "index.html",
            events=EVENTS,
            result=result,
            error=None,
            show_events=False
        )

    except (ValueError, TypeError, OverflowError) as exc:
        return render_template(
            "index.html",
            events=EVENTS,
            result=None,
            error=str(exc),
            show_events=False
        )


# API endpoints for dashboard live simulation
@app.route("/api/status")
def api_status():
    if not sim_state["demo_paused"]:
        # Add realistic micro-variations
        if sim_state["leak_simulated"]:
            sim_state["flow"] = round(56.4 + random.uniform(-0.6, 0.6), 1)
            sim_state["pressure"] = round(44.1 + random.uniform(-0.5, 0.5), 1)
            sim_state["risk"] = random.randint(78, 86)
        else:
            sim_state["flow"] = round(45.2 + random.uniform(-0.4, 0.4), 1)
            sim_state["pressure"] = round(62.5 + random.uniform(-0.4, 0.4), 1)
            sim_state["risk"] = random.randint(12, 16)

    return jsonify(sim_state)


@app.route("/api/toggle-leak", methods=["POST"])
def api_toggle_leak():
    sim_state["leak_simulated"] = not sim_state["leak_simulated"]
    return jsonify({"success": True, "leak_simulated": sim_state["leak_simulated"]})


@app.route("/api/toggle-demo", methods=["POST"])
def api_toggle_demo():
    sim_state["demo_paused"] = not sim_state["demo_paused"]
    return jsonify({"success": True, "demo_paused": sim_state["demo_paused"]})


@app.route("/events")
def event_history():
    return render_template(
        "index.html",
        events=EVENTS,
        result=None,
        error=None,
        show_events=True
    )


if __name__ == "__main__":
    app.run(debug=True)