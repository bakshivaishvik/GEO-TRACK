"""
Space Situational Awareness (SSA) Dashboard
Flask backend with PyTorch LSTM inference + CelesTrak orbital mechanics
CPU-optimized inference | WebGL 3D rendering via Plotly.js
ENHANCED: Model Predictive Sampling for collision avoidance
"""

import os, json, time, threading
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, render_template, jsonify, request
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
MU          = 398600.4418      # Earth's gravitational parameter (km³/s²)
R_EARTH     = 6371.0           # km
SAFE_DIST   = 50.0             # km – collision safety threshold
SEQ_LEN     = 5                # LSTM sequence length
FEATURES    = ['altitude_km','inclination','eccentricity','mean_motion']
DEVICE      = torch.device('cpu')

# ─── LSTM Model (mirrors notebook architecture exactly) ────────────────────────
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(4, 32, batch_first=True)
        self.fc   = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ─── Global state (loaded once at startup) ────────────────────────────────────
model  = None
scaler = None
catalog = []          # list of dicts describing known objects
_lock  = threading.Lock()


# ─── Physics helpers ──────────────────────────────────────────────────────────
def mean_motion_to_altitude(n_rev_per_day: float) -> float:
    """Convert TLE mean_motion (rev/day) → altitude (km)."""
    n_rad = n_rev_per_day * 2 * np.pi / 86400
    a = (MU / n_rad**2) ** (1/3)
    return a - R_EARTH


def orbital_elements_to_cartesian(alt_km, inc_deg, ecc, mean_motion_rev_day, steps=60):
    """
    Simplified Keplerian propagation → 3-D Cartesian positions (km).
    Returns arrays x, y, z each of shape (steps,).
    """
    a       = (alt_km + R_EARTH)          # semi-major axis (km)
    b       = a * np.sqrt(1 - ecc**2)     # semi-minor axis
    inc     = np.radians(inc_deg)
    n_rad   = mean_motion_rev_day * 2 * np.pi / 86400  # rad/s
    T       = 2 * np.pi / n_rad            # period (s)
    ts      = np.linspace(0, T, steps)

    # Mean anomaly → eccentric anomaly (Newton-Raphson, 10 iters)
    M  = n_rad * ts
    E  = M.copy()
    for _ in range(10):
        E = E - (E - ecc * np.sin(E) - M) / (1 - ecc * np.cos(E))

    # Perifocal coords
    xp = a * (np.cos(E) - ecc)
    yp = b * np.sin(E)

    # Rotate to ECI (simplified: RAAN=0, arg_perigee=0)
    x =  xp * np.cos(inc)
    y =  yp
    z =  xp * np.sin(inc)
    return x.tolist(), y.tolist(), z.tolist()


# ─── Scaler bootstrap (fit on synthetic realistic ranges) ────────────────────
def build_scaler():
    rng = np.random.default_rng(42)
    N = 2000
    alt = rng.uniform(300, 2000, N)
    inc = rng.uniform(0, 120, N)
    ecc = rng.uniform(0, 0.01, N)
    mm  = rng.uniform(13.0, 16.5, N)
    data = np.column_stack([alt, inc, ecc, mm])
    sc = MinMaxScaler()
    sc.fit(data)
    return sc


# ─── CelesTrak-style catalog (synthetic but physically realistic) ──────────────
def build_catalog():
    rng = np.random.default_rng(7)
    objects = []
    # ISS
    objects.append(dict(norad=25544, name="ISS (ZARYA)",
                        alt=408, inc=51.6, ecc=0.0006, mm=15.49,
                        category="space_station", rcs="LARGE"))
    # Starlink cluster (some placed in potentially threatening orbits)
    for i in range(30):
        objects.append(dict(
            norad=44700+i, name=f"STARLINK-{1000+i}",
            alt=rng.uniform(540,560), inc=rng.uniform(53,53.2),
            ecc=rng.uniform(0.00001,0.001), mm=rng.uniform(15.05,15.15),
            category="starlink", rcs="SMALL"))
    # GPS satellites (MEO)
    for i in range(8):
        objects.append(dict(
            norad=40534+i, name=f"GPS BIIF-{i+1}",
            alt=rng.uniform(19900,20300), inc=rng.uniform(54,56),
            ecc=rng.uniform(0.001,0.009), mm=rng.uniform(2.0,2.07),
            category="navigation", rcs="LARGE"))
    # Debris - some placed in crossing orbits to trigger alerts
    for i in range(25):
        objects.append(dict(
            norad=99000+i, name=f"DEBRIS-{i+1}",
            alt=rng.uniform(350,900), inc=rng.uniform(20,100),
            ecc=rng.uniform(0.001,0.02), mm=rng.uniform(14.2,16.2),
            category="debris", rcs="SMALL"))
    # Active misc
    for i in range(15):
        objects.append(dict(
            norad=37820+i, name=f"SAT-{37820+i}",
            alt=rng.uniform(400,800), inc=rng.uniform(10,98),
            ecc=rng.uniform(0.0001,0.005), mm=rng.uniform(14.5,15.8),
            category="active", rcs="MEDIUM"))
    
    # Add a few threatening debris objects that will cause collision alerts
    threatening_objects = [
        dict(norad=99901, name="THREAT-DEBRIS-01", alt=412, inc=52.1, ecc=0.0008, mm=15.51, category="debris", rcs="SMALL"),
        dict(norad=99902, name="THREAT-DEBRIS-02", alt=405, inc=51.9, ecc=0.0005, mm=15.48, category="debris", rcs="SMALL"),
        dict(norad=99903, name="THREAT-DEBRIS-03", alt=418, inc=52.3, ecc=0.0012, mm=15.52, category="debris", rcs="SMALL"),
    ]
    objects.extend(threatening_objects)
    
    return objects


# ─── LSTM-based trajectory prediction ────────────────────────────────────────
def predict_trajectory(target: dict, steps: int = 60):
    """
    Feed orbital elements through LSTM to get predicted mean_motion sequence,
    then propagate to Keplerian 3-D positions.
    """
    row = np.array([[target['alt'], target['inc'], target['ecc'], target['mm']]])
    row_scaled = scaler.transform(row)[0]

    seq = np.tile(row_scaled, (SEQ_LEN, 1))          # (5,4) repeated
    x_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1,5,4)

    predicted_mm_scaled_list = []
    current_seq = seq.copy()

    with torch.no_grad():
        for _ in range(steps):
            inp   = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0)
            pred  = model(inp).item()
            predicted_mm_scaled_list.append(pred)
            # shift window
            new_row = current_seq[-1].copy()
            new_row[3] = pred                         # update mean_motion
            current_seq = np.vstack([current_seq[1:], new_row])

    # Inverse transform mean_motion predictions
    dummy = np.zeros((steps, 4))
    dummy[:, 3] = predicted_mm_scaled_list
    # clamp to valid scaled range
    dummy[:, 3] = np.clip(dummy[:, 3], 0, 1)
    unscaled = scaler.inverse_transform(dummy)
    pred_mm = unscaled[:, 3]                          # predicted mean_motions

    # Build trajectory using predicted mm with original orbital geometry
    xs, ys, zs = [], [], []
    for mm_pred in pred_mm:
        alt_pred = mean_motion_to_altitude(max(mm_pred, 0.1))
        xi, yi, zi = orbital_elements_to_cartesian(
            alt_pred, target['inc'], target['ecc'], mm_pred, steps=1)
        xs.extend(xi); ys.extend(yi); zs.extend(zi)

    return np.array(xs), np.array(ys), np.array(zs), pred_mm.tolist()


# ─── Miss-distance computation ────────────────────────────────────────────────
def compute_miss_distances(traj_x, traj_y, traj_z, target_norad):
    """
    For every catalog object (except target), compute representative
    3-D position and minimum distance to the predicted trajectory.
    Returns list of dicts sorted by miss_distance ascending.
    """
    traj = np.column_stack([traj_x, traj_y, traj_z])  # (N,3)
    results = []
    for obj in catalog:
        if obj['norad'] == target_norad:
            continue
        ox, oy, oz = orbital_elements_to_cartesian(
            obj['alt'], obj['inc'], obj['ecc'], obj['mm'], steps=1)
        pos = np.array([[ox[0], oy[0], oz[0]]])
        dists = np.linalg.norm(traj - pos, axis=1)
        miss  = float(np.min(dists))
        results.append(dict(
            norad       = obj['norad'],
            name        = obj['name'],
            category    = obj['category'],
            rcs         = obj['rcs'],
            miss_dist   = round(miss, 2),
            safe        = miss > SAFE_DIST,
            alt         = round(obj['alt'], 1),
            inc         = round(obj['inc'], 2),
        ))
    results.sort(key=lambda r: r['miss_dist'])
    return results

def find_maneuver_point(baseline_traj, evasive_traj, threshold_km=10):
    """
    Find the point where evasive trajectory diverges from baseline.
    Returns the 3D coordinates of the maneuver point.
    """
    if len(baseline_traj[0]) != len(evasive_traj[0]):
        return None
    
    for i in range(len(baseline_traj[0])):
        dist = np.sqrt(
            (baseline_traj[0][i] - evasive_traj[0][i])**2 +
            (baseline_traj[1][i] - evasive_traj[1][i])**2 +
            (baseline_traj[2][i] - evasive_traj[2][i])**2
        )
        if dist > threshold_km:
            return {
                'x': evasive_traj[0][i],
                'y': evasive_traj[1][i],
                'z': evasive_traj[2][i],
                'index': i
            }
    return None

def find_safe_trajectory(target, max_attempts=100):
    """
    MODEL PREDICTIVE SAMPLING with divergence tracking.
    """
    # 1. Predict baseline (unmaneuvered) trajectory
    tx, ty, tz, pred_mm = predict_trajectory(target, steps=60)
    baseline_traj = (tx.copy(), ty.copy(), tz.copy())
    miss_list = compute_miss_distances(tx, ty, tz, target['norad'])
    alerts = [m for m in miss_list if not m['safe']]
    maneuver_applied = 0.0
    maneuver_point = None
    
    # 2. If baseline is safe, return immediately
    if len(alerts) == 0:
        print(f"[SSA] Baseline trajectory for NORAD {target['norad']} is SAFE.")
        return tx, ty, tz, pred_mm, miss_list, alerts, maneuver_applied, maneuver_point
    
    # 3. Collision risk detected! Search for safe maneuver
    print(f"[SSA] ⚠️ Collision risk for NORAD {target['norad']}! Searching for safe trajectory...")
    
    maneuvers = [0.02, -0.02, 0.05, -0.05, 0.08, -0.08, 0.12, -0.12, 0.15, -0.15]
    maneuvers.extend([0.03, -0.03, 0.07, -0.07, 0.10, -0.10])
    maneuvers = maneuvers[:max_attempts]
    
    for delta_mm in maneuvers:
        hypothetical_target = target.copy()
        hypothetical_target['mm'] = max(0.1, target['mm'] + delta_mm)
        
        h_tx, h_ty, h_tz, h_pred_mm = predict_trajectory(hypothetical_target, steps=60)
        h_miss_list = compute_miss_distances(h_tx, h_ty, h_tz, target['norad'])
        h_alerts = [m for m in h_miss_list if not m['safe']]
        
        min_miss = h_miss_list[0]['miss_dist'] if h_miss_list else 9999.0
        
        if len(h_alerts) == 0:
            print(f"[SSA] ✅ Safe trajectory FOUND! ΔMM = {delta_mm:+.3f} rev/day")
            # Find where maneuver occurred
            maneuver_point = find_maneuver_point(baseline_traj, (h_tx, h_ty, h_tz))
            return h_tx, h_ty, h_tz, h_pred_mm, h_miss_list, h_alerts, delta_mm, maneuver_point
        
        baseline_min = miss_list[0]['miss_dist']
        if min_miss > baseline_min * 1.5 and min_miss > 100:
            print(f"[SSA] 📈 Accepting improved trajectory: ΔMM = {delta_mm:+.3f} rev/day")
            maneuver_point = find_maneuver_point(baseline_traj, (h_tx, h_ty, h_tz))
            return h_tx, h_ty, h_tz, h_pred_mm, h_miss_list, h_alerts, delta_mm, maneuver_point
    
    print(f"[SSA] ⚠️ No safe trajectory found after {len(maneuvers)} attempts.")
    return tx, ty, tz, pred_mm, miss_list, alerts, maneuver_applied, maneuver_point


# ─── Flask routes ─────────────────────────────────────────────────────────────
@app.route('/')
def index():
    target_list = [dict(norad=o['norad'], name=o['name'],
                        category=o['category']) for o in catalog]
    return render_template('dashboard.html', targets=target_list)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json(force=True)
        norad = int(data.get('norad', catalog[0]['norad']))
        target = next((o for o in catalog if o['norad'] == norad), catalog[0])

        t0 = time.time()
        
        # Get trajectory with maneuver detection
        tx, ty, tz, pred_mm, miss_list, alerts, maneuver_applied, maneuver_point = find_safe_trajectory(target)

        elapsed = round((time.time() - t0) * 1000, 1)
        min_miss = miss_list[0]['miss_dist'] if miss_list else 9999.0

        # Earth sphere generation
        u = np.linspace(0, 2*np.pi, 40)
        v = np.linspace(0, np.pi, 40)
        ex = (R_EARTH * np.outer(np.cos(u), np.sin(v))).tolist()
        ey = (R_EARTH * np.outer(np.sin(u), np.sin(v))).tolist()
        ez = (R_EARTH * np.outer(np.ones(40), np.cos(v))).tolist()

        response = dict(
            target          = target,
            trajectory      = dict(x=tx.tolist(), y=ty.tolist(), z=tz.tolist()),
            pred_mm         = pred_mm,
            earth           = dict(x=ex, y=ey, z=ez),
            miss_distances  = miss_list[:20],
            alerts          = alerts,
            min_miss        = min_miss,
            collision_free  = len(alerts) == 0,
            inference_ms    = elapsed,
            maneuver_applied= maneuver_applied,
            maneuver_point  = maneuver_point,  # Add this
        )
        
        if maneuver_applied != 0:
            response['message'] = f"Evasive maneuver applied: ΔMM = {maneuver_applied:+.3f} rev/day"
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in predict endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/catalog')
def api_catalog():
    return jsonify(catalog)


# ─── Startup ──────────────────────────────────────────────────────────────────
def load_model():
    global model, scaler, catalog
    model = LSTMModel().to(DEVICE)
    model.eval()
    model_path = 'lstm_trajectory_model.pth'
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print(f"[SSA] Loaded pre-trained weights from {model_path}")
        except Exception as e:
            print(f"[SSA] Could not load model: {e}")
            print("[SSA] Using randomly initialised weights (demo mode).")
    else:
        print("[SSA] No .pth file found — using randomly initialised weights (demo mode).")
        print("[SSA] Place lstm_trajectory_model.pth in the app directory for real inference.")
    
    scaler  = build_scaler()
    catalog = build_catalog()
    print(f"[SSA] Catalog: {len(catalog)} objects  |  Device: {DEVICE}")


if __name__ == '__main__':
    load_model()
    print("\n" + "="*60)
    print("🚀 SSA Dashboard Starting with Model Predictive Sampling...")
    print("📍 Open http://localhost:5000 in your browser")
    print("✨ Collision avoidance active - LSTM will search for safe trajectories")
    print("="*60 + "\n")
    app.run(debug=True, port=5000)