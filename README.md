# SatTrack · LSTM Trajectory Predictor

A full-stack Flask web application that serves a pre-trained PyTorch LSTM model
for satellite trajectory prediction.

---

## Project Structure

```
satellite_app/
│
├── app.py                         # Flask backend + model inference
├── requirements.txt               # Python dependencies
├── lstm_trajectory_model.pth      # ← Place your trained model here
│
└── templates/
    └── index.html                 # Glassmorphic dark-theme UI
```

---

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your trained model
Copy `lstm_trajectory_model.pth` (exported from your Colab notebook) into the
`satellite_app/` root directory (same level as `app.py`).

### 3. Start the server
```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## API Reference

### `POST /predict`

**Request body (JSON):**
```json
{
  "altitude_km":   0.42,
  "inclination":   0.55,
  "eccentricity":  0.01,
  "mean_motion":   0.73
}
```
> All values must be **MinMaxScaler-normalised** to [0, 1], matching the scaler
> fitted during training.

**Response (JSON):**
```json
{
  "predicted_mean_motion": 0.712843,
  "status": "ok",
  "mode": "model"
}
```

| Field                    | Type   | Description                                        |
|--------------------------|--------|----------------------------------------------------|
| `predicted_mean_motion`  | float  | Normalised mean motion prediction from the LSTM    |
| `status`                 | string | `"ok"` or `"error"`                               |
| `mode`                   | string | `"model"` (real inference) or `"demo"` (no `.pth`)|

---

## Model Architecture

```python
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(4, 32, batch_first=True)   # 4 features → 32 hidden
        self.fc   = nn.Linear(32, 1)                   # 32 hidden  → 1 output

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
```

Input tensor shape: `(batch=1, seq=1, features=4)`

---

## Hardware Note

The model is explicitly loaded with `torch.device('cpu')` for smooth
performance on an **Intel i5 11th Gen CPU** without requiring CUDA.
