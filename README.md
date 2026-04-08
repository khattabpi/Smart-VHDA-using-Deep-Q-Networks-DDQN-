# 🔗 VHDA — Vertical Handover Decision Agent

> A Deep Reinforcement Learning agent that intelligently decides when to switch between **WiFi** and **5G** networks to maximize Quality of Service (QoS).

---

## 📌 Project Overview

In heterogeneous wireless networks, a device can be simultaneously connected to multiple network types (WiFi, 5G). Deciding **when to switch** (handover) is critical — switching too often wastes resources, and staying on a poor network degrades QoS.

This project trains a **Double Deep Q-Network (DDQN)** agent to make optimal vertical handover decisions based on real-time radio and QoS features.

---

## 🗂️ Project Structure

```
GP/
├── agent.py          # DDQN Agent + QNetwork (PyTorch)
├── env.py            # Gymnasium Environment (VHDAEnv)
├── run.py            # Training loop + result plotting
├── Datasets/
│   └── vhda_synthetic_normalized.csv
└── Results/
    └── ddqn_results.png
```

---

## 🔄 Project Flow

<img width="1888" height="2228" alt="Gemini_Generated_Image_63pasz63pasz63pa" src="https://github.com/user-attachments/assets/03247686-2188-42f3-9b8a-89125be9f2a2" />

## Result

<img width="1000" height="969" alt="Figure_1" src="https://github.com/user-attachments/assets/e99c386d-4582-4768-acdf-d17b366c88e1" />


---

## 📊 Dataset Features

| Feature | Network | Description |
|---|---|---|
| `RSSI_WIFI` | WiFi | Received Signal Strength |
| `SINR_WIFI` | WiFi | Signal-to-Interference-Noise Ratio |
| `RSRP_5G` | 5G | Reference Signal Received Power |
| `SINR_5G` | 5G | Signal-to-Interference-Noise Ratio |
| `throughput_WIFI` | WiFi | Data transfer rate |
| `latency_WIFI` | WiFi | Response delay |
| `throughput_5G` | 5G | Data transfer rate |
| `latency_5G` | 5G | Response delay |

> All features are **normalized** to the range `[0, 1]`. The agent uses the first `500` rows per episode.

---

## 🧠 Agent Architecture

```
Input (8) → FC(256) → ReLU → FC(256) → ReLU → Output (2)
                                                   │
                                        ┌──────────┴──────────┐
                                     Q(WiFi)              Q(5G)
```

**DDQN vs DQN:**
- **Online Network** selects the best action
- **Target Network** evaluates the Q-value (updated every 10 steps)
- This separation **reduces overestimation** and stabilizes training

---

## ⚙️ Hyperparameters

| Parameter | Value |
|---|---|
| Learning Rate | `1e-3` |
| Discount Factor (γ) | `0.99` |
| Epsilon Start | `1.0` |
| Epsilon Min | `0.02` |
| Epsilon Decay | `0.995` |
| Batch Size | `64` |
| Replay Memory | `10,000` |
| Target Update Frequency | `every 10 steps` |
| Episodes | `500` |
| Steps per Episode | `500` |

---

## 🏆 Reward Function

```python
# Handover penalty (unnecessary switch)
if switched and delta <= 0.05:
    reward -= 0.35

# Reward for good QoS
if throughput >= 0.5 and latency <= 0.5:
    reward += qos_score

# QoS Score
qos_score = 0.7 * throughput - 0.3 * latency
```

---

## 🚀 Installation & Usage

```bash
# 1. Clone the repository
git clone https://github.com/your-username/VHDA-DDQN.git
cd VHDA-DDQN

# 2. Install dependencies
pip install torch numpy pandas gymnasium matplotlib

# 3. Run training
python run.py
```

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| `torch` | Neural network (DDQN) |
| `gymnasium` | RL environment interface |
| `pandas` | Dataset loading |
| `numpy` | Numerical operations |
| `matplotlib` | Results visualization |

---

## 📈 Output

After training, a plot is saved to `Results/ddqn_results.png` showing:
1. **Total Reward** per episode
2. **Average QoS** per episode
3. **Unnecessary Handovers** per episode

The console also prints a **First vs Last Episode** comparison to show how much the agent improved.

---

## 👨‍💻 Author

**Abdulrahman** — Graduation Project  
Faculty of Engineering — Communications & Electronics

---

## 📄 License

This project is for academic purposes only.
