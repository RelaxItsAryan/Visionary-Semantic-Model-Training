# 🎯 Objective - Offroad Semantic Segmentation

**Duality AI Falcon Platform | UGV Desert Navigation | Team Visionary**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://visionary-semantic-model-training.streamlit.app/)

This project is a Machine Learning-based Image Classification System deployed using Streamlit.
It classifies desert images into the following categories:

-Trees
<br>-Lush Bushes
<br>-Dry Grass
<br>-Dry Bushes
<br>-Ground Clutter
<br>-Flowers
<br>-Logs
<br>-Rocks
<br>-Landscape
<br>-Sky

The model was trained on a synthetic desert dataset generated from Falcon’s Digital Twin platform

## How to Test
1. Go to the Deployed link and can use this
2. or can run locally
3. Run locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **mIoU** | 45.04% |
| **mAP@50** | 52.0% |
| **Best Loss** | 0.8159 |

## 🎨 Features

- **Dual-Stream Visualization**: Side-by-side RGB vs Segmentation mask
- **10 Class Legend**: Toggle visibility of each terrain class
- **Real-Time Telemetry**: mIoU gauge, latency monitor, loss curves
- **Failure Analysis**: Uncertainty heatmaps for model debugging
- **Safety Score**: Path planning insight based on terrain analysis

## 🏷️ Segmentation Classes

| Class | Color | Type |
|-------|-------|------|
| Trees | 🟢 Green | Neutral |
| Lush Bushes | 🟢 Emerald | Neutral |
| Dry Grass | 🟡 Amber | Neutral |
| Dry Bushes | 🟠 Orange | Neutral |
| Ground Clutter | 🟣 Purple | ⚠️ Obstacle |
| Flowers | 🩷 Pink | Neutral |
| Logs | 🟤 Brown | ⚠️ Obstacle |
| Rocks | ⚪ Slate | ⚠️ Obstacle |
| Landscape | 🔵 Cyan | ✅ Safe |
| Sky | 🔵 Blue | Neutral |

## 🛠️ Technical Stack

- **Backbone**: DINOv2 ViT-B/14 (768-dim embeddings)
- **Decoder**: ConvNeXt-style segmentation head
- **Framework**: PyTorch 2.0+ with CUDA
- **UI**: Streamlit + Plotly

## 📁 Files

```
├── app.py                      # Streamlit dashboard
├── best_segmentation_model.pth # Trained model weights
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
```

## 🏆 GHR 2.0 Hackathon

**Team Visionary** | Off-road Semantic Segmentation Challenge

---
*Powered by DINOv2 + PyTorch*
collaborators: Aryan Amit Arya, Prateek Das, Dilisha, Rakshit Raj.
