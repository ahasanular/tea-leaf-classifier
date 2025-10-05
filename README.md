# 🍃 Tea Leaf Disease Classifier

An explainable AI system for identifying diseases in tea leaves using prototype-based deep learning. Not just a black box - this model shows you *why* it makes each diagnosis.

## 🎯 What It Does

- **Identifies 7 tea leaf conditions** (6 diseases + healthy)
- **Explains its decisions** with visual heatmaps
- **Detects unfamiliar cases** (out-of-distribution detection)
- **Handles class imbalance** automatically
- **Generates comprehensive reports** with metrics and visualizations

## 🏗️ How It Works

This isn't your typical neural network! We use **prototype learning**:

1. **Learns visual concepts**: The model discovers key patterns (spots, discolorations, textures) that define each disease
2. **Matches against prototypes**: For new images, it finds which learned patterns are present
3. **Shows its work**: Highlights the exact leaf regions that influenced the decision

## ⚡ Quick Start

### 1. Setup Environment
```bash
# Clone and install
git clone https://github.com/ahasanular/tea-leaf-classifier.git
cd tea-leaf-classifier
```

Create python venv
```bash
python3.13 -m venv .venv
```
Activate venv
For linux/mac
```bash
source .venv/bin/activate
```
For windows
```bash
.venv\Scripts\activate
```
Create `.env`
```bash
cp .env.template .env
```
Install dependencies
```bash
pip install -r requirements.txt
```
### 2. Organize Your Data
```bash
data/teaLeafBD/teaLeafBD/
├── Tea algal leaf spot/
├── Brown Blight/
├── Gray Blight/
├── Helopeltis/
├── Red spider/
├── Green mirid bug/
└── Healthy leaf/
```
### 3. Run the System
```bash
python main.py
```

## ⚙️ Key Configuration Tweaks
Change in the .env file
```bash
# Paths
DATA_ROOT=./data/teaLeafBD/teaLeafBD # Load the datasets from 
EXPORT_DIR=./output # Export results, figures, metrics etc. 


# Training duration
EPOCHS = 50                    # Increase for better accuracy

# Model architecture  
PROTOS_PER_CLASS = 12          # More prototypes = more detailed explanations
PROTOTYPE_DIM = 256            # Larger = more complex patterns

# Data handling
USE_OVERSAMPLING = True        # Helps with imbalanced classes
BATCH_SIZE = 16                # Adjust based on GPU memory

# OOD Detection
UNKNOWN_CLASS_NAME = "Helopeltis"  # Which class to treat as "unknown"
```

## 📊 What You Get
After running, check the output/ folder for:
- 📈 Training curves - Monitor model progress
- 🎯 Confusion matrix - See where model gets confused
- 🔍 Prototype overlays - Visual explanations for predictions
- 📋 Classification reports - Precision/recall for each disease
- 🚨 OOD analysis - How well it detects unfamiliar cases

### For further assistance- `ahasanular@gmail.com`