
FastAPI-based Face Verification Service with Evaluation on LFW, CALFW, and CPLFW datasets

# 🧠 Overview

This project implements a complete face verification pipeline composed of:

1. REST API Server (FastAPI)
2. Client-side evaluation framework
3. Benchmark testing on standard datasets

The system evaluates deep face recognition models using cosine similarity and reports:

- Accuracy
- False Match Rate (FMR)
- False Non-Match Rate (FNMR)
- Runtime performance


# 🏗 Project Structure

```text
face-verification-project/
│
├── server/
│   ├── requirements.txt
│   └── app/
│       ├── main.py
│       ├── model.py
│       ├── schemas.py
│       └── utils.py
│
├── client/
│   ├── datasets/                 
│   ├── evaluate.py
│   ├── lfw_loader.py
│   ├── calfw_loader.py
│   ├── cplfw_loader.py
│   ├── metrics.py
│   └── results.json
│
└── README.md
```


## Server – Face Verification API

Endpoint:

POST /verify

### Request Parameters

- image1 → first image file
- image2 → second image file
- model_name → string (e.g., "buffalo_l")

### Response Example

```

{
"similarity": 0.8734
}

```


## Client – Evaluation Framework

The client:

- Loads dataset pairs
- Sends image pairs to the server
- Collects similarity scores
- Computes evaluation metrics
- Measures runtime

# 📚 Supported Datasets

- LFW
- CALFW (Cross-Age)
- CPLFW (Cross-Pose)

Each dataset contains 6000 verification pairs.

# 📊 Evaluation Metrics

Accuracy  
FMR (False Match Rate)  
FNMR (False Non-Match Rate)

# ⚙️ Installation

## Clone Repository

```

git clone https://github.com/mehranghaffarian/insightface-server-evaluation

```

## Setup Server

```

cd server
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

pip install -r requirements.txt

```

## Run Server

```

uvicorn app.main:app --reload

```

Server runs at:

http://127.0.0.1:8000

## Run Evaluation

Open new terminal:

```

cd client
python evaluate.py

```


# 🧪 Example Results (threshold = 0.4)

| Dataset | Model | Accuracy | FMR | FNMR |
|----------|--------|----------|------|--------|
| LFW | buffalo_l | 0.979 | 0.000 | 0.042 |
| CALFW | buffalo_l | 0.5427 | 0.000 | 0.1467 |
| CPLFW | buffalo_l | 0.8478 | 0.000 | 0.3043 |
| LFW | buffalo_s | 0.979 | 0.000 | 0.042 |
| CALFW | buffalo_s | 0.5427 | 0.000 | 0.1467 |
| CPLFW | buffalo_s | 0.8478 | 0.000 | 0.3043 |


# 📈 Observations

- High performance on LFW.
- Significant performance drop on CALFW (age variation).
- Moderate drop on CPLFW (pose variation).
- FMR = 0 due to conservative threshold (0.4).
- Average inference time per pair: ~0.6–0.8 seconds.

# 🛠 Technologies

- Python
- FastAPI
- Uvicorn
- InsightFace
- NumPy
- OpenCV
- Requests
- tqdm

# 📌 Notes

- venv is excluded from version control.
- Datasets are not included due to size.
- Ensure datasets are extracted and properly placed before evaluation.

