# AI vs. Human

> Classify whether an answer was written by a **human** or **ChatGPT**, and measure **similarity** between the two (cosine similarity on embeddings + BLEU score)  
> Steps: **scrape human answers → generate AI answers → preprocess → feature engineering → train models → compare similarity**

---

### 🔍 **Key Features**

* **Data Collection:** Scrapes real Quora answers and generates corresponding ChatGPT responses using the OpenAI API.
* **Preprocessing:** Cleans, tokenizes, and structures text data for analysis.
* **Feature Engineering:** Extracts linguistic, statistical, and embedding-based features (TF-IDF, Sentence Transformers).
* **Model Training:** Implements ML models such as SVM, Naïve Bayes, and Random Forest to classify human vs AI answers.
* **Similarity Analysis:** Compares human and AI responses using cosine similarity and BLEU to measure overlap in style and semantics.
* **Insights:** Reveals patterns in how ChatGPT’s writing differs from human text across domains and question types.

---

## Repository Structure

```
ai-vs-human-quora/
├─ README.md
├─ requirements.txt
├─ data/
│  ├─ scraped_data.xlsx
│  ├─ scraped_and_ai_data.xlsx
└─ src/
   ├─ scrape_quora.py            # 1. Scrape human answers
   ├─ collect_ai_answers.py      # 2. Generate AI answers using OpenAI API
   ├─ preprocess.py              # 3. Data cleaning, labeling, and splits
   ├─ feature_engineering.py     # 4. TF-IDF, embeddings, linguistic features
   ├─ model_training.py          # 5. Model training, evaluation, metrics
   └─ similarity_eval.py         # 6. Cosine/BLEU similarity comparison
```

---

## Installation & Setup

### 1) Clone the repo

```
git clone https://github.com/akshrathod/ai-vs-human.git
cd ai-vs-human-quora
```

### 2) Create a virtual environment & install requirements

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
```

```bash
# macOS / Linux
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3) Set your OpenAI API key

Create a `.env` file in the repo root (or set an environment variable):

```env
OPENAI_API_KEY = ...
```
---

## How to Run

> All commands assume you’re in the repo root and your venv is activated.

### 1) Scrape human answers

```bash
python -m src.scrape_quora \
  --topic "machine learning" \
  --max_pages 5 \
  --out data/scraped_data.xlsx
```

### 2) Collect ChatGPT answers via OpenAI API

```bash
python -m src.collect_ai_ans \
  --in data/scraped_data.xlsx \
  --out data/scraped_and_ai_data.xlsx \
  --model gpt-4o-mini \
  --batch_size 10 \
  --sleep 0.6
```

### 3) Preprocess & split

```bash
python -m src.preprocess \
  --in data/scraped_and_ai_data.xlsx \
  --test_size 0.2 \
  --seed 42
```

### 4) Feature engineering & model training

```bash
python -m src.features \
  --in data/scraped_and_ai_data.xlsx \
  --use_tfidf \
  --use_embeddings \
```

### 5) Model training & evaluation

```bash
python -m src.models \
  --in data/scraped_and_ai_data.xlsx \
  --models "svm,nb,rf,mlp" \
  --results_out results.json
```

### 6) Similarity comparison (AI vs Human answers)

```bash
python -m src.similarity_analysis \
  --in data/scraped_and_ai_data.xlsx \
  --method cosine \
  --embedding_model all-MiniLM-L6-v2 \
  --top_k 10
```

---

## What to Expect

* **Classification:** SVM/NB/MLP baselines with TF-IDF and/or embeddings.
* **Similarity:** Cosine on sentence embeddings and BLEU score.
* **Outputs:** Console metrics (`results.json`), plus any plots you add to notebooks.

---

## Acknowledgements

* Quora content used for academic/demo purposes.
* OpenAI API for AI-generated answers.
