# 03 — Data Models

## 1. Eval question (input dataset)

File: `data/eval_questions.json` — a list of objects.

```json
{
  "id": "hotpotqa_0001",
  "question": "What was Apple's R&D spend in 2023?",
  "expected_answer": "$29.9B",
  "gold_doc_ids": ["doc_00452", "doc_00789"],
  "type": "single_hop"
}
```

Fields:
- `id` — unique string
- `question` — raw question text
- `expected_answer` — short-form gold answer, used for exact/fuzzy match scoring
- `gold_doc_ids` — list of document/chunk IDs known to contain the answer (used for Recall@k / MRR)
- `type` — optional, e.g. `single_hop` / `multi_hop`, useful for slicing results later

## 2. Corpus chunk (retrieval index unit)

```json
{
  "doc_id": "doc_00452",
  "text": "Apple reported research and development expenses of $29.9 billion for fiscal year 2023...",
  "source": "AAPL_10K_2023.pdf",
  "metadata": { "page": 45, "section": "MD&A" }
}
```

- `doc_id` must match the IDs used in `gold_doc_ids` above
- Keep chunks reasonably sized (200–500 tokens) — chunking strategy is a config knob, not a hardcoded constant

## 3. Pipeline config (YAML)

```yaml
name: hybrid_rerank
retriever:
  mode: hybrid          # dense | bm25 | hybrid
  dense_model: text-embedding-3-large   # or local model
  rrf_k: 60
  top_k: 20
rewriter:
  enabled: false
  strategy: multi_query   # multi_query | hyde
reranker:
  enabled: true
  model: bge-reranker-v2-m3
  top_n: 5
generator:
  model: claude-sonnet-4-6
  max_tokens: 512
```

## 4. Per-question result row (output of `run_eval.py`)

CSV columns (one row per question per mode):

| column | meaning |
|---|---|
| `question_id` | matches eval question `id` |
| `mode` | config name, e.g. `hybrid_rerank` |
| `recall_at_5` | 1.0 if any gold doc in top-5 retrieved, else 0.0 |
| `mrr` | reciprocal rank of first gold doc found |
| `faithfulness` | 0–1 LLM-judge score |
| `answer_relevance` | 0–1 LLM-judge score |
| `latency_ms` | total pipeline latency for this question |
| `generated_answer` | the model's final answer text |
| `retrieved_doc_ids` | list of doc IDs retrieved, in rank order |

## 5. Final comparison table (output of `compare_results.py`)

| mode | recall@5 | mrr | faithfulness | avg_latency_ms |
|---|---|---|---|---|
| dense_only | 0.61 | 0.54 | 0.71 | 1200 |
| bm25_only | 0.58 | 0.49 | 0.68 | 400 |
| hybrid_rrf | 0.74 | 0.67 | 0.79 | 1500 |
| hybrid_rerank | 0.81 | 0.75 | 0.85 | 2300 |
| hybrid_rewrite_rerank | ... | ... | ... | ... |

This table is the single most important artifact of the whole project — it's what goes in the README and the CV writeup.
