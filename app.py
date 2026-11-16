import threading
from datetime import datetime

import requests
from fastapi import FastAPI, Query

import finbert

app = FastAPI(title="kr-fin-sentiment")
_bundle = None
_lock = threading.Lock()

def model_bundle():
    global _bundle
    if _bundle is None:
        with _lock:
            if _bundle is None:
                _bundle = finbert.load_model()
    return _bundle

def to_json(name: str, result: dict) -> dict:
    if result.get("score_10pt") is None:
        return {"name": name, "code": result["code"], "n": 0, "score_10pt": None}
    articles = []
    for art in result["articles"]:
        articles.append({
            "datetime": art["datetime"].isoformat(),
            "title": art["title"],
            "probs": art["probs"],
            "score": art["score"],
            "weight": art["weight"],
            "market": art.get("market", False),
        })
    return {
        "name": name,
        "code": result["code"],
        "n": result["n"],
        "as_of": result["as_of"].isoformat(),
        "model_probs": result["model_probs"],
        "counts": result["counts"],
        "score_-1to1": result["score_-1to1"],
        "score_10pt": result["score_10pt"],
        "articles": articles,
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/sentiment")
def sentiment(q: str = Query(min_length=1)):
    tok, model, device, label_to_id = model_bundle()
    names = [part.strip() for part in q.split(",") if part.strip()]
    results = []
    errors = []
    for name in names:
        try:
            result = finbert.news_sentiment(
                name, tok, model, device, label_to_id, now=datetime.now(finbert.KST))
            results.append(to_json(name, result))
        except LookupError as e:
            errors.append({"name": name, "error": str(e)})
        except (KeyError, requests.RequestException) as e:
            errors.append({"name": name, "error": str(e)})
    return {"results": results, "errors": errors}
