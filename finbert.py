import html
import requests
import torch

from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from scoring import aggregate, time_weight

MODEL = "snunlp/KR-FinBert-SC"
H = {"User-Agent": "Mozilla/5.0"}

def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    # id2label = {0: "negative", 1: "neutral", 2: "positive"}
    label_to_id = {label.lower(): int(idx) for idx, label in model.config.id2label.items()}
    return tok, model, device, label_to_id

@torch.no_grad()
def score(text: str, tok, model, device, label_to_id) -> dict:
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
    return {label: float(probs[idx]) for label, idx in label_to_id.items()}

def resolve_code(query: str) -> str:
    if query.isdigit() and len(query) == 6:
        return query
    r = requests.get("https://ac.stock.naver.com/ac",
                     params={"q": query, "target": "stock"}, headers=H).json()
    items = r.get("items") or []
    if not items:
        raise LookupError(f"종목을 찾지 못했습니다: {query}")
    return items[0]["code"]

def fetch_news(code: str, n: int = 20):
    r = requests.get(f"https://m.stock.naver.com/api/news/stock/{code}",
                     params={"pageSize": n, "page": 1}, headers=H).json()
    out = []
    for group in r:
        for it in group["items"]:
            dt = datetime.strptime(it["datetime"], "%Y%m%d%H%M")
            title = html.unescape(it.get("titleFull") or it.get("title") or "")
            body = html.unescape(it.get("body") or "")
            out.append({"datetime": dt, "title": title,
                        "text": f"{title} {body[:200]}".strip()})
    return out

def score_news(news, tok, model, device, label_to_id, half_life_days: float, now: datetime):
    scored = []
    for art in news:
        probs = score(art["text"], tok, model, device, label_to_id)
        signed = probs["positive"] - probs["negative"]
        confidence = probs["positive"] + probs["negative"]
        age_days = (now - art["datetime"]).total_seconds() / 86400
        scored.append({
            "datetime": art["datetime"],
            "title": art["title"],
            "probs": probs,
            "score": signed,
            "confidence": confidence,
            "time_w": time_weight(age_days, half_life_days),
            "label": max(probs, key=probs.get),
        })
    return scored

def news_sentiment(query: str, tok, model, device, label_to_id,
                   n: int = 20, half_life_days: float = 1.0, now: datetime | None = None) -> dict:
    code = resolve_code(query)
    news = fetch_news(code, n)
    if not news:
        return {"code": code, "n": 0, "score_10pt": None}
    if now is None:
        now = datetime.now()
    scored = score_news(news, tok, model, device, label_to_id, half_life_days, now)
    result = aggregate(scored, label_to_id, half_life_days)
    result["code"] = code
    result["n"] = len(news)
    result["as_of"] = now
    return result

def print_result(name: str, result: dict, show: int = 5) -> None:
    code = result["code"]
    if result.get("score_10pt") is None:
        print(f"\n{name} ({code}) 뉴스 없음")
        return
    print(f"\n{name} ({code})  {result['n']}건  {result['as_of']:%Y-%m-%d %H:%M}")
    mp = result["model_probs"]
    c = result["counts"]
    print(f"가중  부정 {mp['negative']:.2f}  중립 {mp['neutral']:.2f}  긍정 {mp['positive']:.2f}")
    print(f"라벨  부정 {c['negative']}  중립 {c['neutral']}  긍정 {c['positive']}")
    print(f"점수  {result['score_-1to1']} (-1~1)  {result['score_10pt']} (10점)")
    for art in result["articles"][:show]:
        title = art["title"]
        if len(title) > 48:
            title = title[:48] + "…"
        p = art["probs"]
        print(f"  {art['datetime']:%m-%d %H:%M}  "
              f"부정 {p['negative']:.2f} 중립 {p['neutral']:.2f} 긍정 {p['positive']:.2f}  "
              f"{art['score']:+.2f}  w={art['weight']:.2f}  {title}")

if __name__ == "__main__":
    print("모델 불러오는 중", flush=True)
    tok, model, device, label_to_id = load_model()
    print("종목명 또는 6자리 코드를 입력하세요. 빈 입력 또는 q로 종료.")
    while True:
        q = input("> ").strip()
        if not q or q.lower() in ("q", "quit", "exit"):
            break
        for name in (part.strip() for part in q.split(",")):
            if not name:
                continue
            try:
                print_result(name, news_sentiment(
                    name, tok, model, device, label_to_id, now=datetime.now()))
            except (LookupError, KeyError, requests.RequestException) as e:
                print(f"조회 실패 ({name}): {e}")
