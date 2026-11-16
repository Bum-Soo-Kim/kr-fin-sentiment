def time_weight(age_days: float, half_life_days: float) -> float:
    return 0.5 ** (age_days / half_life_days)

MARKET_WEIGHT = 0.25

def stock_needles(query: str, official: str) -> list[str]:
    needles = []
    for name in (official, query):
        if not name or (name.isdigit() and len(name) == 6):
            continue
        if name not in needles:
            needles.append(name)
    return needles

def mentions_stock(title: str, text: str, needles: list[str]) -> bool:
    if not needles:
        return True
    blob = f"{title}\n{text}"
    return any(needle in blob for needle in needles)

def apply_market_weight(time_w: float, mentioned: bool) -> float:
    return time_w if mentioned else time_w * MARKET_WEIGHT

def aggregate(scored, label_to_id, half_life_days: float) -> dict:
    num = den = 0.0
    prob_num = {label: 0.0 for label in label_to_id}
    for art in scored:
        w = art["time_w"] * art["confidence"]
        num += art["score"] * w
        den += w
        for label, p in art["probs"].items():
            prob_num[label] += p * w
    raw = num / den if den else 0.0

    counts = {label: 0 for label in label_to_id}
    for art in scored:
        counts[art["label"]] += 1

    articles = [{
        "datetime": art["datetime"],
        "title": art["title"],
        "probs": {k: round(v, 4) for k, v in art["probs"].items()},
        "score": round(art["score"], 4),
        "weight": round(art["time_w"], 4),
    } for art in scored]
    articles.sort(key=lambda a: a["datetime"], reverse=True)
    return {
        "half_life_days": half_life_days,
        "counts": counts,
        "model_probs": {k: round(v / den, 4) if den else 0.0 for k, v in prob_num.items()},
        "score_-1to1": round(raw, 4),
        "score_10pt": round((raw + 1) / 2 * 10, 2),
        "articles": articles,
    }
