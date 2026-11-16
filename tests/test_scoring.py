from scoring import aggregate, time_weight

LABELS = {"negative": 0, "neutral": 1, "positive": 2}

def article(title, label, score, confidence, time_w=1.0):
    probs = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
    probs[label] = confidence if label != "neutral" else 1.0
    return {
        "datetime": title,
        "title": title,
        "probs": probs,
        "score": score,
        "confidence": confidence,
        "time_w": time_w,
        "label": label,
    }

def test_half_life_weight():
    assert time_weight(0, 1) == 1
    assert time_weight(1, 1) == 0.5
    assert time_weight(2, 1) == 0.25

def test_neutral_article_does_not_move_the_score():
    scored = [
        article("중립", "neutral", 0.0, 0.0),
        article("호재", "positive", 1.0, 1.0),
    ]
    out = aggregate(scored, LABELS, 1.0)
    assert out["score_-1to1"] == 1.0
    assert out["score_10pt"] == 10.0
    assert out["counts"] == {"negative": 0, "neutral": 1, "positive": 1}

def test_older_article_weighs_half():
    scored = [
        article("최신 호재", "positive", 1.0, 1.0, time_w=1.0),
        article("하루 전 악재", "negative", -1.0, 1.0, time_w=0.5),
    ]
    out = aggregate(scored, LABELS, 1.0)
    assert out["score_-1to1"] == 0.3333
    assert out["score_10pt"] == 6.67

def test_score_ends():
    negative = aggregate([article("악재", "negative", -1.0, 1.0)], LABELS, 1.0)
    flat = aggregate([article("중립쪽", "negative", 0.0, 1.0)], LABELS, 1.0)
    positive = aggregate([article("호재", "positive", 1.0, 1.0)], LABELS, 1.0)
    assert negative["score_10pt"] == 0.0
    assert flat["score_10pt"] == 5.0
    assert positive["score_10pt"] == 10.0
