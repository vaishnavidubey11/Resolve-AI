import re

STOPWORDS = frozenset("""a an and are as at be but by can could did do does for from had has have how i if in into is it its
me my of on or our so than that the their them then there these they this to us was we were what when where which who
whom why will with would you your about any get got just please tell know want need like also very more most some
sure cant dont isnt arent whats hello hi thanks thank""".split())


def stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower().replace("'", ""))
    return [stem(w) for w in words if w not in STOPWORDS and len(w) > 1]


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]
