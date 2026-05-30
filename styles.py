STYLE_PRESETS = {
    "mid_fade": {
        "keywords": ["mid fade","fade haircut"],
        "prompt": "professional mid fade haircut, clean taper, sharp lineup"
    },
    "blonde": {
        "keywords": ["blonde","balayage"],
        "prompt": "luxury blonde balayage hair, sun kissed highlights"
    },
    "curls": {
        "keywords": ["curly","curls"],
        "prompt": "big bouncy salon curls, high volume"
    }
}

def detect_intent(text):
    text = text.lower()

    if "book" in text:
        return "BOOK"

    if "hair" in text or "fade" in text or "style" in text:
        return "STYLE"

    return "CHAT"

def interpret_request(text):
    for key, val in STYLE_PRESETS.items():
        for word in val["keywords"]:
            if word in text:
                return val["prompt"]

    return "clean modern haircut"
