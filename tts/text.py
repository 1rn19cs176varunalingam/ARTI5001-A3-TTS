"""Text front-end: normalize_text() -> IPA phonemes (espeak-ng via phonemizer) -> integer ids.
Abbreviation list adapted from keithito/tacotron (https://github.com/keithito/tacotron/blob/master/text/cleaners.py).
"""
import json, logging, re, unicodedata
from num2words import num2words

PAD = "<pad>"
PAD_ID = 0

_ABBREVIATIONS = [(re.compile(rf"\b{abbr}\.", re.IGNORECASE), full) for abbr, full in [
    ("mrs", "misess"), ("mr", "mister"), ("dr", "doctor"), ("st", "saint"), ("co", "company"), ("jr", "junior"),
    ("maj", "major"), ("gen", "general"), ("drs", "doctors"), ("rev", "reverend"), ("lt", "lieutenant"),
    ("hon", "honorable"), ("sgt", "sergeant"), ("capt", "captain"), ("esq", "esquire"), ("ltd", "limited"),
    ("col", "colonel"), ("ft", "fort")]]
_PUNCT_MAP = {"“": '"', "”": '"', "‘": "'", "’": "'", "—": ", ", "–": ", ", "--": ", "}


def _money(amount, unit, sub_unit):
    whole, _, frac = amount.partition(".")
    whole, frac = int(whole or 0), int((frac + "00")[:2]) if frac else 0
    parts = []
    if whole:
        parts.append(f"{num2words(whole)} {unit}{'s' if whole != 1 else ''}")
    if frac:
        parts.append(f"{num2words(frac)} {sub_unit}{'s' if frac != 1 else ''}")
    return ", ".join(parts) or f"zero {unit}s"


def expand_numbers(text):
    text = re.sub(r"(\d),(\d{3})", r"\1\2", text)                                      # 1,000 -> 1000
    text = re.sub(r"\$(\d+(?:\.\d+)?)", lambda m: _money(m.group(1), "dollar", "cent"), text)
    text = re.sub(r"£(\d+(?:\.\d+)?)", lambda m: _money(m.group(1), "pound", "penny"), text)
    text = re.sub(r"\b(\d{1,2}):(\d{2})\b", lambda m: num2words(int(m.group(1))) + (                    # times
        " o'clock" if m.group(2) == "00" else " " + ("oh " if m.group(2)[0] == "0" else "") + num2words(int(m.group(2)))), text)
    text = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: num2words(int(m.group(1)), to="ordinal"), text)
    text = re.sub(r"\b(1[1-9]\d\d|20\d\d)\b", lambda m: num2words(int(m.group(1)), to="year"), text)  # years
    text = re.sub(r"\b\d+\.\d+\b", lambda m: num2words(float(m.group(0))), text)
    text = re.sub(r"\b\d+\b", lambda m: num2words(int(m.group(0))), text)
    return text


def normalize_text(text):
    """Make arbitrary English text look like LJSpeech's normalised transcripts (see notebook §1.1.D)."""
    for src, dst in _PUNCT_MAP.items():
        text = text.replace(src, dst)
    for pattern, full in _ABBREVIATIONS:
        text = pattern.sub(full, text)
    text = expand_numbers(text)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()     # é -> e
    text = text.lower()
    text = re.sub(r"[()\[\]]", ", ", text)                                             # parentheses -> pause
    text = re.sub(r"[^a-z' ,.?!;:-]", " ", text)                                       # drop quotes, symbols
    text = re.sub(r"\s+([,.?!;:])", r"\1", text)                                       # no space before punctuation
    text = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", text)                                # ", ," -> ","
    text = re.sub(r"[,;:]+([.?!])", r"\1", text)                                        # ",." -> "."
    text = re.sub(r"([,;:.?!])(?=[a-z])", r"\1 ", text)                                 # space after punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text.lstrip(",;: ")                                                         # trailing commas kept: prosody


_BACKEND = None


def to_phonemes(texts, njobs=1):
    """List of normalised strings -> list of IPA strings (espeak-ng en-us, stress marks, punctuation kept)."""
    global _BACKEND
    if _BACKEND is None:
        from phonemizer.backend import EspeakBackend
        quiet = logging.getLogger("phonemizer"); quiet.setLevel(logging.ERROR)
        _BACKEND = EspeakBackend("en-us", preserve_punctuation=True, with_stress=True, logger=quiet)
    return _BACKEND.phonemize(list(texts), strip=True, njobs=njobs)


def build_vocab(phoneme_strings):
    symbols = sorted(set("".join(phoneme_strings)))
    return {PAD: PAD_ID, **{s: i + 1 for i, s in enumerate(symbols)}}


def phonemes_to_ids(phonemes, vocab):
    return [vocab[ch] for ch in phonemes if ch in vocab]


def text_to_ids(text, vocab):
    """Arbitrary text -> ids (the full front-end used at inference)."""
    return phonemes_to_ids(to_phonemes([normalize_text(text)])[0], vocab)


def save_vocab(vocab, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=0)


def load_vocab(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
