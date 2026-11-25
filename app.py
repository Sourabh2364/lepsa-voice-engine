import gc
import io
import re
import urllib.parse
import urllib.request
import json
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from kokoro_onnx import Kokoro

app = FastAPI(title="LEPSA Ultra Voice Engine", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

kokoro = Kokoro("kokoro-v1.0.int8.onnx", "voices-v1.0.bin")

def build_custom_voices():
    custom = {}
    try:
        v_heart = kokoro.get_voice_style("af_heart")
        v_emma = kokoro.get_voice_style("bf_emma")
        v_adam = kokoro.get_voice_style("am_adam")
        v_george = kokoro.get_voice_style("bm_george")

        custom["lepsa_nova"] = (0.65 * v_heart) + (0.35 * v_emma)
        custom["lepsa_titan"] = (0.70 * v_adam) + (0.30 * v_george)
    except Exception as e:
        print("Voice blend fallback:", e)
    return custom

CUSTOM_VOICES = build_custom_voices()

try:
    kokoro.create("Ready.", voice="af_heart", speed=1.0, lang="en-us")
    gc.collect()
except Exception:
    pass

EMOTION_SPEED = {
    "neutral": 1.0,
    "calm": 0.92,
    "serious": 0.90,
    "excited": 1.08,
    "storyteller": 0.94,
}

# Common Hinglish markers to detect Roman Hindi sentences
HINGLISH_WORDS = {
    "hai", "hain", "ho", "hoon", "hu", "tha", "thi", "the", "kya", "kyu", "kyun",
    "kaise", "kaisa", "kaisi", "kab", "kahan", "kaun", "kis", "ko", "ka", "ki", "ke",
    "se", "me", "mein", "par", "pe", "aur", "ya", "lekin", "parantu", "magar",
    "nahi", "nhi", "na", "mat", "haan", "ha", "ji", "namaste", "shukriya", "dhanyavad",
    "kar", "karo", "karna", "karke", "raha", "rahi", "rahe", "gaya", "gayi", "gaye",
    "mera", "meri", "mere", "tera", "teri", "tumhara", "aapka", "aapki", "aap", "tum",
    "main", "hum", "yeh", "ye", "woh", "wo", "ab", "abhi", "bhi", "hi", "toh", "to",
    "acha", "accha", "theek", "thik", "bahut", "bohot", "jyada", "zyada", "kam",
    "batao", "bataiye", "sunao", "bolo", "bol", "dekho", "suno", "samajh", "bhai", "dost"
}

def is_hinglish(text: str) -> bool:
    if re.search(r"[\u0900-\u097F]", text):
        return True
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if not words:
        return False
    hits = sum(1 for w in words if w in HINGLISH_WORDS)
    return hits >= 2 or (len(words) <= 4 and hits >= 1)

# Zero-RAM Roman Hinglish -> Native Devanagari Transliterator for Authentic Indian Accent
def hinglish_to_devanagari(text: str) -> str:
    if re.search(r"[\u0900-\u097F]", text):
        return text
    try:
        encoded = urllib.parse.quote(text)
        url = f"https://inputtools.google.com/request?text={encoded}&itc=hi-t-i0-und&num=1&cp=0&cs=1&ie=utf-8&oe=utf-8"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data[0] == "SUCCESS":
                # Reconstruct converted tokens
                result = text
                for item in data[1]:
                    src_word = item[0]
                    if item[1]:
                        hi_word = item[1][0]
                        result = re.sub(r"\b" + re.escape(src_word) + r"\b", hi_word, result, count=1)
                return result
    except Exception:
        pass
    return text

def master_audio(audio: np.ndarray) -> np.ndarray:
    if len(audio) == 0:
        return audio
    audio = audio - np.mean(audio)
    high_detail = np.diff(audio, prepend=audio[0])
    audio = audio + (0.08 * high_detail)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = (audio / peak) * 0.89
    return audio.astype(np.float32)

def clean_markdown(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"<tool[^>]*>", "", text)
    text = re.sub(r"```.*?```", " code block ", text, flags=re.S)
    text = re.sub(r"[`*_#>~]", "", text)
    text = re.sub(r"https?://\S+", " link ", text)
    return re.sub(r"\s+", " ", text).strip()

def resolve_voice(voice_name: str, lang: str):
    if lang == "hi":
        if voice_name in ("lepsa_nova", "af_heart", "bf_emma"):
            return "hf_alpha"
        if voice_name in ("lepsa_titan", "am_adam", "bm_george"):
            return "hm_omega"
    if voice_name in CUSTOM_VOICES:
        return CUSTOM_VOICES[voice_name]
    return voice_name

def synthesize_audio(
    text: str,
    voice: str = "lepsa_nova",
    lang: str = "auto",
    emotion: str = "neutral",
    speed: float = 1.0,
) -> io.BytesIO:
    cleaned = clean_markdown(text)[:1500]
    if not cleaned:
        raise HTTPException(status_code=400, detail="Empty text")

    final_speed = speed * EMOTION_SPEED.get(emotion, 1.0)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?।])\s+", cleaned) if s.strip()]

    all_chunks = []
    short_pause = np.zeros(int(24000 * 0.20), dtype=np.float32)

    for sentence in sentences:
        if lang == "hi" or (lang == "auto" and is_hinglish(sentence)):
            seg_lang = "hi"
            spoken_text = hinglish_to_devanagari(sentence)
        else:
            seg_lang = "en-us"
            spoken_text = sentence

        active_voice = resolve_voice(voice, seg_lang)

        samples, _ = kokoro.create(
            spoken_text,
            voice=active_voice,
            speed=final_speed,
            lang=seg_lang,
        )
        if samples is not None and len(samples) > 0:
            all_chunks.append(samples)
            all_chunks.append(short_pause)

    full_audio = np.concatenate(all_chunks) if all_chunks else np.zeros(24000, dtype=np.float32)
    mastered = master_audio(full_audio)

    buf = io.BytesIO()
    sf.write(buf, mastered, 24000, format="WAV")
    buf.seek(0)

    del all_chunks, full_audio, mastered
    gc.collect()
    return buf

class TTSRequest(BaseModel):
    text: str
    voice: str = "lepsa_nova"
    lang: str = "auto"
    emotion: str = "neutral"
    speed: float = 1.0

@app.get("/")
def health():
    return {
        "engine": "LEPSA Ultra Voice Engine v2",
        "features": ["Auto-Hinglish-to-Devanagari", "Native Indian Hindi Accent", "Custom Brand Voices"],
        "custom_voices": list(CUSTOM_VOICES.keys()),
        "standard_voices": ["af_heart", "am_adam", "bf_emma", "bm_george", "hf_alpha", "hm_omega"],
        "status": "ONLINE",
    }

@app.post("/tts")
def tts_post(req: TTSRequest):
    wav_io = synthesize_audio(req.text, req.voice, req.lang, req.emotion, req.speed)
    return StreamingResponse(wav_io, media_type="audio/wav")

@app.get("/tts")
def tts_get(
    text: str = Query(...),
    voice: str = Query("lepsa_nova"),
    lang: str = Query("auto"),
    emotion: str = Query("neutral"),
    speed: float = Query(1.0),
):
    wav_io = synthesize_audio(text, voice, lang, emotion, speed)
    return StreamingResponse(wav_io, media_type="audio/wav")
    
