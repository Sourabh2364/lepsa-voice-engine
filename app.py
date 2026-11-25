import io
import re
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from kokoro_onnx import Kokoro

app = FastAPI(title="LEPSA Ultra Voice Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load lightweight ONNX Kokoro-82M engine (~200MB RAM footprint)
kokoro = Kokoro("kokoro-v1.0.int8.onnx", "voices-v1.0.bin")

# Build custom signature blended voices for LEPSA AI
def build_custom_voices():
    custom = {}
    try:
        v_heart = kokoro.get_voice_style("af_heart")
        v_emma = kokoro.get_voice_style("bf_emma")
        v_adam = kokoro.get_voice_style("am_adam")
        v_george = kokoro.get_voice_style("bm_george")

        # LEPSA Nova: Calm, futuristic female signature voice
        custom["lepsa_nova"] = (0.65 * v_heart) + (0.35 * v_emma)
        # LEPSA Titan: Deep, authoritative Jarvis-style male signature voice
        custom["lepsa_titan"] = (0.70 * v_adam) + (0.30 * v_george)
    except Exception as e:
        print("Voice blend fallback active:", e)
    return custom

CUSTOM_VOICES = build_custom_voices()

EMOTION_SPEED = {
    "neutral": 1.0,
    "calm": 0.92,
    "serious": 0.90,
    "excited": 1.08,
    "storyteller": 0.94,
}

# Studio DSP Mastering (Warmth + Presence Boost + Peak Limiter)
def master_audio(audio: np.ndarray, sr: int = 24000) -> np.ndarray:
    if len(audio) == 0:
        return audio
    b_hp, a_hp = butter(2, 75 / (0.5 * sr), btype="high")
    audio = lfilter(b_hp, a_hp, audio)

    b_bp, a_bp = butter(2, [2500 / (0.5 * sr), 5000 / (0.5 * sr)], btype="band")
    presence = lfilter(b_bp, a_bp, audio)
    audio = audio + (0.12 * presence)

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = (audio / peak) * 0.89
    return audio.astype(np.float32)

# Clean markdown, <think>, <tool>, and code blocks before speaking
def clean_markdown(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"<tool[^>]*>", "", text)
    text = re.sub(r"```.*?```", " code block ", text, flags=re.S)
    text = re.sub(r"[`*_#>~]", "", text)
    text = re.sub(r"https?://\S+", " link ", text)
    return re.sub(r"\s+", " ", text).strip()

def detect_lang(segment: str, default_lang: str) -> str:
    if re.search(r"[\u0900-\u097F]", segment):
        return "hi"
    return default_lang

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
    cleaned = clean_markdown(text)
    if not cleaned:
        raise HTTPException(status_code=400, detail="Empty text after cleaning")

    base_lang = "en-us" if lang in ("auto", "a", "en") else "hi"
    final_speed = speed * EMOTION_SPEED.get(emotion, 1.0)
    sentences = re.split(r"(?<=[.!?।])\s+", cleaned)

    all_chunks = []
    short_pause = np.zeros(int(24000 * 0.22), dtype=np.float32)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        seg_lang = detect_lang(sentence, base_lang) if lang == "auto" else base_lang
        active_voice = resolve_voice(voice, seg_lang)

        samples, sr = kokoro.create(
            sentence,
            voice=active_voice,
            speed=final_speed,
            lang=seg_lang,
        )
        if samples is not None and len(samples) > 0:
            all_chunks.append(samples)
        all_chunks.append(short_pause)

    full_audio = np.concatenate(all_chunks) if all_chunks else np.zeros(24000, dtype=np.float32)
    mastered = master_audio(full_audio, 24000)

    buf = io.BytesIO()
    sf.write(buf, mastered, 24000, format="WAV")
    buf.seek(0)
    return buf

class TTSRequest(BaseModel):
    text: str
    voice: str = "lepsa_nova"   # lepsa_nova, lepsa_titan, af_heart, am_adam, hf_alpha, hm_omega
    lang: str = "auto"          # auto, en-us, hi
    emotion: str = "neutral"    # neutral, calm, serious, excited, storyteller
    speed: float = 1.0

@app.get("/")
def health():
    return {
        "engine": "LEPSA Ultra Voice Engine",
        "runtime": "Kokoro-82M ONNX + Studio DSP",
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
  
from fastapi.responses import HTMLResponse

@app.get("/test", response_class=HTMLResponse)
def test_player():
    return """
    <html>
    <body style="background:#0b0f19;color:#fff;font-family:sans-serif;padding:24px;">
      <h2>LEPSA Ultra Voice Engine Test</h2>
      <textarea id="txt" rows="3" style="width:100%;padding:10px;border-radius:8px;">Namaste Saurav, LEPSA Ultra Voice Engine ab live hai!</textarea><br><br>
      <select id="voice" style="padding:10px;border-radius:8px;">
        <option value="lepsa_nova">lepsa_nova (Signature Female)</option>
        <option value="lepsa_titan">lepsa_titan (Signature Male)</option>
        <option value="hf_alpha">hf_alpha (Hindi Female)</option>
        <option value="hm_omega">hm_omega (Hindi Male)</option>
      </select>
      <button onclick="playVoice()" style="padding:10px 20px;background:#3b82f6;color:#fff;border:none;border-radius:8px;margin-left:8px;">Play Voice</button>
      <p id="status" style="color:#94a3b8;margin-top:16px;">Ready</p>
      <audio id="player" controls style="width:100%;margin-top:12px;"></audio>
      <script>
        async function playVoice() {
          const st = document.getElementById('status');
          st.innerText = 'Generating studio audio...';
          try {
            const res = await fetch('/tts', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({
                text: document.getElementById('txt').value,
                voice: document.getElementById('voice').value,
                lang: 'auto'
              })
            });
            if (!res.ok) {
              st.innerText = 'Error: ' + await res.text();
              return;
            }
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            const p = document.getElementById('player');
            p.src = url;
            p.play();
            st.innerText = 'Playing!';
          } catch (e) {
            st.innerText = 'Failed: ' + e;
          }
        }
      </script>
    </body>
    </html>
    """
    
