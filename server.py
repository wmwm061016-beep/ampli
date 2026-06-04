from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import io
import os
import numpy as np
import soundfile as sf
import librosa
import speech_recognition as sr

# ==========================================
# 설정 ← 여기에 Gemini API 키 입력!
# ==========================================
API_KEY = os.environ.get("GEMINI_API_KEY", "")

app = FastAPI(title="Ampli - Multimodal AI Counselor")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

chat_history = []

AMPLIE_PERSONA = """
당신은 사용자의 내면을 탐구하고 창의적인 사고를 돕는 대화 전문가 '엠플리(Ampli)'입니다.
당신은 유저의 텍스트뿐만 아니라, 정교한 페이스 메시 센서 데이터를 통해 유저의 눈빛, 미간, 입술의 미세한 변화를 실시간으로 완벽하게 감지하고 있습니다.

[★ 초고도 비언어적 다차원 피드백 지침]
매 턴마다 아래의 3단계 프로세스(관찰 ➔ 인지적 해석 ➔ 발화 설계)를 거쳐 답변의 톤앤매너와 공감의 깊이를 실시간으로 렌더링하세요.

1. [미간을 깊게 찌푸림] → 인지적 과부하: 대화 템포를 늦추고, 짧게 분절하여 여백을 제공.
2. [입꼬리가 올라간 미소] → 지적 유희: 에너지를 끌어올려 아이디어를 확장하는 질문 제시.
3. [입술을 굳게 다물거나 처짐] → 심리적 방어: 무게감 있는 톤으로 사유의 공간 제공.
4. [입을 약간 벌림] → 스키마 파괴: 역설적이거나 직관적인 통찰의 질문을 던짐.

[음성 분석 지침]
1. [말이 빠름/큰 목소리] → 도파민 과각성: 에너지 매칭, 빠른 부스팅 질문.
2. [말이 느림/작은 목소리] → 심연 탐색: 톤을 낮추고 조심스럽게 수용.
3. [긴 침묵/한숨] → 인지적 공백: 침묵을 인정하고 안전한 공간 제공.
4. [음성 떨림/말끝 흐림] → 취약성 노출: 분석 배제, 감정 수용.

# RULES
- 모든 응답은 반드시 질문으로 마무리한다.
- 사용자 말을 1~2문장 요약 + 공감 후 깊이 있는 개방형 질문 1개만 던진다.
- 해결책 직접 제시 금지 (2회 이상 반복 요구 시 제공).
- 8~11번째 대화 전후로 마무리 준비, 종료 시 구조화된 최종 결과물 제공.
- 금지: "MAR 수치", "데시벨" 등 기계적 표현. 오직 따뜻한 인간 관찰자의 언어 사용.
"""

class ChatRequest(BaseModel):
    text: str
    face_state: str = ""
    voice_state: str = "[음성: 텍스트 입력]"

class TTSRequest(BaseModel):
    text: str

def analyze_audio_features(wav_bytes: bytes, text: str) -> str:
    try:
        data, samplerate = sf.read(io.BytesIO(wav_bytes))
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        duration = len(data) / samplerate
        if duration == 0:
            return "[음성: 분석 불가]"
        rms_frames = librosa.feature.rms(y=data)[0]
        mean_rms = np.mean(rms_frames)
        volume_state = "안정적인 톤"
        if mean_rms > 0.08:
            volume_state = "에너지가 높은 큰 목소리"
        elif mean_rms < 0.015:
            volume_state = "조심스럽고 작은 목소리"
        speed = len(text) / duration if duration > 0 else 0
        speed_state = "보통 속도"
        if speed > 8.0:
            speed_state = "말이 빠름"
        elif speed < 3.5:
            speed_state = "말이 느림"
        non_mute_intervals = librosa.effects.split(data, top_db=30)
        speaking_duration = sum([(end - start) / samplerate for start, end in non_mute_intervals])
        silence_ratio = 1 - (speaking_duration / duration) if duration > 0 else 0
        pause_state = " / 긴 침묵 동반" if silence_ratio > 0.4 else ""
        affect_state = ""
        split_idx = int(len(rms_frames) * 0.8)
        if split_idx > 0 and len(rms_frames[split_idx:]) > 0:
            if np.mean(rms_frames[split_idx:]) < mean_rms * 0.4:
                affect_state = " / 말끝을 흐림"
        cv = np.std(rms_frames) / mean_rms if mean_rms > 0 else 0
        if cv > 1.8:
            affect_state = " / 음성의 불규칙한 떨림"
        return f"[음성: {volume_state}, {speed_state}{pause_state}{affect_state}]"
    except Exception as e:
        return "[음성: 분석 오류]"

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/chat")
async def chat(req: ChatRequest):
    global chat_history
    prompt = f"[실시간 멀티모달 상태 -> {req.face_state} / {req.voice_state}]\n유저 발화: {req.text}"
    chat_history.append({"role": "user", "parts": [{"text": prompt}]})

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"
    payload = {
        "systemInstruction": {"parts": [{"text": AMPLIE_PERSONA}]},
        "contents": chat_history,
        "generationConfig": {"temperature": 0.7}
    }

    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        ).json()

        if "error" in response:
            chat_history.pop()
            raise HTTPException(status_code=500, detail=response["error"]["message"])

        ai_reply = response["candidates"][0]["content"]["parts"][0]["text"]
        chat_history.append({"role": "model", "parts": [{"text": ai_reply}]})
        return {"reply": ai_reply, "turn": len(chat_history) // 2}

    except requests.exceptions.RequestException as e:
        chat_history.pop()
        raise HTTPException(status_code=503, detail=f"Gemini 연결 오류: {str(e)}")

@app.post("/audio")
async def process_audio(file: UploadFile = File(...)):
    import subprocess, tempfile
    webm_bytes = await file.read()

    # webm → wav 변환
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(webm_bytes)
        webm_path = f.name
    wav_path = webm_path.replace(".webm", ".wav")
    subprocess.run(["ffmpeg", "-y", "-i", webm_path, "-ar", "16000", "-ac", "1", wav_path], capture_output=True)

    with open(wav_path, "rb") as f:
        wav_bytes = f.read()

    r = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
            audio_data = r.record(source)
        text = r.recognize_google(audio_data, language="ko-KR")
    except sr.UnknownValueError:
        raise HTTPException(status_code=400, detail="음성을 인식하지 못했습니다.")
    except sr.RequestError:
        raise HTTPException(status_code=503, detail="구글 STT 서버에 연결할 수 없습니다.")
    voice_state = analyze_audio_features(wav_bytes, text)
    return {"text": text, "voice_state": voice_state}

@app.post("/reset")
async def reset_history():
    global chat_history
    chat_history = []
    return {"status": "ok"}

@app.get("/health")
async def health():
    return {"status": "running", "turns": len(chat_history) // 2}
