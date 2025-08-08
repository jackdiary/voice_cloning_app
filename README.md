# AI 음성 클로닝 웹앱 - Flask 기반 TTS/Voice Cloning 서비스

> Coqui TTS(xtts_v2) + Flask로 구현한 실시간 음성 분석·클로닝 웹 애플리케이션

![voicecloning](s 소개

업로드한 참조 음성으로 화자를 클로닝하고, 입력 텍스트를 해당 화자 목소리로 합성하는 Flask 웹앱입니다. 음성 전처리(정규화/무음 제거), 특징 추출(MFCC, F0, 스펙트럼), 다국어 합성(ko 기본)을 지원하며, 파일 업로드·합성·다운로드를 HTTP API와 간단한 웹 UI로 제공합니다.

### ★ 핵심 차별점

| 기능 | 일반 TTS | 본 프로젝트 |
|------|----------|-------------|
| 화자 적응 | 고정 화자 | 참조 음성 기반 음성 클로닝 |
| 전처리 | 단순 리샘플링 | 프리엠퍼시스+정규화+무음제거+최소길이 보정 |
| 분석 기능 | 미제공 | F0, 스펙트럼, MFCC, 길이 등 음성 특징 제공 |
| 배포 용이성 | CLI 위주 | Flask API + 웹 UI |
| 파일 관리 | 수동 정리 | 만료 기준 자동 정리(cleanup API) |

***

## 주요 기능

### 1) 지능형 음성 전처리
- 프리엠퍼시스, 정규화, 무음 구간 제거로 합성 품질 개선
- 3초 미만 참조 음성 자동 확장 처리

### 2) 음성 특징 분석
- 기본 주파수(F0), 스펙트럴 센트로이드/롤오프, MFCC 평균, 길이, 샘플레이트 반환
- 업로드 즉시 JSON으로 피처 확인

### 3) 음성 클로닝 합성
- Coqui TTS xtts_v2 기반
- 참조 음성(speaker_wav)로 화자 스타일 반영
- 한국어(ko) 기본, 다국어 코드 지원

### 4) 파일 관리/다운로드
- 업로드 및 합성 결과 파일 저장/다운로드
- 업로드 1시간/결과 24시간 기준 자동 정리(cleanup)

***

## 기술 스택

- Backend: Flask, Flask-CORS
- Audio: PyTorch, Torchaudio, Librosa, SoundFile
- TTS: Coqui TTS (xtts_v2)
- Infra: UUID 기반 안전 파일명, 업로드/출력 디렉토리 분리, 50MB 업로드 제한

***

## 디렉토리 구조

```
project/
├─ app.py                 # Flask 앱 (본 스크립트)
├─ templates/
│   └─ index.html         # 기본 웹 UI
├─ uploads/               # 업로드된 참조 음성 저장
├─ outputs/               # 합성된 결과 음성 저장
├─ requirements.txt
└─ .env                   # (선택) 환경변수
```

***

## 설치 및 실행

### 1) 환경 준비
```bash
# Python 3.10~3.11 권장 (CUDA 사용 시 PyTorch 호환 확인)
python -V

# 가상환경 생성/활성화 (예: venv)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2) 필수 라이브러리 설치
```bash
# PyTorch (CUDA 사용 시 공식 안내에 맞춰 설치)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121  # 예시(CUDA 12.1)
# CPU만 사용할 경우
# pip install torch torchaudio

# 기타 패키지
pip install Flask flask-cors librosa soundfile numpy
pip install TTS==0.22.0  # Coqui TTS
```

권장 requirements.txt 예시:
```txt
Flask==3.0.0
flask-cors==4.0.0
torch==2.3.0
torchaudio==2.3.0
librosa==0.10.1
soundfile==0.12.1
numpy==1.26.4
TTS==0.22.0
```

### 3) 앱 실행
```bash
python app.py
# "http://127.0.0.1:5000 에서 접속할 수 있습니다."
```

브라우저에서 `http://127.0.0.1:5000` 접속

***

## 시스템 요구사항

- 최소
  - Python 3.10+
  - 메모리 4GB
  - 디스크 2GB+
  - 인터넷(모델 다운로드용)
- 권장
  - Python 3.11
  - 메모리 8GB+
  - 디스크 5GB+
  - NVIDIA GPU(CUDA) 사용 시 합성 속도 향상

***

## 사용법

### 1) 모델 초기화
- UI: [모델 초기화] 버튼
- API: `POST /initialize`
- 응답 예시
```json
{ "success": true, "message": "모델이 성공적으로 초기화되었습니다." }
```

### 2) 참조 음성 업로드 및 분석
- UI: 파일 선택 후 업로드
- API: `POST /upload` (form-data: `audio`=파일)
- 성공 응답
```json
{
  "success": true,
  "message": "파일이 성공적으로 업로드되고 분석되었습니다.",
  "filename": "uuid_filename.wav",
  "features": {
    "fundamental_frequency": 128.3,
    "spectral_centroid": 2045.7,
    "spectral_rolloff": 4300.2,
    "mfcc_mean": [...],
    "duration": 8.21,
    "sample_rate": 22050
  }
}
```

### 3) 텍스트 합성(음성 클로닝)
- UI: 텍스트 입력, 참조 파일 선택, 언어(ko 등) 선택 후 합성
- API: `POST /synthesize` (JSON)
```json
{
  "text": "안녕하세요. 데모 음성입니다.",
  "reference_filename": "uuid_filename.wav",
  "language": "ko"
}
```
- 응답
```json
{ "success": true, "output_filename": "synthesized_xxx.wav" }
```

### 4) 결과 다운로드
- API: `GET /download/`

### 5) 임시 파일 정리
- API: `POST /cleanup`
  - uploads: 1시간 경과 파일 삭제
  - outputs: 24시간 경과 파일 삭제

***

## API 요약

- `GET /`  
  웹 UI

- `POST /initialize`  
  TTS 모델 로드

- `POST /upload` (form-data)  
  audio: wav/mp3/flac/m4a

- `POST /synthesize` (JSON)  
  text, reference_filename, language(기본 ko)

- `GET /download/`  
  합성 음성 다운로드

- `POST /cleanup`  
  오래된 파일 삭제

***

## 아키텍처

```
┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐
│   Web UI     │───▶│   Flask API  │───▶│   Coqui TTS (xtts_v2)    │
└──────────────┘    └──────────────┘    └──────────────────────────┘
         │                   │                     ▲
         │                   ├── uploads/ ◀───────┘ (speaker_wav)
         │                   └── outputs/ (합성 결과)
```

***

## 보안/운영 팁

- 화이트리스트 확장자: wav, mp3, flac, m4a
- `secure_filename`으로 경로 조작 방지
- `MAX_CONTENT_LENGTH=50MB` 업로드 제한
- UUID 기반 고유 파일명으로 충돌 방지
- 정기적으로 `/cleanup` 호출하여 스토리지 관리
- GPU 사용 시 Torch/CUDA 버전 호환 필수
- 프로덕션:
  - `DEBUG=False`, gunicorn+nginx 권장
  - 업로드 디렉토리 권한/모니터링
  - 서버 시작 시 `/initialize` 워밍업

***

## 트러블슈팅

| 증상 | 원인 | 해결 |
|-----|------|------|
| 모델 로드 실패 | 네트워크/권한/버전 불일치 | `/initialize` 로그 확인, TTS 버전 고정, 재설치 |
| CUDA 미인식 | 드라이버/버전 불일치 | `nvidia-smi` 확인, Torch CUDA 호환 재설치 |
| 합성 품질 낮음 | 참조 음성 잡음/짧은 길이 | 10~30초 양질 음성 사용, 노이즈/무음 최소화 |
| 400/415 오류 | 잘못된 업로드 포맷 | wav/mp3/flac/m4a 사용, form-data 확인 |
| 파일 누적 | 정리 미수행 | `/cleanup` 주기 호출(크론/스케줄러) |

***

## 커스터마이징 가이드

- 언어 변경: `/synthesize`의 `language` 값을 `en`, `ja` 등으로 변경
- 전처리 강도: `librosa.effects.trim(top_db)`, `preemphasis` 계수 조절
- 최소 길이: `preprocess_audio`의 `min_length` 변경(기본 3초)
- 업로드/출력 경로: `UPLOAD_FOLDER`, `OUTPUT_FOLDER` 변경
- 파일 만료 기준: `cleanup_folder(hours=...)` 조정

***

## 라이선스/법적 고지

- 참조 음성의 권리자 동의를 반드시 확보하세요.
- 사칭, 상업적 오남용을 금지합니다.
- Coqui TTS 및 의존 라이브러리 라이선스를 준수하세요.

***

## 프로젝트 문의

- 이메일: 9radaaa@gmail.com

> “음성 AI로 더 자연스러운 소통을 만듭니다”  
> Building Better Voice Experiences with AI
