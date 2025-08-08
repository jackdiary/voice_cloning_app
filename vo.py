from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import os
import uuid
from werkzeug.utils import secure_filename
import torch
import torchaudio
import librosa
import soundfile as sf
import numpy as np
from TTS.api import TTS
import tempfile
import shutil
from pathlib import Path

# 현재 스크립트 디렉토리 기준으로 템플릿 폴더 설정
template_dir = os.path.abspath('./templates')
app = Flask(__name__, template_folder=template_dir)
CORS(app)

# 설정
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a'}

# 폴더 생성
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB 제한

# TTS 모델 초기화 (글로벌)
tts_model = None

class VoiceCloner:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
    def initialize_model(self):
        """TTS 모델 초기화"""
        global tts_model
        try:
            # Coqui TTS 모델 로드 (voice cloning 지원)
            tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
            print("TTS 모델이 성공적으로 로드되었습니다.")
            return True
        except Exception as e:
            print(f"모델 로드 실패: {str(e)}")
            return False
    
    def preprocess_audio(self, audio_path, target_sr=22050):
        """오디오 전처리"""
        try:
            # 오디오 로드
            audio, sr = librosa.load(audio_path, sr=target_sr)
            
            # 노이즈 제거 및 정규화
            audio = librosa.effects.preemphasis(audio)
            audio = librosa.util.normalize(audio)
            
            # 무음 구간 제거
            audio, _ = librosa.effects.trim(audio, top_db=20)
            
            # 최소 길이 확보 (3초 이상)
            min_length = target_sr * 3
            if len(audio) < min_length:
                audio = np.tile(audio, int(np.ceil(min_length / len(audio))))[:min_length]
            
            return audio, target_sr
        except Exception as e:
            raise Exception(f"오디오 전처리 실패: {str(e)}")
    
    def analyze_voice_features(self, audio_path):
        """음성 특징 분석"""
        try:
            audio, sr = self.preprocess_audio(audio_path)
            
            # 기본 음성 특징 추출
            features = {
                'fundamental_frequency': float(np.mean(librosa.yin(audio, fmin=50, fmax=400))),
                'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))),
                'spectral_rolloff': float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))),
                'mfcc_mean': np.mean(librosa.feature.mfcc(y=audio, sr=sr), axis=1).tolist(),
                'duration': len(audio) / sr,
                'sample_rate': sr
            }
            
            return features
        except Exception as e:
            raise Exception(f"음성 특징 분석 실패: {str(e)}")
    
    def clone_voice(self, text, reference_audio_path, output_path, language="ko"):
        """음성 클로닝"""
        try:
            global tts_model
            if tts_model is None:
                raise Exception("TTS 모델이 초기화되지 않았습니다.")
            
            # 참조 오디오 전처리
            processed_audio, sr = self.preprocess_audio(reference_audio_path)
            
            # 임시 파일로 전처리된 오디오 저장
            temp_audio_path = os.path.join(tempfile.gettempdir(), f"temp_{uuid.uuid4()}.wav")
            sf.write(temp_audio_path, processed_audio, sr)
            
            # 음성 합성
            tts_model.tts_to_file(
                text=text,
                speaker_wav=temp_audio_path,
                language=language,
                file_path=output_path
            )
            
            # 임시 파일 정리
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            
            return True
            
        except Exception as e:
            raise Exception(f"음성 클로닝 실패: {str(e)}")

# VoiceCloner 인스턴스 생성
voice_cloner = VoiceCloner()

def allowed_file(filename):
    """허용된 파일 형식 확인"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/initialize', methods=['POST'])
def initialize_model():
    """모델 초기화"""
    try:
        success = voice_cloner.initialize_model()
        if success:
            return jsonify({
                'success': True,
                'message': '모델이 성공적으로 초기화되었습니다.'
            })
        else:
            return jsonify({
                'success': False,
                'message': '모델 초기화에 실패했습니다.'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'오류 발생: {str(e)}'
        }), 500

@app.route('/upload', methods=['POST'])
def upload_audio():
    """오디오 파일 업로드 및 분석"""
    try:
        if 'audio' not in request.files:
            return jsonify({'success': False, 'message': '파일이 선택되지 않았습니다.'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'success': False, 'message': '파일이 선택되지 않았습니다.'}), 400
        
        if file and allowed_file(file.filename):
            # 파일명 보안 처리
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            # 음성 특징 분석
            features = voice_cloner.analyze_voice_features(filepath)
            
            return jsonify({
                'success': True,
                'message': '파일이 성공적으로 업로드되고 분석되었습니다.',
                'filename': unique_filename,
                'features': features
            })
        else:
            return jsonify({
                'success': False,
                'message': '지원되지 않는 파일 형식입니다. (wav, mp3, flac, m4a만 지원)'
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'오류 발생: {str(e)}'
        }), 500

@app.route('/synthesize', methods=['POST'])
def synthesize_speech():
    """텍스트를 음성으로 변환"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        reference_filename = data.get('reference_filename', '').strip()
        language = data.get('language', 'ko')
        
        if not text:
            return jsonify({'success': False, 'message': '텍스트를 입력해주세요.'}), 400
        
        if not reference_filename:
            return jsonify({'success': False, 'message': '참조 음성 파일이 없습니다.'}), 400
        
        # 참조 오디오 파일 경로
        reference_path = os.path.join(app.config['UPLOAD_FOLDER'], reference_filename)
        if not os.path.exists(reference_path):
            return jsonify({'success': False, 'message': '참조 음성 파일을 찾을 수 없습니다.'}), 404
        
        # 출력 파일 경로
        output_filename = f"synthesized_{uuid.uuid4()}.wav"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # 음성 합성
        success = voice_cloner.clone_voice(text, reference_path, output_path, language)
        
        if success and os.path.exists(output_path):
            return jsonify({
                'success': True,
                'message': '음성 합성이 완료되었습니다.',
                'output_filename': output_filename
            })
        else:
            return jsonify({
                'success': False,
                'message': '음성 합성에 실패했습니다.'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'오류 발생: {str(e)}'
        }), 500

@app.route('/download/<filename>')
def download_file(filename):
    """생성된 오디오 파일 다운로드"""
    try:
        filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
        else:
            return jsonify({'success': False, 'message': '파일을 찾을 수 없습니다.'}), 404
    except Exception as e:
        return jsonify({'success': False, 'message': f'오류 발생: {str(e)}'}), 500

@app.route('/cleanup', methods=['POST'])
def cleanup_files():
    """임시 파일 정리"""
    try:
        # 업로드 폴더 정리 (1시간 이상 된 파일)
        cleanup_folder(app.config['UPLOAD_FOLDER'], hours=1)
        # 출력 폴더 정리 (24시간 이상 된 파일)
        cleanup_folder(app.config['OUTPUT_FOLDER'], hours=24)
        
        return jsonify({
            'success': True,
            'message': '파일 정리가 완료되었습니다.'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'오류 발생: {str(e)}'
        }), 500

def cleanup_folder(folder_path, hours=24):
    """지정된 시간보다 오래된 파일들을 삭제"""
    import time
    current_time = time.time()
    cutoff_time = current_time - (hours * 3600)
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            file_time = os.path.getctime(file_path)
            if file_time < cutoff_time:
                try:
                    os.remove(file_path)
                    print(f"삭제된 파일: {filename}")
                except Exception as e:
                    print(f"파일 삭제 실패 {filename}: {str(e)}")

if __name__ == '__main__':
    print("음성 클로닝 웹앱을 시작합니다...")
    print("http://127.0.0.1:5000 에서 접속할 수 있습니다.")
    app.run(debug=True, host='0.0.0.0', port=5000)
