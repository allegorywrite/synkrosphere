"""
オーディオ解析モジュール
リアルタイムでオーディオ信号を解析し、ビート検出や周波数解析を行う
"""

import numpy as np
import librosa
import sounddevice as sd
import threading
import time
import logging

logger = logging.getLogger(__name__)

class AudioAnalyzer:
    """
    リアルタイムオーディオ解析クラス
    オーディオストリームを取得し、ビート検出や周波数解析を行う
    """
    
    def __init__(self, sample_rate=44100, buffer_size=2048, hop_length=512):
        """
        Args:
            sample_rate: サンプリングレート (Hz)
            buffer_size: 解析バッファサイズ
            hop_length: ホップ長 (STFTのフレームシフト)
        """
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.hop_length = hop_length
        
        self.audio_buffer = np.zeros(buffer_size, dtype=np.float32)
        
        self.spectrum = np.zeros(buffer_size // 2 + 1, dtype=np.float32)
        self.beat_detected = False
        self.beat_energy = 0.0
        self.rms = 0.0
        self.bpm = 120.0
        
        self.running = False
        self.thread = None
        
        logger.info(f"AudioAnalyzer initialized: sample_rate={sample_rate}, buffer_size={buffer_size}")
    
    def start(self, device_id=None):
        """オーディオ解析を開始"""
        if self.running:
            logger.warning("AudioAnalyzer is already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._audio_thread, args=(device_id,))
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"AudioAnalyzer started with device_id={device_id}")
    
    def stop(self):
        """オーディオ解析を停止"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
        logger.info("AudioAnalyzer stopped")
    
    def _audio_thread(self, device_id):
        """オーディオ処理スレッド"""
        try:
            with sd.InputStream(
                device=device_id,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.hop_length,
                callback=self._audio_callback
            ):
                logger.info("Audio stream started")
                while self.running:
                    time.sleep(0.01)
        except Exception as e:
            logger.error(f"Error in audio thread: {e}")
            self.running = False
    
    def _audio_callback(self, indata, frames, time_info, status):
        """オーディオコールバック関数"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        self.audio_buffer = np.roll(self.audio_buffer, -frames)
        self.audio_buffer[-frames:] = indata.flatten()
        
        self._analyze()
    
    def _analyze(self):
        """オーディオ信号の解析"""
        D = np.abs(librosa.stft(self.audio_buffer, n_fft=self.buffer_size, hop_length=self.buffer_size))
        self.spectrum = D.mean(axis=1)
        
        self.rms = np.sqrt(np.mean(self.audio_buffer**2))
        
        bass_energy = np.sum(self.spectrum[:10])  # 低周波数帯のエネルギー
        
        threshold = 0.5  # 要調整
        if bass_energy > threshold and bass_energy > self.beat_energy * 1.2:
            self.beat_detected = True
        else:
            self.beat_detected = False
        
        self.beat_energy = bass_energy
    
    def get_features(self):
        """解析結果を取得"""
        return {
            'spectrum': self.spectrum.copy(),
            'beat_detected': self.beat_detected,
            'rms': self.rms,
            'bpm': self.bpm
        }
