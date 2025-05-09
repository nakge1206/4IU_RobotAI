"""
AudioToTextRecorder 클래스는 제공된 코드에서 빠른 음성-텍스트 변환(Speech-to-Text) 처리를 가능하게 해주는 클래스입니다.

이 클래스는 faster_whisper 라이브러리를 사용하여 녹음된 음성을 기계학습 모델을 통해 텍스트로 변환하며,
GPU 또는 CPU에서 실행될 수 있습니다. 음성 활동 감지(VAD) 기능이 내장되어 있어, 
음성이 감지되면 자동으로 녹음을 시작하거나, 음성이 사라지면 자동으로 녹음을 중지할 수 있습니다. 
또한, pvporcupine 라이브러리를 활용한 웨이크 워드(호출어) 감지 기능도 포함되어 있어, 
특정 단어나 문장이 들리면 녹음을 시작할 수 있습니다. 이 시스템은 실시간 피드백도 제공하며, 사용자 정의도 가능합니다.

Features:
- 음성이 감지되면 자동으로 녹음을 시작하고, 음성이 멈추면 녹음을 중단합니다.

- 설정된 호출어가 감지되면 녹음을 시작합니다.

- 녹음 시작 및 종료 시점에 사용자 정의 동작을 수행할 수 있도록 콜백 함수를 설정할 수 있습니다.

- 녹음된 음성을 가능한 한 빠르게 텍스트로 변환하여 반환합니다.

Author: Kolja Beigel
"""

from faster_whisper import WhisperModel, BatchedInferencePipeline
import faster_whisper

from typing import Iterable, List, Optional, Union
from openwakeword.model import Model
import torch.multiprocessing as mp
from scipy.signal import resample
import signal as system_signal
from ctypes import c_bool
from scipy import signal
import soundfile as sf
import openwakeword
import collections
import numpy as np
import pvporcupine
import traceback
import threading
import webrtcvad
import datetime
import platform
import logging
import struct
import base64
import queue
import torch
import halo
import time
import copy
import os
import re
import gc

# Named logger for this module.
logger = logging.getLogger("realtimestt")
logger.propagate = False

# Set OpenMP runtime duplicate library handling to OK (Use only for development!)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

INIT_REALTIME_PROCESSING_PAUSE = 0.2
INIT_REALTIME_INITIAL_PAUSE = 0.2
INIT_SILERO_SENSITIVITY = 0.4
INIT_WEBRTC_SENSITIVITY = 3
INIT_POST_SPEECH_SILENCE_DURATION = 0.6
INIT_MIN_LENGTH_OF_RECORDING = 0.5
INIT_MIN_GAP_BETWEEN_RECORDINGS = 0
INIT_WAKE_WORDS_SENSITIVITY = 0.6
INIT_PRE_RECORDING_BUFFER_DURATION = 1.0
INIT_WAKE_WORD_ACTIVATION_DELAY = 0.0
INIT_WAKE_WORD_TIMEOUT = 5.0
INIT_WAKE_WORD_BUFFER_DURATION = 0.1
ALLOWED_LATENCY_LIMIT = 100

TIME_SLEEP = 0.02
SAMPLE_RATE = 16000
BUFFER_SIZE = 512
INT16_MAX_ABS_VALUE = 32768.0

INIT_HANDLE_BUFFER_OVERFLOW = False
if platform.system() != 'Darwin':
    INIT_HANDLE_BUFFER_OVERFLOW = True


from funasr import AutoModel

class TranscriptionWorker:
    def __init__(self, conn, stdout_pipe, compute_type, gpu_device_index, device,
                 ready_event, shutdown_event, interrupt_stop_event):
        self.conn = conn
        self.stdout_pipe = stdout_pipe
        self.compute_type = compute_type
        self.gpu_device_index = gpu_device_index
        self.device = device
        self.ready_event = ready_event
        self.shutdown_event = shutdown_event
        self.interrupt_stop_event = interrupt_stop_event
        self.queue = queue.Queue()

    def custom_print(self, *args, **kwargs):
        message = ' '.join(map(str, args))
        try:
            self.stdout_pipe.send(message)
        except (BrokenPipeError, EOFError, OSError):
            pass

    def poll_connection(self):
        while not self.shutdown_event.is_set():
            if self.conn.poll(0.01):
                try:
                    data = self.conn.recv()
                    self.queue.put(data)
                except Exception as e:
                    logging.error(f"Error receiving data from connection: {e}", exc_info=True)
            else:
                time.sleep(0.01)

    def run(self):
        if __name__ == "__main__":
            system_signal.signal(system_signal.SIGINT, system_signal.SIG_IGN)
            __builtins__['print'] = self.custom_print

        logging.info(f"Initializing SenseVoice transcription model...")

        try:
            # SenseVoice 모델 불러오기
            model = AutoModel(
                model="iic/SenseVoiceSmall",
                trust_remote_code=True,
                remote_code="./model.py",
                device=self.device
            )
        except Exception as e:
            logging.exception("Failed to load SenseVoice model")
            raise

        self.ready_event.set()
        polling_thread = threading.Thread(target=self.poll_connection)
        polling_thread.start()

        try:
            while not self.shutdown_event.is_set():
                try:
                    audio, language = self.queue.get(timeout=0.1)
                    try:
                        # SenseVoice 입력 처리
                        result = model.generate(
                            input=audio,  # 파일 경로 or bytes
                            cache={},
                            language=language if language else "auto",
                            use_itn=True,
                            show_tqdm = False
                        )

                        # SenseVoice의 text 내 태그들을 구분하기 위한 함수.
                        def parse_sensevoice_tags(text):
                            tag_pattern = re.compile(r"<\|([^|]+)\|>")
                            tags = tag_pattern.findall(text)
                            parsed = {
                                "language": None,
                                "emotion": None,
                                "event": None,
                                "itn": None,
                                "transcription": None
                            }
                            # 순서대로 할당
                            if len(tags) >= 4:
                                parsed["language"] = tags[0]
                                parsed["emotion"] = tags[1]
                                parsed["event"] = tags[2]
                                parsed["itn"] = tags[3]
                            parsed["transcription"] = tag_pattern.sub('', text).strip()
                            return parsed
                        #디버그용 출력
                        print(result)

                        #text 및 태그 전송
                        raw_text = result[0]["text"]
                        parsed_result = parse_sensevoice_tags(raw_text)
                        text = parsed_result["transcription"]
                        info = {
                                "language": parsed_result["language"],
                                "emotion": parsed_result["emotion"],
                                "event": parsed_result["event"],
                                "itn": parsed_result["itn"]
                            }
                        # 텍스트와 info 딕셔너리를 함께 전송
                        self.conn.send(('success', (text, info)))
                    except Exception as e:
                        logging.error(f"Transcription error: {e}", exc_info=True)
                        self.conn.send(('error', str(e)))
                except queue.Empty:
                    continue
        finally:
            __builtins__['print'] = print
            self.conn.close()
            self.stdout_pipe.close()
            self.shutdown_event.set()
            polling_thread.join()


class bcolors:
    OKGREEN = '\033[92m'  # Green for active speech detection
    WARNING = '\033[93m'  # Yellow for silence detection
    ENDC = '\033[0m'      # Reset to default color


class AudioToTextRecorder:
    """
    A class responsible for capturing audio from the microphone, detecting
    voice activity, and then transcribing the captured audio using the
    `faster_whisper` model.
    """

    def __init__(self,
                 language: str = "",
                 compute_type: str = "default",
                 input_device_index: int = None,
                 gpu_device_index: Union[int, List[int]] = 0,
                 device: str = "cuda",
                 on_recording_start=None,
                 on_recording_stop=None,
                 on_transcription_start=None,
                 ensure_sentence_starting_uppercase=True,
                 ensure_sentence_ends_with_period=True,
                 use_microphone=True,
                 spinner=True,
                 level=logging.WARNING,

                 # Voice activation parameters
                 silero_sensitivity: float = INIT_SILERO_SENSITIVITY,
                 silero_use_onnx: bool = False,
                 silero_deactivity_detection: bool = False,
                 webrtc_sensitivity: int = INIT_WEBRTC_SENSITIVITY,
                 post_speech_silence_duration: float = (
                     INIT_POST_SPEECH_SILENCE_DURATION
                 ),
                 min_length_of_recording: float = (
                     INIT_MIN_LENGTH_OF_RECORDING
                 ),
                 min_gap_between_recordings: float = (
                     INIT_MIN_GAP_BETWEEN_RECORDINGS
                 ),
                 pre_recording_buffer_duration: float = (
                     INIT_PRE_RECORDING_BUFFER_DURATION
                 ),
                 on_vad_start=None,
                 on_vad_stop=None,
                 on_vad_detect_start=None,
                 on_vad_detect_stop=None,

                 # Wake word parameters
                 wakeword_backend: str = "",
                 openwakeword_model_paths: str = None,
                 openwakeword_inference_framework: str = "onnx",
                 wake_words: str = "",
                 wake_words_sensitivity: float = INIT_WAKE_WORDS_SENSITIVITY,
                 wake_word_activation_delay: float = (
                    INIT_WAKE_WORD_ACTIVATION_DELAY
                 ),
                 wake_word_timeout: float = INIT_WAKE_WORD_TIMEOUT,
                 wake_word_buffer_duration: float = INIT_WAKE_WORD_BUFFER_DURATION,
                 on_wakeword_detected=None,
                 on_wakeword_timeout=None,
                 on_wakeword_detection_start=None,
                 on_wakeword_detection_end=None,
                 on_recorded_chunk=None,
                 debug_mode=False,
                 handle_buffer_overflow: bool = INIT_HANDLE_BUFFER_OVERFLOW,
                 buffer_size: int = BUFFER_SIZE,
                 sample_rate: int = SAMPLE_RATE,
                 print_transcription_time: bool = False,
                 early_transcription_on_silence: int = 0,
                 allowed_latency_limit: int = ALLOWED_LATENCY_LIMIT,
                 no_log_file: bool = False,
                 use_extended_logging: bool = False,
                 ):
        """
        Initializes an audio recorder and  transcription
        and wake word detection.

        Args:
        - model (str, default="tiny"): Specifies the size of the transcription
            model to use or the path to a converted model directory.
            Valid options are 'tiny', 'tiny.en', 'base', 'base.en',
            'small', 'small.en', 'medium', 'medium.en', 'large-v1',
            'large-v2'.
            If a specific size is provided, the model is downloaded
            from the Hugging Face Hub.
        - language (str, default=""): Language code for speech-to-text engine.
            If not specified, the model will attempt to detect the language
            automatically.
        - compute_type (str, default="default"): Specifies the type of
            computation to be used for transcription.
            See https://opennmt.net/CTranslate2/quantization.html.
        - input_device_index (int, default=0): The index of the audio input
            device to use.
        - gpu_device_index (int, default=0): Device ID to use.
            The model can also be loaded on multiple GPUs by passing a list of
            IDs (e.g. [0, 1, 2, 3]). In that case, multiple transcriptions can
            run in parallel when transcribe() is called from multiple Python
            threads
        - device (str, default="cuda"): Device for model to use. Can either be 
            "cuda" or "cpu".
        - on_recording_start (callable, default=None): Callback function to be
            called when recording of audio to be transcripted starts.
        - on_recording_stop (callable, default=None): Callback function to be
            called when recording of audio to be transcripted stops.
        - on_transcription_start (callable, default=None): Callback function
            to be called when transcription of audio to text starts.
        - ensure_sentence_starting_uppercase (bool, default=True): Ensures
            that every sentence detected by the algorithm starts with an
            uppercase letter.
        - ensure_sentence_ends_with_period (bool, default=True): Ensures that
            every sentence that doesn't end with punctuation such as "?", "!"
            ends with a period
        - use_microphone (bool, default=True): Specifies whether to use the
            microphone as the audio input source. If set to False, the
            audio input source will be the audio data sent through the
            feed_audio() method.
        - spinner (bool, default=True): Show spinner animation with current
            state.
        - level (int, default=logging.WARNING): Logging level.
        - enable_realtime_transcription (bool, default=False): Enables or
            disables real-time transcription of audio. When set to True, the
            audio will be transcribed continuously as it is being recorded.
        - use_main_model_for_realtime (str, default=False):
            If True, use the main transcription model for both regular and
            real-time transcription. If False, use a separate model specified
            by realtime_model_type for real-time transcription.
            Using a single model can save memory and potentially improve
            performance, but may not be optimized for real-time processing.
            Using separate models allows for a smaller, faster model for
            real-time transcription while keeping a more accurate model for
            final transcription.
        - realtime_model_type (str, default="tiny"): Specifies the machine
            learning model to be used for real-time transcription. Valid
            options include 'tiny', 'tiny.en', 'base', 'base.en', 'small',
            'small.en', 'medium', 'medium.en', 'large-v1', 'large-v2'.
        - realtime_processing_pause (float, default=0.1): Specifies the time
            interval in seconds after a chunk of audio gets transcribed. Lower
            values will result in more "real-time" (frequent) transcription
            updates but may increase computational load.
        - init_realtime_after_seconds (float, default=0.2): Specifies the 
            initial waiting time after the recording was initiated before
            yielding the first realtime transcription
        - on_realtime_transcription_update = A callback function that is
            triggered whenever there's an update in the real-time
            transcription. The function is called with the newly transcribed
            text as its argument.
        - on_realtime_transcription_stabilized = A callback function that is
            triggered when the transcribed text stabilizes in quality. The
            stabilized text is generally more accurate but may arrive with a
            slight delay compared to the regular real-time updates.
        - silero_sensitivity (float, default=SILERO_SENSITIVITY): Sensitivity
            for the Silero Voice Activity Detection model ranging from 0
            (least sensitive) to 1 (most sensitive). Default is 0.5.
        - silero_use_onnx (bool, default=False): Enables usage of the
            pre-trained model from Silero in the ONNX (Open Neural Network
            Exchange) format instead of the PyTorch format. This is
            recommended for faster performance.
        - silero_deactivity_detection (bool, default=False): Enables the Silero
            model for end-of-speech detection. More robust against background
            noise. Utilizes additional GPU resources but improves accuracy in
            noisy environments. When False, uses the default WebRTC VAD,
            which is more sensitive but may continue recording longer due
            to background sounds.
        - webrtc_sensitivity (int, default=WEBRTC_SENSITIVITY): Sensitivity
            for the WebRTC Voice Activity Detection engine ranging from 0
            (least aggressive / most sensitive) to 3 (most aggressive,
            least sensitive). Default is 3.
        - post_speech_silence_duration (float, default=0.2): Duration in
            seconds of silence that must follow speech before the recording
            is considered to be completed. This ensures that any brief
            pauses during speech don't prematurely end the recording.
        - min_gap_between_recordings (float, default=1.0): Specifies the
            minimum time interval in seconds that should exist between the
            end of one recording session and the beginning of another to
            prevent rapid consecutive recordings.
        - min_length_of_recording (float, default=1.0): Specifies the minimum
            duration in seconds that a recording session should last to ensure
            meaningful audio capture, preventing excessively short or
            fragmented recordings.
        - pre_recording_buffer_duration (float, default=0.2): Duration in
            seconds for the audio buffer to maintain pre-roll audio
            (compensates speech activity detection latency)
        - on_vad_start (callable, default=None): Callback function to be called
            when the system detected the start of voice activity presence.
        - on_vad_stop (callable, default=None): Callback function to be called
            when the system detected the stop (end) of voice activity presence.
        - on_vad_detect_start (callable, default=None): Callback function to
            be called when the system listens for voice activity. This is not
            called when VAD actually happens (use on_vad_start for this), but
            when the system starts listening for it.
        - on_vad_detect_stop (callable, default=None): Callback function to be
            called when the system stops listening for voice activity. This is
            not called when VAD actually stops (use on_vad_stop for this), but
            when the system stops listening for it.
        - wakeword_backend (str, default="pvporcupine"): Specifies the backend
            library to use for wake word detection. Supported options include
            'pvporcupine' for using the Porcupine wake word engine or 'oww' for
            using the OpenWakeWord engine.
        - openwakeword_model_paths (str, default=None): Comma-separated paths
            to model files for the openwakeword library. These paths point to
            custom models that can be used for wake word detection when the
            openwakeword library is selected as the wakeword_backend.
        - openwakeword_inference_framework (str, default="onnx"): Specifies
            the inference framework to use with the openwakeword library.
            Can be either 'onnx' for Open Neural Network Exchange format 
            or 'tflite' for TensorFlow Lite.
        - wake_words (str, default=""): Comma-separated string of wake words to
            initiate recording when using the 'pvporcupine' wakeword backend.
            Supported wake words include: 'alexa', 'americano', 'blueberry',
            'bumblebee', 'computer', 'grapefruits', 'grasshopper', 'hey google',
            'hey siri', 'jarvis', 'ok google', 'picovoice', 'porcupine',
            'terminator'. For the 'openwakeword' backend, wake words are
            automatically extracted from the provided model files, so specifying
            them here is not necessary.
        - wake_words_sensitivity (float, default=0.5): Sensitivity for wake
            word detection, ranging from 0 (least sensitive) to 1 (most
            sensitive). Default is 0.5.
        - wake_word_activation_delay (float, default=0): Duration in seconds
            after the start of monitoring before the system switches to wake
            word activation if no voice is initially detected. If set to
            zero, the system uses wake word activation immediately.
        - wake_word_timeout (float, default=5): Duration in seconds after a
            wake word is recognized. If no subsequent voice activity is
            detected within this window, the system transitions back to an
            inactive state, awaiting the next wake word or voice activation.
        - wake_word_buffer_duration (float, default=0.1): Duration in seconds
            to buffer audio data during wake word detection. This helps in
            cutting out the wake word from the recording buffer so it does not
            falsely get detected along with the following spoken text, ensuring
            cleaner and more accurate transcription start triggers.
            Increase this if parts of the wake word get detected as text.
        - on_wakeword_detected (callable, default=None): Callback function to
            be called when a wake word is detected.
        - on_wakeword_timeout (callable, default=None): Callback function to
            be called when the system goes back to an inactive state after when
            no speech was detected after wake word activation
        - on_wakeword_detection_start (callable, default=None): Callback
             function to be called when the system starts to listen for wake
             words
        - on_wakeword_detection_end (callable, default=None): Callback
            function to be called when the system stops to listen for
            wake words (e.g. because of timeout or wake word detected)
        - on_recorded_chunk (callable, default=None): Callback function to be
            called when a chunk of audio is recorded. The function is called
            with the recorded audio chunk as its argument.
        - debug_mode (bool, default=False): If set to True, the system will
            print additional debug information to the console.
        - handle_buffer_overflow (bool, default=True): If set to True, the system
            will log a warning when an input overflow occurs during recording and
            remove the data from the buffer.
        - buffer_size (int, default=512): The buffer size to use for audio
            recording. Changing this may break functionality.
        - sample_rate (int, default=16000): The sample rate to use for audio
            recording. Changing this will very probably functionality (as the
            WebRTC VAD model is very sensitive towards the sample rate).
        - print_transcription_time (bool, default=False): Logs processing time
            of main model transcription 
        - early_transcription_on_silence (int, default=0): If set, the
            system will transcribe audio faster when silence is detected.
            Transcription will start after the specified milliseconds, so 
            keep this value lower than post_speech_silence_duration. 
            Ideally around post_speech_silence_duration minus the estimated
            transcription time with the main model.
            If silence lasts longer than post_speech_silence_duration, the 
            recording is stopped, and the transcription is submitted. If 
            voice activity resumes within this period, the transcription 
            is discarded. Results in faster final transcriptions to the cost
            of additional GPU load due to some unnecessary final transcriptions.
        - allowed_latency_limit (int, default=100): Maximal amount of chunks
            that can be unprocessed in queue before discarding chunks.
        - no_log_file (bool, default=False): Skips writing of debug log file.
        - use_extended_logging (bool, default=False): Writes extensive
            log messages for the recording worker, that processes the audio
            chunks.

        Raises:
            Exception: Errors related to initializing transcription
            model, wake word detection, or audio recording.
        """

        self.language = language
        self.compute_type = compute_type
        self.input_device_index = input_device_index
        self.gpu_device_index = gpu_device_index
        self.device = device
        self.wake_words = wake_words
        self.wake_word_activation_delay = wake_word_activation_delay
        self.wake_word_timeout = wake_word_timeout
        self.wake_word_buffer_duration = wake_word_buffer_duration
        self.ensure_sentence_starting_uppercase = (
            ensure_sentence_starting_uppercase
        )
        self.ensure_sentence_ends_with_period = (
            ensure_sentence_ends_with_period
        )
        self.use_microphone = mp.Value(c_bool, use_microphone)
        self.min_gap_between_recordings = min_gap_between_recordings
        self.min_length_of_recording = min_length_of_recording
        self.pre_recording_buffer_duration = pre_recording_buffer_duration
        self.post_speech_silence_duration = post_speech_silence_duration
        self.on_recording_start = on_recording_start
        self.on_recording_stop = on_recording_stop
        self.on_wakeword_detected = on_wakeword_detected
        self.on_wakeword_timeout = on_wakeword_timeout
        self.on_vad_start = on_vad_start
        self.on_vad_stop = on_vad_stop
        self.on_vad_detect_start = on_vad_detect_start
        self.on_vad_detect_stop = on_vad_detect_stop
        self.on_wakeword_detection_start = on_wakeword_detection_start
        self.on_wakeword_detection_end = on_wakeword_detection_end
        self.on_recorded_chunk = on_recorded_chunk
        self.on_transcription_start = on_transcription_start
        self.debug_mode = debug_mode
        self.handle_buffer_overflow = handle_buffer_overflow
        self.allowed_latency_limit = allowed_latency_limit

        self.level = level
        self.audio_queue = mp.Queue()
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.recording_start_time = 0
        self.recording_stop_time = 0
        self.last_recording_start_time = 0
        self.last_recording_stop_time = 0
        self.wake_word_detect_time = 0
        self.silero_check_time = 0
        self.silero_working = False
        self.speech_end_silence_start = 0
        self.silero_sensitivity = silero_sensitivity
        self.silero_deactivity_detection = silero_deactivity_detection
        self.listen_start = 0
        self.spinner = spinner
        self.halo = None
        self.state = "inactive"
        self.wakeword_detected = False
        self.text_storage = []
        self.realtime_stabilized_text = ""
        self.realtime_stabilized_safetext = ""
        self.is_webrtc_speech_active = False
        self.is_silero_speech_active = False
        self.recording_thread = None
        self.realtime_thread = None
        self.audio_interface = None
        self.audio = None
        self.stream = None
        self.start_recording_event = threading.Event()
        self.stop_recording_event = threading.Event()
        self.backdate_stop_seconds = 0.0
        self.backdate_resume_seconds = 0.0
        self.last_transcription_bytes = None
        self.last_transcription_bytes_b64 = None
        self.use_wake_words = wake_words or wakeword_backend in {'oww', 'openwakeword', 'openwakewords'}
        self.detected_realtime_language = None
        self.detected_realtime_language_probability = 0
        self.transcription_lock = threading.Lock()
        self.shutdown_lock = threading.Lock()
        self.transcribe_count = 0
        self.print_transcription_time = print_transcription_time
        self.early_transcription_on_silence = early_transcription_on_silence
        self.use_extended_logging = use_extended_logging
        self.awaiting_speech_end = False

        # ----------------------------------------------------------------------------
        # Named logger configuration
        # By default, let's set it up so it logs at 'level' to the console.
        # If you do NOT want this default configuration, remove the lines below
        # and manage your "realtimestt" logger from your application code.
        #지정된 이름의 로거 설정
        #기본적으로, 아래 설정은 콘솔에 'level' 수준으로 로그를 출력하게 합니다.
        #만약 이 기본 설정을 사용하고 싶지 않다면, 아래 줄들을 제거하고
        #애플리케이션 코드에서 "realtimestt" 로거를 직접 관리하세요.
        logger.setLevel(logging.DEBUG)  # We capture all, then filter via handlers
        # 모든 로그를 수집하고, 핸들러에서 필터링합니다.

        log_format = "RealTimeSTT: %(name)s - %(levelname)s - %(message)s"
        file_log_format = "%(asctime)s.%(msecs)03d - " + log_format

        # Create and set up console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.level)
        console_handler.setFormatter(logging.Formatter(log_format))

        logger.addHandler(console_handler)

        if not no_log_file:
            file_handler = logging.FileHandler('realtimesst.log')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(file_log_format, datefmt='%Y-%m-%d %H:%M:%S'))
            logger.addHandler(file_handler)
        # ----------------------------------------------------------------------------

        self.is_shut_down = False
        self.shutdown_event = mp.Event()
        
        try:
            # Only set the start method if it hasn't been set already
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method("spawn")
        except RuntimeError as e:
            logger.info(f"Start method has already been set. Details: {e}")

        logger.info("Starting RealTimeSTT")

        if use_extended_logging:
            logger.info("RealtimeSTT was called with these parameters:")
            for param, value in locals().items():
                logger.info(f"{param}: {value}")

        self.interrupt_stop_event = mp.Event()
        self.was_interrupted = mp.Event()
        self.main_transcription_ready_event = mp.Event()
        self.parent_transcription_pipe, child_transcription_pipe = mp.Pipe()
        self.parent_stdout_pipe, child_stdout_pipe = mp.Pipe()

        #모델 실행에 사용할 장치를 설정합니다.
        self.device = "cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu"

        self.transcript_process = self._start_thread(
            target=AudioToTextRecorder._transcription_worker,
                args=(
                    child_transcription_pipe,        # 부모와 통신하는 Pipe. 부모에서 .send() → 자식에서 .recv()
                    child_stdout_pipe,               # 출력 내용을 부모에 보내는 용도의 Pipe
                    self.compute_type,               # 예: "int8", "float16", "float32"
                    self.gpu_device_index,           # 사용할 GPU의 인덱스 (예: 0번 GPU)
                    self.device,                     # 최종 장치 설정 ("cpu" 또는 "cuda")
                    self.main_transcription_ready_event,  # 모델 준비 완료를 알리는 Event (동기화 용도)
                    self.shutdown_event,             # 종료 플래그 (프로세스 중지 신호)
                    self.interrupt_stop_event,       # 수동 인터럽트 플래그 (Ctrl+C 같은)
                )
        )

        # Start audio data reading process
        if self.use_microphone.value:
            logger.info("Initializing audio recording"
                         " (creating pyAudio input stream,"
                         f" sample rate: {self.sample_rate}"
                         f" buffer size: {self.buffer_size}"
                         )
            self.reader_process = self._start_thread(
                target=AudioToTextRecorder._audio_data_worker,
                args=(
                    self.audio_queue,
                    self.sample_rate,
                    self.buffer_size,
                    self.input_device_index,
                    self.shutdown_event,
                    self.interrupt_stop_event,
                    self.use_microphone
                )
            )

        # Setup wake word detection
        if wake_words or wakeword_backend in {'oww', 'openwakeword', 'openwakewords', 'pvp', 'pvporcupine'}:
            self.wakeword_backend = wakeword_backend

            self.wake_words_list = [
                word.strip() for word in wake_words.lower().split(',')
            ]
            self.wake_words_sensitivity = wake_words_sensitivity
            self.wake_words_sensitivities = [
                float(wake_words_sensitivity)
                for _ in range(len(self.wake_words_list))
            ]

            if self.wakeword_backend in {'pvp', 'pvporcupine'}:

                try:
                    self.porcupine = pvporcupine.create(
                        keywords=self.wake_words_list,
                        sensitivities=self.wake_words_sensitivities
                    )
                    self.buffer_size = self.porcupine.frame_length
                    self.sample_rate = self.porcupine.sample_rate

                except Exception as e:
                    logger.exception(
                        "Error initializing porcupine "
                        f"wake word detection engine: {e}"
                    )
                    raise

                logger.debug(
                    "Porcupine wake word detection engine initialized successfully"
                )

            elif self.wakeword_backend in {'oww', 'openwakeword', 'openwakewords'}:
                    
                openwakeword.utils.download_models()

                try:
                    if openwakeword_model_paths:
                        model_paths = openwakeword_model_paths.split(',')
                        self.owwModel = Model(
                            wakeword_models=model_paths,
                            inference_framework=openwakeword_inference_framework
                        )
                        logger.info(
                            "Successfully loaded wakeword model(s): "
                            f"{openwakeword_model_paths}"
                        )
                    else:
                        self.owwModel = Model(
                            inference_framework=openwakeword_inference_framework)
                    
                    self.oww_n_models = len(self.owwModel.models.keys())
                    if not self.oww_n_models:
                        logger.error(
                            "No wake word models loaded."
                        )

                    for model_key in self.owwModel.models.keys():
                        logger.info(
                            "Successfully loaded openwakeword model: "
                            f"{model_key}"
                        )

                except Exception as e:
                    logger.exception(
                        "Error initializing openwakeword "
                        f"wake word detection engine: {e}"
                    )
                    raise

                logger.debug(
                    "Open wake word detection engine initialized successfully"
                )
            
            else:
                logger.exception(f"Wakeword engine {self.wakeword_backend} unknown/unsupported. Please specify one of: pvporcupine, openwakeword.")


        # Setup voice activity detection model WebRTC
        try:
            logger.info("Initializing WebRTC voice with "
                         f"Sensitivity {webrtc_sensitivity}"
                         )
            self.webrtc_vad_model = webrtcvad.Vad()
            self.webrtc_vad_model.set_mode(webrtc_sensitivity)

        except Exception as e:
            logger.exception("Error initializing WebRTC voice "
                              f"activity detection engine: {e}"
                              )
            raise

        logger.debug("WebRTC VAD voice activity detection "
                      "engine initialized successfully"
                      )

        # Setup voice activity detection model Silero VAD
        try:
            self.silero_vad_model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                verbose=False,
                onnx=silero_use_onnx
            )

        except Exception as e:
            logger.exception(f"Error initializing Silero VAD "
                              f"voice activity detection engine: {e}"
                              )
            raise

        logger.debug("Silero VAD voice activity detection "
                      "engine initialized successfully"
                      )

        self.audio_buffer = collections.deque(
            maxlen=int((self.sample_rate // self.buffer_size) *
                       self.pre_recording_buffer_duration)
        )
        self.last_words_buffer = collections.deque(
            maxlen=int((self.sample_rate // self.buffer_size) *
                       0.3)
        )
        self.frames = []
        self.last_frames = []

        # Recording control flags
        self.is_recording = False
        self.is_running = True
        self.start_recording_on_voice_activity = False
        self.stop_recording_on_voice_deactivity = False

        # Start the recording worker thread
        self.recording_thread = threading.Thread(target=self._recording_worker)
        self.recording_thread.daemon = True
        self.recording_thread.start()
                   
        # Wait for transcription models to start
        logger.debug('Waiting for main transcription model to start')
        self.main_transcription_ready_event.wait()
        logger.debug('Main transcription model ready')

        self.stdout_thread = threading.Thread(target=self._read_stdout)
        self.stdout_thread.daemon = True
        self.stdout_thread.start()

        logger.debug('RealtimeSTT initialization completed successfully')
                   
    def _start_thread(self, target=None, args=()):
        """
        라이브러리 전반에서 일관된 쓰레딩 모델을 구현합니다.

        이 메서드는 라이브러리 내에서 어떤 쓰레드를 시작할 때도 사용됩니다.
        Linux에서는 표준 threading.Thread를 사용하고,
        그 외의 운영체제에서는 PyTorch의 멀티프로세싱 라이브러리인 multiprocessing.Process를 사용합니다.
        Args:
            실행될 대상 함수나 메서드를 나타내는 호출 가능한 객체입니다.
              기본값은 None이며, 이 경우 실행할 함수가 없습니다.

            target 호출 시 전달할 인자들을 담은 리스트 또는 튜플입니다.
              기본값은 빈 튜플 ()입니다.
        """
        if (platform.system() == 'Linux'):
            thread = threading.Thread(target=target, args=args)
            thread.deamon = True
            thread.start()
            return thread
        else:
            thread = mp.Process(target=target, args=args)
            thread.start()
            return thread

    def _read_stdout(self):
        while not self.shutdown_event.is_set():
            try:
                if self.parent_stdout_pipe.poll(0.1):
                    logger.debug("Receive from stdout pipe")
                    message = self.parent_stdout_pipe.recv()
                    logger.info(message)
            except (BrokenPipeError, EOFError, OSError):
                # The pipe probably has been closed, so we ignore the error
                pass
            except KeyboardInterrupt:  # handle manual interruption (Ctrl+C)
                logger.info("KeyboardInterrupt in read from stdout detected, exiting...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in read from stdout: {e}", exc_info=True)
                logger.error(traceback.format_exc())  # Log the full traceback here
                break 
            time.sleep(0.1)

    def _transcription_worker(*args, **kwargs):
        worker = TranscriptionWorker(*args, **kwargs)
        worker.run()

    @staticmethod
    def _audio_data_worker(
        audio_queue,
        target_sample_rate,
        buffer_size,
        input_device_index,
        shutdown_event,
        interrupt_stop_event,
        use_microphone
    ):
        """
        오디오 녹음 처리를 담당하는 워커 메서드입니다.

        이 메서드는 별도의 프로세스에서 실행되며 다음 작업들을 수행합니다:
        - 오디오 입력 스트림을 설정하여 가능한 가장 높은 샘플링 속도로 녹음을 시작합니다.

        - 입력 스트림에서 오디오 데이터를 지속적으로 읽고, 필요 시 리샘플링(샘플 속도 변환)하며,
        전처리를 거친 후 완전한 오디오 청크를 큐에 넣습니다.

        - 녹음 중 발생하는 오류를 처리합니다.

        - 종료 이벤트가 설정되면, 녹음 프로세스를 정상적으로 종료합니다.

        Args:
            audio_queue (queue.Queue): A queue where recorded audio data is placed.
                                        녹음된 오디오 데이터를 저장할 큐 객체입니다.
            target_sample_rate (int): The desired sample rate for the output audio (for Silero VAD).
                                        출력 오디오의 목표 샘플링 속도입니다. (Silero VAD와 같은 모델이 요구하는 속도)
            buffer_size (int): The number of samples expected by the Silero VAD model.
                                Silero VAD 모델이 요구하는 버퍼 크기(샘플 수) 입니다.
            input_device_index (int): The index of the audio input device.
                                        사용할 오디오 입력 장치의 인덱스 번호입니다. (마이크 장치 선택용)
            shutdown_event (threading.Event): An event that, when set, signals this worker method to terminate.
                                                이 이벤트가 설정되면, 녹음 루프를 종료하라는 신호입니다.
            interrupt_stop_event (threading.Event): An event to signal keyboard interrupt.
                                                    키보드 인터럽트 (Ctrl+C) 를 처리하기 위한 이벤트입니다.
            use_microphone (multiprocessing.Value): A shared value indicating whether to use the microphone.
                                                    마이크를 사용할지 여부를 나타내는 공유 값입니다. (다중 프로세스 간 상태 공유에 사용)

        Raises:
            Exception: If there is an error while initializing the audio recording.
                        오디오 녹음 초기화 중 문제가 생기면 예외가 발생합니다.
        """
        import pyaudio
        import numpy as np
        from scipy import signal

        if __name__ == '__main__':
            system_signal.signal(system_signal.SIGINT, system_signal.SIG_IGN)

        def get_highest_sample_rate(audio_interface, device_index):
            """마이크(오디오 입력 장치)의 최대 샘플링 속도를 찾아주는 역할"""
            try:
                device_info = audio_interface.get_device_info_by_index(device_index)
                logger.debug(f"Retrieving highest sample rate for device index {device_index}: {device_info}")
                max_rate = int(device_info['defaultSampleRate'])

                if 'supportedSampleRates' in device_info:
                    supported_rates = [int(rate) for rate in device_info['supportedSampleRates']]
                    if supported_rates:
                        max_rate = max(supported_rates)

                logger.debug(f"Highest supported sample rate for device index {device_index} is {max_rate}")
                return max_rate
            except Exception as e:
                logger.warning(f"Failed to get highest sample rate: {e}")
                return 48000  # Fallback to a common high sample rate

        def initialize_audio_stream(audio_interface, sample_rate, chunk_size):
            nonlocal input_device_index

            def validate_device(device_index):
                """Validate that the device exists and is actually available for input."""
                try:
                    device_info = audio_interface.get_device_info_by_index(device_index)
                    logger.debug(f"Validating device index {device_index} with info: {device_info}")
                    if not device_info.get('maxInputChannels', 0) > 0:
                        logger.debug("Device has no input channels, invalid for recording.")
                        return False

                    # Try to actually read from the device
                    test_stream = audio_interface.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=target_sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size,
                        input_device_index=device_index,
                        start=False  # Don't start the stream yet
                    )

                    test_stream.start_stream()
                    test_data = test_stream.read(chunk_size, exception_on_overflow=False)
                    test_stream.stop_stream()
                    test_stream.close()

                    if len(test_data) == 0:
                        logger.debug("Device produced no data, invalid for recording.")
                        return False

                    logger.debug(f"Device index {device_index} successfully validated.")
                    return True

                except Exception as e:
                    logger.debug(f"Device validation failed for index {device_index}: {e}")
                    return False

            """Initialize the audio stream with error handling."""
            while not shutdown_event.is_set():
                try:
                    # First, get a list of all available input devices
                    input_devices = []
                    device_count = audio_interface.get_device_count()
                    logger.debug(f"Found {device_count} total audio devices on the system.")
                    for i in range(device_count):
                        try:
                            device_info = audio_interface.get_device_info_by_index(i)
                            if device_info.get('maxInputChannels', 0) > 0:
                                input_devices.append(i)
                        except Exception as e:
                            logger.debug(f"Could not retrieve info for device index {i}: {e}")
                            continue

                    logger.debug(f"Available input devices with input channels: {input_devices}")
                    if not input_devices:
                        raise Exception("No input devices found")

                    # If input_device_index is None or invalid, try to find a working device
                    if input_device_index is None or input_device_index not in input_devices:
                        # First try the default device
                        try:
                            default_device = audio_interface.get_default_input_device_info()
                            logger.debug(f"Default device info: {default_device}")
                            if validate_device(default_device['index']):
                                input_device_index = default_device['index']
                                logger.debug(f"Default device {input_device_index} selected.")
                        except Exception:
                            # If default device fails, try other available input devices
                            logger.debug("Default device validation failed, checking other devices...")
                            for device_index in input_devices:
                                if validate_device(device_index):
                                    input_device_index = device_index
                                    logger.debug(f"Device {input_device_index} selected.")
                                    break
                            else:
                                raise Exception("No working input devices found")

                    # Validate the selected device one final time
                    if not validate_device(input_device_index):
                        raise Exception("Selected device validation failed")

                    # If we get here, we have a validated device
                    logger.debug(f"Opening stream with device index {input_device_index}, "
                                f"sample_rate={sample_rate}, chunk_size={chunk_size}")
                    stream = audio_interface.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size,
                        input_device_index=input_device_index,
                    )

                    logger.info(f"Microphone connected and validated (device index: {input_device_index}, "
                                f"sample rate: {sample_rate}, chunk size: {chunk_size})")
                    return stream

                except Exception as e:
                    logger.error(f"Microphone connection failed: {e}. Retrying...", exc_info=True)
                    input_device_index = None
                    time.sleep(3)  # Wait before retrying
                    continue

        def preprocess_audio(chunk, original_sample_rate, target_sample_rate):
            """feed_audio 메서드처럼 오디오 청크를 전처리하는 함수입니다."""
            #입력된 청크가 numpy배열이라면,
            if isinstance(chunk, np.ndarray):
                # Handle stereo to mono conversion if necessary
                #필요하다면 스테레오->모노 변경을 처리함. (2차원배열[스테레오] 라면, 평균으 구해서 모노로 만듬.)
                if chunk.ndim == 2:
                    chunk = np.mean(chunk, axis=1)

                # Resample to target_sample_rate if necessary
                #원래샘플레이트와 목표샘플레이트와 다르면 리샘플링을 수행.
                if original_sample_rate != target_sample_rate:
                    logger.debug(f"Resampling from {original_sample_rate} Hz to {target_sample_rate} Hz.")
                    num_samples = int(len(chunk) * target_sample_rate / original_sample_rate)
                    chunk = signal.resample(chunk, num_samples)

                #최종적으로 int16 데이터 타입으로 변환해 저장. (PCM 16비트)
                chunk = chunk.astype(np.int16)
            else:
                # If chunk is bytes, convert to numpy array
                #만약 청크가 바이트 형식이면, numpy배열로 변환.
                chunk = np.frombuffer(chunk, dtype=np.int16)

                # Resample if necessary
                #원래샘플레이트와 목표샘플레이트와 다르면 리샘플링을 수행.
                if original_sample_rate != target_sample_rate:
                    logger.debug(f"Resampling from {original_sample_rate} Hz to {target_sample_rate} Hz.")
                    num_samples = int(len(chunk) * target_sample_rate / original_sample_rate)
                    chunk = signal.resample(chunk, num_samples)
                    chunk = chunk.astype(np.int16)

            #청크를 바이트형식으로 바꾸어 반환함.
            return chunk.tobytes()

        audio_interface = None #PyAudio 객체. 오디오 장치와 상호작용하기 위한 인터페이스.
        stream = None #녹음 스트림 객체. 마이크에서 데이터를 받아오는 실질적인 통로.
        device_sample_rate = None #현재 오디오 장치가 사용하는 샘플링 속도.
        chunk_size = 1024  #한 번에 읽어올 오디오 프레임 수. 크면 CPU 부하가 줄고 지연도 줄어듦.

        def setup_audio():  
            nonlocal audio_interface, stream, device_sample_rate, input_device_index
            try:
                #pyAudio 객채 생성
                if audio_interface is None:
                    logger.debug("Creating PyAudio interface...")
                    audio_interface = pyaudio.PyAudio()

                #명시된 마이크인덱스 없으면 기본으로 설정
                if input_device_index is None:
                    try:
                        default_device = audio_interface.get_default_input_device_info()
                        input_device_index = default_device['index']
                        logger.debug(f"No device index supplied; using default device {input_device_index}")
                    except OSError as e:
                        logger.debug(f"Default device retrieval failed: {e}")
                        input_device_index = None

                #16000Hz로 시도해본 후, 입력이 가지고 있는 최고 샘플레이트로 시도. 필요하다면 48000Hz로 fallback이 됨.
                sample_rates_to_try = [16000]
                if input_device_index is not None:
                    highest_rate = get_highest_sample_rate(audio_interface, input_device_index)
                    if highest_rate != 16000:
                        sample_rates_to_try.append(highest_rate)
                else:
                    sample_rates_to_try.append(48000)

                logger.debug(f"Sample rates to try for device {input_device_index}: {sample_rates_to_try}")

                #샘플레이트별 초기화
                for rate in sample_rates_to_try:
                    try:
                        device_sample_rate = rate
                        logger.debug(f"Attempting to initialize audio stream at {device_sample_rate} Hz.")
                        stream = initialize_audio_stream(audio_interface, device_sample_rate, chunk_size)
                        if stream is not None:
                            logger.debug(
                                f"Audio recording initialized successfully at {device_sample_rate} Hz, "
                                f"reading {chunk_size} frames at a time"
                            )
                            return True
                    except Exception as e:
                        logger.warning(f"Failed to initialize audio stream at {device_sample_rate} Hz: {e}")
                        continue

                # If we reach here, none of the sample rates worked
                #샘플레이트를 모두 시도해도 실패하면 Exception을 발생시키고 로그 남김.
                raise Exception("Failed to initialize audio stream with all sample rates.")

            except Exception as e:
                logger.exception(f"Error initializing pyaudio audio recording: {e}")
                if audio_interface:
                    audio_interface.terminate()
                return False

        logger.debug(f"Starting audio data worker with target_sample_rate={target_sample_rate}, "
                    f"buffer_size={buffer_size}, input_device_index={input_device_index}")

        #setup_audio가 실패했다면, 오류냄.
        if not setup_audio():
            raise Exception("Failed to set up audio recording.")

        buffer = bytearray()
        #Silero VAD 모델이 요구하는 최소 오디오 길이. 너무 짧은 청크를 주면 분석이 어려우니 2배 버퍼를 설정.
        silero_buffer_size = 2 * buffer_size  # Silero complains if too short

        #마지막으로 큐에 데이터를 보낸 시간 저장 → 로깅 및 성능 감시용.
        time_since_last_buffer_message = 0

        try:
            #shutdown_event가 설정될 때까지 계속해서 오디오 입력을 받고 처리함.
            while not shutdown_event.is_set():
                try:
                    #stream.read()로 마이크에서 chunk 단위로 raw 오디오 데이터를 읽음.
                    data = stream.read(chunk_size, exception_on_overflow=False)

                    #마이크를 사용중이라면,
                    if use_microphone.value:
                        #오디어 청크 전처리해서 buffer에 더해두기.
                        processed_data = preprocess_audio(data, device_sample_rate, target_sample_rate)
                        buffer += processed_data

                        #버퍼가 silero_buffer_size에 도달했거나 초과했는지 확인한다
                        while len(buffer) >= silero_buffer_size:
                            #버퍼에서 silero_buffer_size만큼의 데이터를 추출한다
                            to_process = buffer[:silero_buffer_size]
                            buffer = buffer[silero_buffer_size:]

                            #1초 이상 간격이 생기면 큐에 오디오가 잘 들어가는지 로그로 확인.
                            if time_since_last_buffer_message:
                                time_passed = time.time() - time_since_last_buffer_message
                                if time_passed > 1:
                                    logger.debug("_audio_data_worker writing audio data into queue.")
                                    time_since_last_buffer_message = time.time()
                            else:
                                time_since_last_buffer_message = time.time()

                            #추출된 데이터를 audio_queue에 전달한다
                            audio_queue.put(to_process)

                except OSError as e:
                    #마이크오버플로우라면, 프레임드랍될수 있다는 warning 출력
                    if e.errno == pyaudio.paInputOverflowed:
                        logger.warning("Input overflowed. Frame dropped.")
                    else:
                        #마이크오버플로우가 아닌 OSError라면 로그에 표시
                        logger.error(f"OSError during recording: {e}", exc_info=True)
                        # Attempt to reinitialize the stream
                        logger.error("Attempting to reinitialize the audio stream...")

                        #재설정 해보기
                        try:
                            if stream:
                                stream.stop_stream()
                                stream.close()
                        except Exception:
                            pass
                        time.sleep(1)
                        if not setup_audio():
                            logger.error("Failed to reinitialize audio stream. Exiting.")
                            break
                        else:
                            logger.error("Audio stream reinitialized successfully.")
                    continue

                #그외 오류가 떠도 재설정 시도.
                except Exception as e:
                    logger.error(f"Unknown error during recording: {e}")
                    tb_str = traceback.format_exc()
                    logger.error(f"Traceback: {tb_str}")
                    logger.error(f"Error: {e}")
                    # Attempt to reinitialize the stream
                    logger.info("Attempting to reinitialize the audio stream...")
                    try:
                        if stream:
                            stream.stop_stream()
                            stream.close()
                    except Exception:
                        pass

                    time.sleep(1)
                    if not setup_audio():
                        logger.error("Failed to reinitialize audio stream. Exiting.")
                        break
                    else:
                        logger.info("Audio stream reinitialized successfully.")
                    continue
        
        #키보드 인터럽트나면 종료.
        except KeyboardInterrupt:
            interrupt_stop_event.set()
            logger.debug("Audio data worker process finished due to KeyboardInterrupt")
        finally:
            # After recording stops, feed any remaining audio data
            #루프가 종료되면 남아있는 버퍼 데이터를 마지막으로 전송.
            if buffer:
                audio_queue.put(bytes(buffer))

            try:
                if stream:
                    stream.stop_stream()
                    stream.close()
            except Exception:
                pass
            if audio_interface:
                audio_interface.terminate()

    def wakeup(self):
        """
        만약 wake word 모드라면, 마치 wake word(예: ‘Jarvis’)를 들은 것처럼 깨워라.
        wakeup한 후에 오디오 듣기 활성화가 되는거임.
        """
        self.listen_start = time.time()

    def abort(self):
        #비상중단
        #현재 상태 저장
        state = self.state
        #음성 감지를 기반으로 녹음을 자동 시작/정리하는 설정을 비활성화
        self.start_recording_on_voice_activity = False
        self.stop_recording_on_voice_deactivity = False

        #인터럽트이벤트 플래그설정
        self.interrupt_stop_event.set()

        #inactive가 아니라면->녹음중이나 감지중이라면 스레드동기화 기다림.
        if self.state != "inactive": # if inactive, was_interrupted will never be set
            self.was_interrupted.wait()
            self._set_state("transcribing")
        self.was_interrupted.clear()
        if self.is_recording: # if recording, make sure to stop the recorder
            self.stop()


    def wait_audio(self):
        """
        오디오 녹음 프로세스의 시작과 완료를 기다립니다.

        이 메서드는 다음 작업을 수행합니다:
        - 음성 활동(voice activity)이 감지되어 녹음이 시작될 때까지 기다립니다. (녹음이 아직 시작되지 않았다면)

        - 음성 비활성(voice inactivity)이 감지되어 녹음이 완료될 때까지 기다립니다.
        
        - 녹음된 프레임을 이용하여 오디오 버퍼를 설정합니다.

        - 녹음과 관련된 속성들을 초기화(리셋)합니다.

        부수 효과:
        - 인스턴스의 상태(state)를 업데이트합니다.

        - 녹음된 오디오 데이터를 audio 속성에 저장합니다.
        """

        try:
            #리스닝시작이 설정되지 않았다면, 현재시간으로 설정함.
            logger.info("Setting listen time")
            if self.listen_start == 0:
                self.listen_start = time.time()

            #아직 녹음이 시작되지 않았다면, 음성이 인식될때까지 기다림.
            if not self.is_recording and not self.frames:
                self._set_state("listening")
                self.start_recording_on_voice_activity = True

                # Wait until recording starts
                logger.debug('Waiting for recording start')
                while not self.interrupt_stop_event.is_set():
                    if self.start_recording_event.wait(timeout=0.02):
                        break

            #만약 녹음중 이라면, 음성이 끊길때 자동으로 멈추도록 설정.
            if self.is_recording:
                self.stop_recording_on_voice_deactivity = True

                #stop_recording_event가 True가 될 때까지 대기 (음성 끊김 감지)
                logger.debug('Waiting for recording stop')
                while not self.interrupt_stop_event.is_set():
                    if (self.stop_recording_event.wait(timeout=0.02)):
                        break
            
            #녹음된 오디오가 없다면 마지막 프레임(last_frames)을 대신 사용
            frames = self.frames
            if len(frames) == 0:
                frames = self.last_frames

            #resume용 백데이팅에 필요한 샘플 계산.
            samples_to_keep = int(self.sample_rate * self.backdate_resume_seconds)

            # First convert all current frames to audio array
            full_audio_array = np.frombuffer(b''.join(frames), dtype=np.int16)
            full_audio = full_audio_array.astype(np.float32) / INT16_MAX_ABS_VALUE

            # Calculate how many samples we need to keep for backdating resume
            if samples_to_keep > 0:
                samples_to_keep = min(samples_to_keep, len(full_audio))
                # Keep the last N samples for backdating resume
                frames_to_read_audio = full_audio[-samples_to_keep:]

                # Convert the audio back to int16 bytes for frames
                frames_to_read_int16 = (frames_to_read_audio * INT16_MAX_ABS_VALUE).astype(np.int16)
                frame_bytes = frames_to_read_int16.tobytes()

                # Split into appropriate frame sizes (assuming standard frame size)
                FRAME_SIZE = 2048  # Typical frame size
                frames_to_read = []
                for i in range(0, len(frame_bytes), FRAME_SIZE):
                    frame = frame_bytes[i:i + FRAME_SIZE]
                    if frame:  # Only add non-empty frames
                        frames_to_read.append(frame)
            else:
                frames_to_read = []

            # Process backdate stop seconds
            samples_to_remove = int(self.sample_rate * self.backdate_stop_seconds)

            if samples_to_remove > 0:
                if samples_to_remove < len(full_audio):
                    self.audio = full_audio[:-samples_to_remove]
                    logger.debug(f"Removed {samples_to_remove} samples "
                        f"({samples_to_remove/self.sample_rate:.3f}s) from end of audio")
                else:
                    self.audio = np.array([], dtype=np.float32)
                    logger.debug("Cleared audio (samples_to_remove >= audio length)")
            else:
                self.audio = full_audio
                logger.debug(f"No samples removed, final audio length: {len(self.audio)}")

            self.frames.clear()
            self.last_frames.clear()
            self.frames.extend(frames_to_read)

            # Reset backdating parameters
            self.backdate_stop_seconds = 0.0
            self.backdate_resume_seconds = 0.0

            self.listen_start = 0

            self._set_state("inactive")

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt in wait_audio, shutting down")
            self.shutdown()
            raise  # Re-raise the exception after cleanup

    '''
    오디오 바이트 데이터를 STT모듈 백엔드에 전달하여 최종 텍스트로 변환하고, 결과를 리턴합니다.
    비동기 큐나 실시간 STT와 달리, 이건 문장이 끝났을 때 호출되는 최종 변환 과정입니다.
    '''
    def perform_final_transcription(self, audio_bytes=None):
        start_time = 0
        with self.transcription_lock:
            try:
                if self.transcribe_count == 0:
                    logger.debug("Adding transcription request, no early transcription started")
                    start_time = time.time()  # Start timing
                    self.parent_transcription_pipe.send((audio_bytes, self.language))
                    self.transcribe_count += 1

                while self.transcribe_count > 0:
                    logger.debug(F"Receive from parent_transcription_pipe after sendiung transcription request, transcribe_count: {self.transcribe_count}")
                    if not self.parent_transcription_pipe.poll(0.1): # check if transcription done
                        if self.interrupt_stop_event.is_set(): # 인터럽트 확인
                            self.was_interrupted.set()
                            self._set_state("inactive")
                            return "" # return empty string if interrupted
                        continue
                    status, result = self.parent_transcription_pipe.recv()
                    self.transcribe_count -= 1
                    
                    
                self._set_state("inactive")
                if status == 'success':
                    text, info = result 

                    self.last_transcription_bytes = copy.deepcopy(audio_bytes)
                    self.last_transcription_bytes_b64 = base64.b64encode(self.last_transcription_bytes.tobytes()).decode('utf-8')
                    
                    end_time = time.time()  # End timing
                    transcription_time = end_time - start_time

                    if start_time:
                        if self.print_transcription_time:
                            print(f"모델 지연시간: {transcription_time:.2f} seconds")
                        else:
                            logger.debug(f"모델 지연시간: {transcription_time:.2f} seconds")
                    return "" if self.interrupt_stop_event.is_set() else (text, info) #인터럽트나면, 빈 문자열 반환
                else:
                    logger.error(f"Transcription error: {result}")
                    raise Exception(result)
            except Exception as e:
                logger.error(f"Error during transcription: {str(e)}", exc_info=True)
                raise e


    def transcribe(self):
        """
        이 클래스 인스턴스가 캡처한 오디오를 faster_whisper 모델을 사용하여 텍스트로 변환(transcribe) 합니다.

        음성 감지 시 자동으로 녹음을 시작합니다.
          (단, 사용자가 recorder.start()를 직접 호출하지 않은 경우에만)

        음성 비활성(침묵) 시 자동으로 녹음을 멈춥니다.
          (단, 사용자가 recorder.stop()을 직접 호출하지 않은 경우에만)
        
        녹음이 완료된 오디오 데이터를 처리하여 텍스트로 변환된 결과를 생성합니다.
        
        매개변수:
            on_transcription_finished (callable, optional):
            변환이 완료되었을 때 실행할 콜백 함수입니다.

            이 인자를 제공하면 비동기(asynchronous) 방식으로 동작하며,
              해당 콜백 함수의 인자로 텍스트 결과가 전달됩니다.
            이 인자를 생략하면, 동기(synchronous) 방식으로 작동하며,
              함수는 텍스트 결과를 즉시 반환합니다.

        반환값 (콜백이 없는 경우):
            str: 변환된 텍스트 문자열 (녹음된 오디오의 STT 결과)

        예외:
            Exception: 음성 인식(transcription) 처리 중 오류가 발생한 경우 예외가 발생합니다.
        """
        audio_copy = copy.deepcopy(self.audio)
        self._set_state("transcribing")
        if self.on_transcription_start:
            abort_value = self.on_transcription_start(audio_copy)
            if not abort_value:
                return self.perform_final_transcription(audio_copy)
            return None
        else:
            return self.perform_final_transcription(audio_copy)


    def _process_wakeword(self, data):
        """
        Processes audio data to detect wake words.
        """
        if self.wakeword_backend in {'pvp', 'pvporcupine'}:
            pcm = struct.unpack_from(
                "h" * self.buffer_size,
                data
            )
            porcupine_index = self.porcupine.process(pcm)
            if self.debug_mode:
                logger.info(f"wake words porcupine_index: {porcupine_index}")
            return porcupine_index

        elif self.wakeword_backend in {'oww', 'openwakeword', 'openwakewords'}:
            pcm = np.frombuffer(data, dtype=np.int16)
            prediction = self.owwModel.predict(pcm)
            max_score = -1
            max_index = -1
            wake_words_in_prediction = len(self.owwModel.prediction_buffer.keys())
            self.wake_words_sensitivities
            if wake_words_in_prediction:
                for idx, mdl in enumerate(self.owwModel.prediction_buffer.keys()):
                    scores = list(self.owwModel.prediction_buffer[mdl])
                    if scores[-1] >= self.wake_words_sensitivity and scores[-1] > max_score:
                        max_score = scores[-1]
                        max_index = idx
                if self.debug_mode:
                    logger.info(f"wake words oww max_index, max_score: {max_index} {max_score}")
                return max_index  
            else:
                if self.debug_mode:
                    logger.info(f"wake words oww_index: -1")
                return -1

        if self.debug_mode:        
            logger.info("wake words no match")

        return -1

    def text(self,
             on_transcription_finished=None,
             ):
        """
        이 클래스 인스턴스에서 캡처한 오디오를
        faster_whisper 모델을 사용해 텍스트로 변환합니다.

        - recorder.start()를 수동으로 호출하지 않은 경우,
          음성 활동(voice activity)이 감지되면 자동으로 녹음을 시작합니다.

        - recorder.stop()을 수동으로 호출하지 않은 경우,
          음성 비활동(voice deactivity)이 감지되면 자동으로 녹음을 중지합니다.

        - 녹음된 오디오를 처리하여 텍스트로 변환합니다.

        Args:
        매개변수:
            on_transcription_finished (callable, optional): 
              변환이 완료되었을 때 실행할 콜백 함수입니다.
            이 함수가 제공되면 변환은 비동기적으로 수행되며, 결과 텍스트는 이 콜백의 인자로 전달됩니다.
              제공되지 않으면 변환은 동기적으로 수행되며, 결과는 함수의 반환값으로 전달됩니다.

        반환값 (콜백이 제공되지 않은 경우):
            str: 변환된 텍스트 문자열.
        """
        self.interrupt_stop_event.clear()
        self.was_interrupted.clear()
        try:
            self.wait_audio()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt in text() method")
            self.shutdown()
            raise  # Re-raise the exception after cleanup

        if self.is_shut_down or self.interrupt_stop_event.is_set():
            if self.interrupt_stop_event.is_set():
                self.was_interrupted.set()
            return ""

        #콜백함수가 있다면 스레드할당하여 ars시작, 아니면 문자열 반환
        if on_transcription_finished:
            threading.Thread(target=on_transcription_finished,
                            args=(self.transcribe(),)).start()
        else:
            return self.transcribe()


    def format_number(self, num):
        # Convert the number to a string
        num_str = f"{num:.10f}"  # Ensure precision is sufficient
        # Split the number into integer and decimal parts
        integer_part, decimal_part = num_str.split('.')
        # Take the last two digits of the integer part and the first two digits of the decimal part
        result = f"{integer_part[-2:]}.{decimal_part[:2]}"
        return result

    def start(self, frames = None):
        """
        Starts recording audio directly without waiting for voice activity.
        음성 활동을 기다리지 않고 바로 오디오 녹음을 시작합니다.
        """

        # Ensure there's a minimum interval
        # between stopping and starting recording
        if (time.time() - self.recording_stop_time
                < self.min_gap_between_recordings):
            logger.info("Attempted to start recording "
                         "too soon after stopping."
                         )
            return self

        logger.info("recording started")
        self._set_state("recording")
        self.text_storage = []
        self.realtime_stabilized_text = ""
        self.realtime_stabilized_safetext = ""
        self.wakeword_detected = False
        self.wake_word_detect_time = 0
        self.frames = []
        if frames:
            self.frames = frames
        self.is_recording = True

        self.recording_start_time = time.time()
        self.is_silero_speech_active = False
        self.is_webrtc_speech_active = False
        self.stop_recording_event.clear()
        self.start_recording_event.set()

        if self.on_recording_start:
            self.on_recording_start()

        return self

    def stop(self,
             backdate_stop_seconds: float = 0.0,
             backdate_resume_seconds: float = 0.0,
        ):
        """
        오디오 녹음을 중지합니다.

        Args:
        - backdate_stop_seconds (float, default="0.0"): 
            실제로 음성이 멈춘 시점보다 나중에 stop 명령이 호출된 경우,
            실제 멈춘 시점으로 시간을 되돌려서 중지하도록 지정하는 값입니다.

        - backdate_resume_seconds (float, default="0.0"): 
            다시 청취를 시작하는 시점을 몇 초 앞당겨서 설정할지 지정합니다.
            예: 재녹음을 위해 음성 시작 부분을 놓치지 않도록 약간 앞 시점부터 다시 듣기 위함입니다.
        """

        # Ensure there's a minimum interval
        # between starting and stopping recording
        if (time.time() - self.recording_start_time
                < self.min_length_of_recording):
            logger.info("Attempted to stop recording "
                         "too soon after starting."
                         )
            return self

        logger.info("recording stopped")
        self.last_frames = copy.deepcopy(self.frames)
        self.backdate_stop_seconds = backdate_stop_seconds
        self.backdate_resume_seconds = backdate_resume_seconds
        self.is_recording = False
        self.recording_stop_time = time.time()
        self.is_silero_speech_active = False
        self.is_webrtc_speech_active = False
        self.silero_check_time = 0
        self.start_recording_event.clear()
        self.stop_recording_event.set()

        self.last_recording_start_time = self.recording_start_time
        self.last_recording_stop_time = self.recording_stop_time

        if self.on_recording_stop:
            self.on_recording_stop()

        return self

    def listen(self):
        """
        즉시 “듣기(listen)” 상태로 전환합니다.
        이 상태는 예를 들어 웨이크 워드(wake word)가 감지된 이후의 상태입니다.
        이제 녹음기는 음성 활성화를 “청취”하게 됩니다.
        음성이 감지되면 “녹음(recording)” 상태로 전환됩니다.
        """
        self.listen_start = time.time()
        self._set_state("listening")
        self.start_recording_on_voice_activity = True

    def feed_audio(self, chunk, original_sample_rate=16000):
        """
        오디오 청크(audio chunk)를 처리 파이프라인에 입력합니다.
        청크는 버퍼 크기에 도달할 때까지 누적되며,  
        버퍼 크기에 도달하면 누적된 데이터가 audio_queue로 전달됩니다.
        """
        # Check if the buffer attribute exists, if not, initialize it
        if not hasattr(self, 'buffer'):
            self.buffer = bytearray()

        # Check if input is a NumPy array
        if isinstance(chunk, np.ndarray):
            # Handle stereo to mono conversion if necessary
            if chunk.ndim == 2:
                chunk = np.mean(chunk, axis=1)

            # Resample to 16000 Hz if necessary
            if original_sample_rate != 16000:
                num_samples = int(len(chunk) * 16000 / original_sample_rate)
                chunk = resample(chunk, num_samples)

            # Ensure data type is int16
            chunk = chunk.astype(np.int16)

            # Convert the NumPy array to bytes
            chunk = chunk.tobytes()

        # Append the chunk to the buffer
        self.buffer += chunk
        buf_size = 2 * self.buffer_size  # silero complains if too short

        # Check if the buffer has reached or exceeded the buffer_size
        while len(self.buffer) >= buf_size:
            # Extract self.buffer_size amount of data from the buffer
            to_process = self.buffer[:buf_size]
            self.buffer = self.buffer[buf_size:]

            # Feed the extracted data to the audio_queue
            self.audio_queue.put(to_process)

    def set_microphone(self, microphone_on=True):
        """
        Set the microphone on or off.
        마이크 온오프를 설정합니다.
        """
        logger.info("Setting microphone to: " + str(microphone_on))
        self.use_microphone.value = microphone_on

    def shutdown(self):
        """
        Safely shuts down the audio recording by stopping the
        recording worker and closing the audio stream.
        """

        with self.shutdown_lock:
            if self.is_shut_down:
                return

            print("\033[91mRealtimeSTT shutting down\033[0m")

            # Force wait_audio() and text() to exit
            self.is_shut_down = True
            self.start_recording_event.set()
            self.stop_recording_event.set()

            self.shutdown_event.set()
            self.is_recording = False
            self.is_running = False

            logger.debug('Finishing recording thread')
            if self.recording_thread:
                self.recording_thread.join()

            logger.debug('Terminating reader process')

            # Give it some time to finish the loop and cleanup.
            if self.use_microphone.value:
                self.reader_process.join(timeout=10)

                if self.reader_process.is_alive():
                    logger.warning("Reader process did not terminate "
                                    "in time. Terminating forcefully."
                                    )
                    self.reader_process.terminate()

            logger.debug('Terminating transcription process')
            self.transcript_process.join(timeout=10)

            if self.transcript_process.is_alive():
                logger.warning("Transcript process did not terminate "
                                "in time. Terminating forcefully."
                                )
                self.transcript_process.terminate()

            self.parent_transcription_pipe.close()

            logger.debug('Finishing realtime thread')
            if self.realtime_thread:
                self.realtime_thread.join()

            if self.enable_realtime_transcription:
                if self.realtime_model_type:
                    del self.realtime_model_type
                    self.realtime_model_type = None
            gc.collect()

    def _recording_worker(self):
        """
        오디오 입력을 지속적으로 모니터링하면서 음성 활동을 감지하고,
        그에 따라 녹음을 시작하거나 중지하는 주요 워커 메서드입니다.
        """

        if self.use_extended_logging:
            logger.debug('Debug: Entering try block')

        last_inner_try_time = 0
        try:
            if self.use_extended_logging:
                logger.debug('Debug: Initializing variables')
            time_since_last_buffer_message = 0
            was_recording = False
            delay_was_passed = False
            wakeword_detected_time = None
            wakeword_samples_to_remove = None
            self.allowed_to_early_transcribe = True

            if self.use_extended_logging:
                logger.debug('Debug: Starting main loop')
            # Continuously monitor audio for voice activity
            while self.is_running:

                # if self.use_extended_logging:
                #     logger.debug('Debug: Entering inner try block')
                if last_inner_try_time:
                    last_processing_time = time.time() - last_inner_try_time
                    if last_processing_time > 0.1:
                        if self.use_extended_logging:
                            logger.warning('### WARNING: PROCESSING TOOK TOO LONG')
                last_inner_try_time = time.time()
                try:
                    # if self.use_extended_logging:
                    #     logger.debug('Debug: Trying to get data from audio queue')
                    try:
                        data = self.audio_queue.get(timeout=0.01)
                        self.last_words_buffer.append(data)
                    except queue.Empty:
                        # if self.use_extended_logging:
                        #     logger.debug('Debug: Queue is empty, checking if still running')
                        if not self.is_running:
                            if self.use_extended_logging:
                                logger.debug('Debug: Not running, breaking loop')
                            break
                        # if self.use_extended_logging:
                        #     logger.debug('Debug: Continuing to next iteration')
                        continue

                    if self.use_extended_logging:
                        logger.debug('Debug: Checking for on_recorded_chunk callback')
                    if self.on_recorded_chunk:
                        if self.use_extended_logging:
                            logger.debug('Debug: Calling on_recorded_chunk')
                        self.on_recorded_chunk(data)

                    if self.use_extended_logging:
                        logger.debug('Debug: Checking if handle_buffer_overflow is True')
                    if self.handle_buffer_overflow:
                        if self.use_extended_logging:
                            logger.debug('Debug: Handling buffer overflow')
                        # Handle queue overflow
                        if (self.audio_queue.qsize() >
                                self.allowed_latency_limit):
                            if self.use_extended_logging:
                                logger.debug('Debug: Queue size exceeds limit, logging warnings')
                            logger.warning("Audio queue size exceeds "
                                            "latency limit. Current size: "
                                            f"{self.audio_queue.qsize()}. "
                                            "Discarding old audio chunks."
                                            )

                        if self.use_extended_logging:
                            logger.debug('Debug: Discarding old chunks if necessary')
                        while (self.audio_queue.qsize() >
                                self.allowed_latency_limit):

                            data = self.audio_queue.get()

                except BrokenPipeError:
                    logger.error("BrokenPipeError _recording_worker", exc_info=True)
                    self.is_running = False
                    break

                if self.use_extended_logging:
                    logger.debug('Debug: Updating time_since_last_buffer_message')
                # Feed the extracted data to the audio_queue
                if time_since_last_buffer_message:
                    time_passed = time.time() - time_since_last_buffer_message
                    if time_passed > 1:
                        if self.use_extended_logging:
                            logger.debug("_recording_worker processing audio data")
                        time_since_last_buffer_message = time.time()
                else:
                    time_since_last_buffer_message = time.time()

                if self.use_extended_logging:
                    logger.debug('Debug: Initializing failed_stop_attempt')
                failed_stop_attempt = False

                if self.use_extended_logging:
                    logger.debug('Debug: Checking if not recording')
                if not self.is_recording:
                    if self.use_extended_logging:
                        logger.debug('Debug: Handling not recording state')
                    # Handle not recording state
                    time_since_listen_start = (time.time() - self.listen_start
                                            if self.listen_start else 0)

                    wake_word_activation_delay_passed = (
                        time_since_listen_start >
                        self.wake_word_activation_delay
                    )

                    if self.use_extended_logging:
                        logger.debug('Debug: Handling wake-word timeout callback')
                    # Handle wake-word timeout callback
                    if wake_word_activation_delay_passed \
                            and not delay_was_passed:

                        if self.use_wake_words and self.wake_word_activation_delay:
                            if self.on_wakeword_timeout:
                                if self.use_extended_logging:
                                    logger.debug('Debug: Calling on_wakeword_timeout')
                                self.on_wakeword_timeout()
                    delay_was_passed = wake_word_activation_delay_passed

                    if self.use_extended_logging:
                        logger.debug('Debug: Setting state and spinner text')
                    # Set state and spinner text
                    if not self.recording_stop_time:
                        if self.use_wake_words \
                                and wake_word_activation_delay_passed \
                                and not self.wakeword_detected:
                            if self.use_extended_logging:
                                logger.debug('Debug: Setting state to "wakeword"')
                            self._set_state("wakeword")
                        else:
                            if self.listen_start:
                                if self.use_extended_logging:
                                    logger.debug('Debug: Setting state to "listening"')
                                self._set_state("listening")
                            else:
                                if self.use_extended_logging:
                                    logger.debug('Debug: Setting state to "inactive"')
                                self._set_state("inactive")

                    if self.use_extended_logging:
                        logger.debug('Debug: Checking wake word conditions')
                    if self.use_wake_words and wake_word_activation_delay_passed:
                        try:
                            if self.use_extended_logging:
                                logger.debug('Debug: Processing wakeword')
                            wakeword_index = self._process_wakeword(data)

                        except struct.error:
                            logger.error("Error unpacking audio data "
                                        "for wake word processing.", exc_info=True)
                            continue

                        except Exception as e:
                            logger.error(f"Wake word processing error: {e}", exc_info=True)
                            continue

                        if self.use_extended_logging:
                            logger.debug('Debug: Checking if wake word detected')
                        # If a wake word is detected                        
                        if wakeword_index >= 0:
                            if self.use_extended_logging:
                                logger.debug('Debug: Wake word detected, updating variables')
                            self.wake_word_detect_time = time.time()
                            wakeword_detected_time = time.time()
                            wakeword_samples_to_remove = int(self.sample_rate * self.wake_word_buffer_duration)
                            self.wakeword_detected = True
                            if self.on_wakeword_detected:
                                if self.use_extended_logging:
                                    logger.debug('Debug: Calling on_wakeword_detected')
                                self.on_wakeword_detected()

                    if self.use_extended_logging:
                        logger.debug('Debug: Checking voice activity conditions')
                    # Check for voice activity to
                    # trigger the start of recording
                    if ((not self.use_wake_words
                        or not wake_word_activation_delay_passed)
                            and self.start_recording_on_voice_activity) \
                            or self.wakeword_detected:

                        if self.use_extended_logging:
                            logger.debug('Debug: Checking if voice is active')

                        if self._is_voice_active():

                            if self.on_vad_start:
                               self.on_vad_start()

                            if self.use_extended_logging:
                                logger.debug('Debug: Voice activity detected')
                            logger.info("voice activity detected")

                            if self.use_extended_logging:
                                logger.debug('Debug: Starting recording')
                            self.start()

                            self.start_recording_on_voice_activity = False

                            if self.use_extended_logging:
                                logger.debug('Debug: Adding buffered audio to frames')
                            # Add the buffered audio
                            # to the recording frames
                            self.frames.extend(list(self.audio_buffer))
                            self.audio_buffer.clear()

                            if self.use_extended_logging:
                                logger.debug('Debug: Resetting Silero VAD model states')
                            self.silero_vad_model.reset_states()
                        else:
                            if self.use_extended_logging:
                                logger.debug('Debug: Checking voice activity')
                            data_copy = data[:]
                            self._check_voice_activity(data_copy)

                    if self.use_extended_logging:
                        logger.debug('Debug: Resetting speech_end_silence_start')
                    self.speech_end_silence_start = 0

                else:
                    if self.use_extended_logging:
                        logger.debug('Debug: Handling recording state')
                    # If we are currently recording
                    if wakeword_samples_to_remove and wakeword_samples_to_remove > 0:
                        if self.use_extended_logging:
                            logger.debug('Debug: Removing wakeword samples')
                        # Remove samples from the beginning of self.frames
                        samples_removed = 0
                        while wakeword_samples_to_remove > 0 and self.frames:
                            frame = self.frames[0]
                            frame_samples = len(frame) // 2  # Assuming 16-bit audio
                            if wakeword_samples_to_remove >= frame_samples:
                                self.frames.pop(0)
                                samples_removed += frame_samples
                                wakeword_samples_to_remove -= frame_samples
                            else:
                                self.frames[0] = frame[wakeword_samples_to_remove * 2:]
                                samples_removed += wakeword_samples_to_remove
                                samples_to_remove = 0
                        
                        wakeword_samples_to_remove = 0

                    if self.use_extended_logging:
                        logger.debug('Debug: Checking if stop_recording_on_voice_deactivity is True')
                    # Stop the recording if silence is detected after speech
                    if self.stop_recording_on_voice_deactivity:
                        if self.use_extended_logging:
                            logger.debug('Debug: Determining if speech is detected')
                        is_speech = (
                            self._is_silero_speech(data) if self.silero_deactivity_detection
                            else self._is_webrtc_speech(data, True)
                        )

                        if self.use_extended_logging:
                            logger.debug('Debug: Formatting speech_end_silence_start')
                        if not self.speech_end_silence_start:
                            str_speech_end_silence_start = "0"
                        else:
                            str_speech_end_silence_start = datetime.datetime.fromtimestamp(self.speech_end_silence_start).strftime('%H:%M:%S.%f')[:-3]
                        if self.use_extended_logging:
                            logger.debug(f"is_speech: {is_speech}, str_speech_end_silence_start: {str_speech_end_silence_start}")

                        if self.use_extended_logging:
                            logger.debug('Debug: Checking if speech is not detected')
                        if not is_speech:
                            if self.use_extended_logging:
                                logger.debug('Debug: Handling voice deactivity')
                            # Voice deactivity was detected, so we start
                            # measuring silence time before stopping recording
                            if self.speech_end_silence_start == 0 and \
                                (time.time() - self.recording_start_time > self.min_length_of_recording):

                                self.speech_end_silence_start = time.time()
                                self.awaiting_speech_end = True

                            if self.use_extended_logging:
                                logger.debug('Debug: Checking early transcription conditions')
                            if self.speech_end_silence_start and self.early_transcription_on_silence and len(self.frames) > 0 and \
                                (time.time() - self.speech_end_silence_start > self.early_transcription_on_silence) and \
                                self.allowed_to_early_transcribe:
                                    if self.use_extended_logging:
                                        logger.debug("Debug:Adding early transcription request")
                                    self.transcribe_count += 1
                                    audio_array = np.frombuffer(b''.join(self.frames), dtype=np.int16)
                                    audio = audio_array.astype(np.float32) / INT16_MAX_ABS_VALUE

                                    if self.use_extended_logging:
                                        logger.debug("Debug: early transcription request pipe send")
                                    self.parent_transcription_pipe.send((audio, self.language))
                                    if self.use_extended_logging:
                                        logger.debug("Debug: early transcription request pipe send return")
                                    self.allowed_to_early_transcribe = False

                        else:
                            self.awaiting_speech_end = False
                            if self.use_extended_logging:
                                logger.debug('Debug: Handling speech detection')
                            if self.speech_end_silence_start:
                                if self.use_extended_logging:
                                    logger.info("Resetting self.speech_end_silence_start")
                                self.speech_end_silence_start = 0
                                self.allowed_to_early_transcribe = True

                        if self.use_extended_logging:
                            logger.debug('Debug: Checking if silence duration exceeds threshold')
                        # Wait for silence to stop recording after speech
                        if self.speech_end_silence_start and time.time() - \
                                self.speech_end_silence_start >= \
                                self.post_speech_silence_duration:

                            if self.on_vad_stop:
                                self.on_vad_stop()

                            if self.use_extended_logging:
                                logger.debug('Debug: Formatting silence start time')
                            # Get time in desired format (HH:MM:SS.nnn)
                            silence_start_time = datetime.datetime.fromtimestamp(self.speech_end_silence_start).strftime('%H:%M:%S.%f')[:-3]

                            if self.use_extended_logging:
                                logger.debug('Debug: Calculating time difference')
                            # Calculate time difference
                            time_diff = time.time() - self.speech_end_silence_start

                            if self.use_extended_logging:
                                logger.debug('Debug: Logging voice deactivity detection')
                                logger.info(f"voice deactivity detected at {silence_start_time}, "
                                        f"time since silence start: {time_diff:.3f} seconds")

                                logger.debug('Debug: Appending data to frames and stopping recording')
                            self.frames.append(data)
                            self.stop()
                            if not self.is_recording:
                                if self.use_extended_logging:
                                    logger.debug('Debug: Resetting speech_end_silence_start')
                                self.speech_end_silence_start = 0

                                if self.use_extended_logging:
                                    logger.debug('Debug: Handling non-wake word scenario')
                            else:
                                if self.use_extended_logging:
                                    logger.debug('Debug: Setting failed_stop_attempt to True')
                                failed_stop_attempt = True

                            self.awaiting_speech_end = False

                if self.use_extended_logging:
                    logger.debug('Debug: Checking if recording stopped')
                if not self.is_recording and was_recording:
                    if self.use_extended_logging:
                        logger.debug('Debug: Resetting after stopping recording')
                    # Reset after stopping recording to ensure clean state
                    self.stop_recording_on_voice_deactivity = False

                if self.use_extended_logging:
                    logger.debug('Debug: Checking Silero time')
                if time.time() - self.silero_check_time > 0.1:
                    self.silero_check_time = 0

                if self.use_extended_logging:
                    logger.debug('Debug: Handling wake word timeout')
                # Handle wake word timeout (waited to long initiating
                # speech after wake word detection)
                if self.wake_word_detect_time and time.time() - \
                        self.wake_word_detect_time > self.wake_word_timeout:

                    self.wake_word_detect_time = 0
                    if self.wakeword_detected and self.on_wakeword_timeout:
                        if self.use_extended_logging:
                            logger.debug('Debug: Calling on_wakeword_timeout')
                        self.on_wakeword_timeout()
                    self.wakeword_detected = False

                if self.use_extended_logging:
                    logger.debug('Debug: Updating was_recording')
                was_recording = self.is_recording

                if self.use_extended_logging:
                    logger.debug('Debug: Checking if recording and not failed stop attempt')
                if self.is_recording and not failed_stop_attempt:
                    if self.use_extended_logging:
                        logger.debug('Debug: Appending data to frames')
                    self.frames.append(data)

                if self.use_extended_logging:
                    logger.debug('Debug: Checking if not recording or speech end silence start')
                if not self.is_recording or self.speech_end_silence_start:
                    if self.use_extended_logging:
                        logger.debug('Debug: Appending data to audio buffer')
                    self.audio_buffer.append(data)

        except Exception as e:
            logger.debug('Debug: Caught exception in main try block')
            if not self.interrupt_stop_event.is_set():
                logger.error(f"Unhandled exeption in _recording_worker: {e}", exc_info=True)
                raise

        if self.use_extended_logging:
            logger.debug('Debug: Exiting _recording_worker method')

    def _is_silero_speech(self, chunk):
        """
        Returns true if speech is detected in the provided audio data

        Args:
            data (bytes): raw bytes of audio data (1024 raw bytes with
            16000 sample rate and 16 bits per sample)
        """
        if self.sample_rate != 16000:
            pcm_data = np.frombuffer(chunk, dtype=np.int16)
            data_16000 = signal.resample_poly(
                pcm_data, 16000, self.sample_rate)
            chunk = data_16000.astype(np.int16).tobytes()

        self.silero_working = True
        audio_chunk = np.frombuffer(chunk, dtype=np.int16)
        audio_chunk = audio_chunk.astype(np.float32) / INT16_MAX_ABS_VALUE
        vad_prob = self.silero_vad_model(
            torch.from_numpy(audio_chunk),
            SAMPLE_RATE).item()
        is_silero_speech_active = vad_prob > (1 - self.silero_sensitivity)
        if is_silero_speech_active:
            if not self.is_silero_speech_active and self.use_extended_logging:
                logger.info(f"{bcolors.OKGREEN}Silero VAD detected speech{bcolors.ENDC}")
        elif self.is_silero_speech_active and self.use_extended_logging:
            logger.info(f"{bcolors.WARNING}Silero VAD detected silence{bcolors.ENDC}")
        self.is_silero_speech_active = is_silero_speech_active
        self.silero_working = False
        return is_silero_speech_active

    def _is_webrtc_speech(self, chunk, all_frames_must_be_true=False):
        """
        Returns true if speech is detected in the provided audio data

        Args:
            data (bytes): raw bytes of audio data (1024 raw bytes with
            16000 sample rate and 16 bits per sample)
        """
        speech_str = f"{bcolors.OKGREEN}WebRTC VAD detected speech{bcolors.ENDC}"
        silence_str = f"{bcolors.WARNING}WebRTC VAD detected silence{bcolors.ENDC}"
        if self.sample_rate != 16000:
            pcm_data = np.frombuffer(chunk, dtype=np.int16)
            data_16000 = signal.resample_poly(
                pcm_data, 16000, self.sample_rate)
            chunk = data_16000.astype(np.int16).tobytes()

        # Number of audio frames per millisecond
        frame_length = int(16000 * 0.01)  # for 10ms frame
        num_frames = int(len(chunk) / (2 * frame_length))
        speech_frames = 0

        for i in range(num_frames):
            start_byte = i * frame_length * 2
            end_byte = start_byte + frame_length * 2
            frame = chunk[start_byte:end_byte]
            if self.webrtc_vad_model.is_speech(frame, 16000):
                speech_frames += 1
                if not all_frames_must_be_true:
                    if self.debug_mode:
                        logger.info(f"Speech detected in frame {i + 1}"
                              f" of {num_frames}")
                    if not self.is_webrtc_speech_active and self.use_extended_logging:
                        logger.info(speech_str)
                    self.is_webrtc_speech_active = True
                    return True
        if all_frames_must_be_true:
            if self.debug_mode and speech_frames == num_frames:
                logger.info(f"Speech detected in {speech_frames} of "
                      f"{num_frames} frames")
            elif self.debug_mode:
                logger.info(f"Speech not detected in all {num_frames} frames")
            speech_detected = speech_frames == num_frames
            if speech_detected and not self.is_webrtc_speech_active and self.use_extended_logging:
                logger.info(speech_str)
            elif not speech_detected and self.is_webrtc_speech_active and self.use_extended_logging:
                logger.info(silence_str)
            self.is_webrtc_speech_active = speech_detected
            return speech_detected
        else:
            if self.debug_mode:
                logger.info(f"Speech not detected in any of {num_frames} frames")
            if self.is_webrtc_speech_active and self.use_extended_logging:
                logger.info(silence_str)
            self.is_webrtc_speech_active = False
            return False

    def _check_voice_activity(self, data):
        """
        Initiate check if voice is active based on the provided data.

        Args:
            data: The audio data to be checked for voice activity.
        """
        self._is_webrtc_speech(data)

        # First quick performing check for voice activity using WebRTC
        if self.is_webrtc_speech_active:

            if not self.silero_working:
                self.silero_working = True

                # Run the intensive check in a separate thread
                threading.Thread(
                    target=self._is_silero_speech,
                    args=(data,)).start()

    def clear_audio_queue(self):
        """
        녹음기를 깨운 이후 등과 같은 상황에서,
        남아 있는 오디오 조각들이 처리되지 않도록
        audio_queue를 안전하게 비웁니다.
        """
        self.audio_buffer.clear()
        try:
            while True:
                self.audio_queue.get_nowait()
        except:
            # PyTorch's mp.Queue doesn't have a specific Empty exception
            # so we catch any exception that might occur when the queue is empty
            pass

    def _is_voice_active(self):
        """
        Determine if voice is active.

        Returns:
            bool: True if voice is active, False otherwise.
        """
        return self.is_webrtc_speech_active and self.is_silero_speech_active

    def _set_state(self, new_state):
        """
        Update the current state of the recorder and execute
        corresponding state-change callbacks.

        Args:
            new_state (str): The new state to set.

        """
        # Check if the state has actually changed
        if new_state == self.state:
            return

        # Store the current state for later comparison
        old_state = self.state

        # Update to the new state
        self.state = new_state

        # Log the state change
        logger.info(f"State changed from '{old_state}' to '{new_state}'")

        # Execute callbacks based on transitioning FROM a particular state
        if old_state == "listening":
            if self.on_vad_detect_stop:
                self.on_vad_detect_stop()
        elif old_state == "wakeword":
            if self.on_wakeword_detection_end:
                self.on_wakeword_detection_end()

        # Execute callbacks based on transitioning TO a particular state
        if new_state == "listening":
            if self.on_vad_detect_start:
                self.on_vad_detect_start()
            self._set_spinner("speak now")
            if self.spinner and self.halo:
                self.halo._interval = 250
        elif new_state == "wakeword":
            if self.on_wakeword_detection_start:
                self.on_wakeword_detection_start()
            self._set_spinner(f"say {self.wake_words}")
            if self.spinner and self.halo:
                self.halo._interval = 500
        elif new_state == "transcribing":
            self._set_spinner("transcribing")
            if self.spinner and self.halo:
                self.halo._interval = 50
        elif new_state == "recording":
            self._set_spinner("recording")
            if self.spinner and self.halo:
                self.halo._interval = 100
        elif new_state == "inactive":
            if self.spinner and self.halo:
                self.halo.stop()
                self.halo = None

    def _set_spinner(self, text):
        """
        Update the spinner's text or create a new
        spinner with the provided text.

        Args:
            text (str): The text to be displayed alongside the spinner.
        """
        if self.spinner:
            # If the Halo spinner doesn't exist, create and start it
            if self.halo is None:
                self.halo = halo.Halo(text=text)
                self.halo.start()
            # If the Halo spinner already exists, just update the text
            else:
                self.halo.text = text

    def _preprocess_output(self, text, preview=False):
        """
        Preprocesses the output text by removing any leading or trailing
        whitespace, converting all whitespace sequences to a single space
        character, and capitalizing the first character of the text.

        Args:
            text (str): The text to be preprocessed.

        Returns:
            str: The preprocessed text.
        """
        text = re.sub(r'\s+', ' ', text.strip())

        if self.ensure_sentence_starting_uppercase:
            if text:
                text = text[0].upper() + text[1:]

        # Ensure the text ends with a proper punctuation
        # if it ends with an alphanumeric character
        if not preview:
            if self.ensure_sentence_ends_with_period:
                if text and text[-1].isalnum():
                    text += '.'

        return text

    def _find_tail_match_in_text(self, text1, text2, length_of_match=10):
        """
        text1의 마지막 'n' 글자가 text2의 어떤 부분 문자열과 일치하는 위치를 찾는 메서드입니다.

        이 메서드는 두 개의 텍스트를 입력으로 받아, text1의 마지막 'n' 글자(n은 'length_of_match' 변수로 지정됨)를 추출한 뒤,
        이 부분 문자열이 text2 내에 존재하는지를 text2의 끝에서 시작하여 앞쪽으로 탐색합니다.

        매개변수:
        - text1 (str): text2에서 찾고자 하는 부분 문자열을 포함한 텍스트입니다.
        - text2 (str): text1의 부분 문자열을 찾고자 하는 대상 텍스트입니다.
        - length_of_match (int): 찾고자 하는 일치 문자열의 길이입니다.

        반환값:
        int: 일치하는 부분 문자열이 text2에서 시작하는 위치(0부터 시작하는 인덱스)를 반환합니다.
        일치하는 문자열이 없거나, 입력 텍스트의 길이가 너무 짧을 경우 -1을 반환합니다.
        """

        # Check if either of the texts is too short
        if len(text1) < length_of_match or len(text2) < length_of_match:
            return -1

        # The end portion of the first text that we want to compare
        target_substring = text1[-length_of_match:]

        # Loop through text2 from right to left
        for i in range(len(text2) - length_of_match + 1):
            # Extract the substring from text2
            # to compare with the target_substring
            current_substring = text2[len(text2) - i - length_of_match:
                                      len(text2) - i]

            # Compare the current_substring with the target_substring
            if current_substring == target_substring:
                # Position in text2 where the match starts
                return len(text2) - i

        return -1

    def __enter__(self):
        """
        컨텍스트 매니저 프로토콜을 설정하는 메서드입니다.

        이 메서드를 통해 해당 인스턴스는 `with` 문에서 사용할 수 있게 되며,
        자원 관리를 적절히 수행할 수 있도록 합니다. `with` 블록에 진입할 때
        자동으로 호출됩니다.

        반환값:
            self: 클래스의 현재 인스턴스.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        컨텍스트 매니저 프로토콜이 종료될 때의 동작을 정의하는 메서드입니다.

        이 메서드는 `with` 블록을 빠져나갈 때 호출되며, 시스템을 적절히 종료하거나
        자원을 해제하는 등의 필요한 정리 작업이 수행되도록 보장합니다.

        매개변수:
            exc_type (Exception 또는 None): 컨텍스트를 종료시킨 예외의 타입 (예외가 없으면 None).
            exc_value (Exception 또는 None): 예외 인스턴스 (예외가 없으면 None).
            traceback (Traceback 또는 None): 예외에 해당하는 traceback 객체 (예외가 없으면 None).
        """
        self.shutdown()