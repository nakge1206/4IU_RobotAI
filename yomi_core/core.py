import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP 중복 방지 설정
import threading
import random
import time
#ros통신용
import rospy
from std_msgs.msg import String, Bool


# 모듈 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'stt'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'llm_core'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'tts'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'vision_face'))
sys.path.append(os.path.abspath('./yomi_motor/scripts'))


# 각 모듈 임포트
from stt.realtime_stt_module import STTModule #STT
# from robot_core.inference_koalpaca_12B import LLMResponder
from tts.TTS_server import TTSServer, VITS, TTSClient
from vision_face.VisionFace import VisonFaceMain #Vision
# from llm_core.gpt_fine_tuning_model import FineTunedGPTClient #GPT LLM
# from yomi_motor import MotionSequenceExecutor


class Yomi:
    def __init__(self, isSTT=True, isTTS=True, isLLM=True, isVisionFace=True):
        self.results = [] #
        self.is_tts_running = False
        self.sttEnable = True
        self.lastVision = None

        self.isSTT = isSTT
        self.isTTS = isTTS
        self.isLLM = isLLM
        self.isVisionFace = isVisionFace
        
        #조이스틱 제어 플래그
        self.joy_master_flag = True

        #stt timeout
        self.stt_timeout = 20
        self.sttTimer = None
        self.visionEnable = False

        #모듈 초기화
        self.stt = STTModule(on_text_callback=self.handle_stt) if isSTT else None
        print(f"STT : 준비완료 ({isSTT})")
        if isTTS:
            self.vits_model = VITS()
            self.tts_server = TTSServer(self.vits_model)
            print(f"TTSServer : 준비완료 ({isTTS})")
        self.tts = TTSClient(on_done=self.on_tts_done, on_start=self.on_tts_start) if isTTS else None
        print(f"TTSClient : 준비완료 ({isTTS})")
        # self.llm = FineTunedGPTClient() if isLLM else None
        print(f"LLM : 준비완료 ({isLLM})")
        self.VisionFace = VisonFaceMain(interval=2, on_vision_callback=self.handle_vision, viewGUI=True) if isVisionFace else None
        print(f"VisionFace : 준비완료 ({isVisionFace})")
        
        #ROS Publisher
        if not rospy.core.is_initialized():
            rospy.init_node('yomi_core', anonymous=True)
        # self.llm_emotion = rospy.Publisher('/llm_emotion', String, queue_size=10)
        self.tts_state = rospy.Publisher('/tts_state', Bool, queue_size=10)
        print("ROS Publisher: 토픽(감정, tts실행여부) 연결완료")
        #ROS Subscriber
        rospy.Subscriber('/switch_1_state', Bool, self.handle_switch, callback_args=1)
        rospy.Subscriber('/switch_2_state', Bool, self.handle_switch, callback_args=2)
        rospy.Subscriber('/switch_3_state', Bool, self.handle_switch, callback_args=3)
        rospy.Subscriber('/switch_4_state', Bool, self.handle_switch, callback_args=4)
        rospy.Subscriber('/switch_5_state', Bool, self.handle_switch, callback_args=5)
        rospy.Subscriber('/switch_6_state', Bool, self.handle_switch, callback_args=6)
        print("ROS Subscriber: 토픽(switch, ) 연결완료")
        

    def start(self):
        """모듈 시작"""
        if self.isSTT:
            threading.Thread(target=self._sttStart, daemon=True).start()
        if self.isTTS:
            threading.Thread(target=self.tts_server.run, daemon=True).start()
        if self.isVisionFace:
            self.VisionFace.run()

    def stop(self):
        """모듈 종료"""
        if self.isSTT: 
            self.stt.stop()
        if self.isTTS: 
            pass
            # self.tts.stop()
        if self.isVisionFace: 
            self.VisionFace.stop()
        print("모든 모듈 종료")

    def _sttStart(self):
        """STT모듈 멀티스레드로 시작 후, 1초마다 실행여부 판단"""
        self.stt.start()
        while True:
            time.sleep(1)
            self.resume()

    def pause(self):
        """"STT 마이크 비활성화"""
        if self.isSTT:
            self.stt.micOff()

    def resume(self):
        """STT 마이크 활성화"""
        if self.joy_master_flag and self.isSTT and not self.is_tts_running and self.sttEnable:
            self.stt.micOn()
    
    def handle_stt(self, stt_texts):
        """STT발생 시 처리 함수"""
        self.sttEnable = False
        self.pause()
        stt_text, stt_info = stt_texts

        #디버깅구문 - LLM안쓰고 하울링구현
        if self.tts:
            print("[handle_stt] 직접 TTS 전송 호출")
            self.try_send_tts(stt_text)
        else:
            print("[handle_stt] self.tts가 None입니다.")
        
        #stt_timeout이 경과하면, handle_vision함수 실행
        if self.sttTimer:
            self.sttTimer.cancel()
        self.sttTimer = threading.Timer(self.stt_timeout, self.handle_vision)
        self.sttTimer.start()

        try:
            print(f"\n STT 결과: {stt_texts}")                
            labels = [item['label'] for item in self.lastVision] if self.lastVision else None
            # self.llm_promt(stt_texts, labels)
        except Exception as e:
            print(f"LLM 처리 중 오류 발생 : {e}")
            

    def handle_vision(self, visionText=None):
        """STT N초 이상 안들어오면 vision 정보 활용"""
        self.lastVision = visionText
        if self.joy_master_flag and self.isVisionFace and self.visionEnable:
            # labels = [item['label'] for item in self.lastVision] if self.lastVision else None
            print(self.lastVision)
            # if random.random() < 0.1:
            #     self.llm_promt(None, labels, False, True)
            pass

    def llm_promt(self, sttTexts, visionText):
        if not self.isLLM:
            self.is_tts_running = False
            self.resume()
            if self.tts: #근데 tts는 켜져있을 때
                stt_text, _ = sttTexts
                print("[llm_promt] LLM OFF 상태, TTS 전송 시도")
                self.try_send_tts(stt_text)
            return

        if self.isSTT and self.isVision: #STT가 들어올때(영상정보까지 포함됨).
            # gsq 모델
            # response = self.llm.generate_response(
            #         stt_text,
            #         emotion=emotion,
            #         event=event,
            #         mbti="INFP"
            #     )

            # gpt 모델
            stt_text, metadata = sttTexts
            stt_emotion = metadata.get("stt_emotion", "")
            event = metadata.get("event", "")
            user_prompt = self.llm.build_instruction(stt_text, stt_emotion, event, visionText)

            emotion, response = self.llm.chat(user_prompt)
            print(f" LLM_emotion 결과: {emotion}")
            print(f" LLM 결과: {response}")
            # self.llm_emotion.publish(emotion)

            if self.isTTS:
                print("LLM->TTS")
                # self.tts.send_text(response)
                # self.llm_emotion.publish(emotion)
                # if not self.is_tts_running:
                #     self.is_tts_running = True  # 재생 중 플래그
                #     print("LLM->TTS")
                #     self.tts.send_text(response)
                # else:
                #     print("TTS가 아직 끝나지 않았습니다. 새 요청 무시.")
            return
        
        if not self.isSTT and self.isVision:
            print("YOLO->LLM, 시각정보 : ", visionText)
            user_prompt = self.llm.build_instruction_vision(visionText)

            emotion, response = self.llm.chat(user_prompt)
            print(f" LLM 결과: {emotion}")
            print(f" LLM 결과: {response}")

            if self.tts:
                if not self.is_tts_running:
                    self.is_tts_running = True
                    print("LLM->TTS")
                    # self.tts.send_text(response)

        else: #llm 껐을때
            if self.tts: #근데 tts는 켜져있을 때
                pass
                # self.tts.send_text(stt_text)

    
    def test_llm(self):
        if self.isLLM:
            # gsq 모델
            # response = self.llm.generate_response(
            #         stt_text,
            #         emotion=emotion,
            #         event=event,
            #         mbti="INFP"
            #     )
            
            # gpt 모델
            stt_text = "안녕. 밥 먹었어?"
            emotion = "행복"
            event = "말하는 중"
            visionText = None

            user_prompt = self.llm.build_instruction(stt_text, emotion, event)
            response = self.llm.generate_response(
                stt_text,
                emotion=emotion,
                event=event,
                mbti="INFP"
            )

            print(f" LLM 결과: {response}")
        else: #llm 껐을때
            pass

    def handle_switch(self, msg, index): 
        """ROS토픽 이용 Switch 처리함수"""
        state = True if msg.data else False
        if state:
            touch_map = {
                0: "등",
                1: "왼팔",
                2: "오른팔",
                3: "왼손",
                4: "오른손",
                5: "머리"
            }

            if index in touch_map:
                print(f"[Touch_INFO] 눌린 부위: {touch_map[index]} (switch={index})")
            else:
                print(f"[Touch_WARN] 알 수 없는 switch 값: {index}")


    def send_tts(self, text):
        """TTS 서버에 텍스트를 전송"""
        if self.tts:
            self.tts.send_text(text)        

    def start_tts_publish(self):
        """TTS가 시작되면 tts_state topic에 True값 전송"""
        self.tts_state.publish(True)

    def done_tts_publish(self):
        """TTS가 끝나면 tts_state topic에 False값 전송"""
        self.tts_state.publish(False)

    def try_send_tts(self, response_text):
        print(f"[try_send_tts 호출됨] text: {response_text}")
        """TTS 상태 확인 후 텍스트 전송"""
        if self.tts and not self.is_tts_running:
            self.is_tts_running = True
            self.start_tts_publish()
            self.send_tts(response_text)
        else:
            print("TTS가 아직 끝나지 않았습니다. 새 요청 무시.")

    def on_tts_start(self):
        """TTS가 시작될 때 호출되는 콜백 함수"""
        self.start_tts_publish()
        self.is_tts_running = True

    def on_tts_done(self):
        """TTS가 끝날 때 호출되는 콜백 함수"""
        self.done_tts_publish()
        self.is_tts_running = False
        self.sttEnable = True


if __name__ == "__main__":
    service = Yomi(isSTT=True, isLLM=False, isTTS=True, isVisionFace=False)

    service.start()

    try:
        while not rospy.is_shutdown():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n 종료 중...")
    finally:
        service.stop()