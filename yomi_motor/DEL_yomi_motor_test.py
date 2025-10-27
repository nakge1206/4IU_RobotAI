#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import String, Int16MultiArray
import json, os, time, threading, sys

class MotionSequenceExecutor:
    def __init__(self):
        if not rospy.core.is_initialized():
            rospy.init_node('yomi_motor', anonymous=True)

        # --- Publishers ---
        self.motor_speed_pub = rospy.Publisher('/motor_speed_cmd', Int16MultiArray, queue_size=10)
        self.motor_position_pub = rospy.Publisher('/motor_position_cmd', Int16MultiArray, queue_size=10)
        self.servo_angle_pub = rospy.Publisher('/servo_angle_cmd', Int16MultiArray, queue_size=8)

        # Feedback
        rospy.Subscriber('/motor_speed_fb', Int16MultiArray, self.motor_callback, callback_args=1)
        rospy.Subscriber('/motor_position_fb', Int16MultiArray, self.motor_callback, callback_args=2)
        rospy.Subscriber('/servo_angle_fb', Int16MultiArray, self.motor_callback, callback_args=3)

        # 재생 요청 토픽(외부에서 쓸 수도 있으니 유지)
        rospy.Subscriber('/play_motion_sequence', String, self.handle_sequence_request)

        # 상태 배열
        self.motor_limits = {3:(120,300),4:(110,260),5:(90,270),6:(95,180),7:(60,240),8:(100,250),9:(90,270),10:(180,265),11:(160,240),12:(110,250)}
        self.curMotorSpeed = [2]*10
        self.curMotorPos   = [180,110,180,180,180,250,180,180,190,180]
        self.servo_valid_ids = [1,2,3,4,6,7,8,9]
        self.curServoAngle = [90]*len(self.servo_valid_ids)

        # 고정 모션 디렉토리
        self.motion_dir = "/home/micca/catkin_ws/src/4IU_RobotAI/yomi_motor/motion"

        rospy.loginfo("✅ Node up (interactive CLI enabled).")
        rospy.loginfo(f"   • motion_dir: {self.motion_dir}")
        rospy.loginfo("   • 터미널에 파일명 또는 경로를 입력하세요 (예: joy2). 종료: exit/quit/q")

        # === 입력 스레드 시작 ===
        self._stop_flag = threading.Event()
        self._cli_thread = threading.Thread(target=self._interactive_loop, daemon=True)
        self._cli_thread.start()

    # ---------- 입력 스레드 ----------
    def _interactive_loop(self):
        while not rospy.is_shutdown() and not self._stop_flag.is_set():
            try:
                # 안내 프롬프트
                sys.stdout.write("> ")
                sys.stdout.flush()
                line = sys.stdin.readline()  # input() 대신 readline() 사용 (Ctrl+D 대응)
                if not line:
                    # EOF (Ctrl+D 등)
                    break
                key = line.strip()
                if not key:
                    continue
                if key.lower() in ("exit", "quit", "q"):
                    rospy.loginfo("👋 종료 명령을 받았습니다. 노드를 종료합니다.")
                    rospy.signal_shutdown("User requested exit")
                    break

                # 내부 핸들러 재사용
                self.handle_sequence_request(String(data=key))

            except Exception as e:
                rospy.logerr(f"CLI 입력 처리 중 오류: {e}")
                time.sleep(0.2)

    # ---------- 콜백 ----------
    def motor_callback(self, msg, index):
        if index == 1:
            self.curMotorSpeed = list(msg.data)
        elif index == 2:
            self.curMotorPos = list(msg.data)
        elif index == 3:
            self.curServoAngle = list(msg.data)

    # ---------- 재생 요청 처리 ----------
    def handle_sequence_request(self, msg: String):
        key = msg.data.strip()
        if not key:
            rospy.logwarn("⚠️ 빈 요청입니다.")
            return

        if key.endswith('.json') or '/' in key:
            file_path = key
        else:
            file_path = os.path.join(self.motion_dir, f"{key}.json")

        file_path = os.path.expanduser(file_path)

        if not os.path.isfile(file_path):
            rospy.logwarn(f"⚠️ JSON 파일을 찾을 수 없습니다: {file_path}")
            return

        rospy.loginfo(f"▶ Executing motion sequence from: {file_path}")
        self.execute_sequence(file_path)

    # ---------- JSON 실행 ----------
    def execute_sequence(self, json_path: str):
        try:
            with open(json_path, "r") as f:
                motion_list = json.load(f)
        except Exception as e:
            rospy.logerr(f"❌ JSON 로드 실패: {e}")
            return

        if not isinstance(motion_list, list):
            rospy.logerr("❌ JSON 최상위 구조가 리스트가 아닙니다.")
            return

        start_time = time.time()
        for idx, motion in enumerate(motion_list):
            try:
                ts_ms = int(motion.get("timestamp_ms", 0))
                target_sec = ts_ms / 1000.0
                while (time.time() - start_time) < target_sec and not rospy.is_shutdown():
                    time.sleep(0.001)

                speed_out = self.curMotorSpeed.copy()
                pos_out   = self.curMotorPos.copy()
                servo_out = self.curServoAngle.copy()

                for k, v in motion.get("motor_speeds", {}).items():
                    mid = int(k); i = mid - 3
                    if 0 <= i < 10: speed_out[i] = int(v)
                for k, v in motion.get("motor_positions", {}).items():
                    mid = int(k); i = mid - 3
                    if 0 <= i < 10:
                        tgt = int(v)
                        if mid in self.motor_limits:
                            lo, hi = self.motor_limits[mid]
                            tgt = max(lo, min(hi, tgt))
                        pos_out[i] = tgt
                for k, v in motion.get("servo_angles", {}).items():
                    sid = int(k)
                    if sid in self.servo_valid_ids:
                        sidx = self.servo_valid_ids.index(sid)
                        servo_out[sidx] = int(v)

                self.motor_speed_pub.publish(Int16MultiArray(data=speed_out))
                self.motor_position_pub.publish(Int16MultiArray(data=pos_out))
                self.servo_angle_pub.publish(Int16MultiArray(data=servo_out))

                self.curMotorSpeed = speed_out
                self.curMotorPos   = pos_out
                self.curServoAngle = servo_out

                rospy.loginfo(f"✅ Executed motion[{idx}] at {ts_ms} ms")
            except Exception as e:
                rospy.logerr(f"❌ motion[{idx}] 실행 중 오류: {e}")

def main():
    node = MotionSequenceExecutor()
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
