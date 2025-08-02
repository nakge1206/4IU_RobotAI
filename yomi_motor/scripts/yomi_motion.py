import rospy
from std_msgs.msg import String, Int16MultiArray
from yomi_motor import MotionSequenceExecutor

class MotionController:
    def __init__(self, vision = None):
        rospy.init_node('motion_controller_node', anonymous=True)
        self.executor = MotionSequenceExecutor()
        self.vision = vision

    def stand(self):
        motor_speeds = [3] * 10
        motor_positions = [180, 110, 180, 180, 180, 250, 180, 180, 190, 180]
        servo_angles = [90] * 8

        self.executor.motor_speed_pub.publish(Int16MultiArray(data=motor_speeds))
        self.executor.motor_position_pub.publish(Int16MultiArray(data=motor_positions))
        self.executor.servo_angle_pub.publish(Int16MultiArray(data=servo_angles))

    def ifConflicting(self):
        pass
    
    def I_joy1(self):
        """(좋아하는 것에 대해) 팔을 살랑살랑 움직인다."""
        # self.executor.request_all_data()
        self.executor.motor_publisher_batch(motor_ids=[6], motor_positions=[180], motor_speeds=[5])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[10], motor_positions=[180], motor_speeds=[5])
        rospy.sleep(0.2)
        self.stand()
        rospy.sleep(0.1)
        for i in range(3):
            self.executor.motor_publisher_batch(motor_ids=[3, 7], motor_positions=[160, 160], motor_speeds=[4, 4])
            rospy.sleep(1.5)
            self.executor.motor_publisher_batch(motor_ids=[3, 7], motor_positions=[200, 200], motor_speeds=[4, 4])
            rospy.sleep(1.5)


        


    def move_to_greeting_pose(self):
        # 5번 모터 위치로 팔을 올리고 서보로 손 흔들기
        self.executor.motor_publisher(motor_id=5, motor_position=320)
        self.executor.motor_publisher(servo_id=6, servo_angle=120)
    
    def bow(self):
        self.executor.motor_publisher(motor_id=7, motor_position=400)
        self.executor.motor_publisher(servo_id=4, servo_angle=45)

    def run(self):
        # 예시: 반복 동작
        rate = rospy.Rate(0.2)  # 5초마다
        while not rospy.is_shutdown():
            self.move_to_greeting_pose()
            rospy.sleep(2)
            self.bow()
            rospy.sleep(2)
            rate.sleep()

if __name__ == '__main__':
    try:
        controller = MotionController()
        controller.I_joy1()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass