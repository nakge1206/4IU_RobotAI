import rospy
import math
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib

# 전역 변수
latest_scan = None
robot_pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
is_moving = False
move_base_client = None  # 🔧 전역 클라이언트 객체

# LaserScan 콜백
def scan_callback(msg):
    global latest_scan
    latest_scan = msg

# 장애물 탐지
def get_obstacle_angle(threshold=1.0):
    if latest_scan is None:
        return None

    ranges = latest_scan.ranges
    angle_min = latest_scan.angle_min
    angle_increment = latest_scan.angle_increment

    min_distance = float('inf')
    obstacle_angle = None

    for i, r in enumerate(ranges):
        if math.isinf(r) or r < 0.1 or r >= threshold:
            continue
        angle = angle_min + i * angle_increment
        if r < min_distance:
            min_distance = r
            obstacle_angle = math.degrees(angle)

    if obstacle_angle is not None:
        return (obstacle_angle, min_distance)
    return None

# Odometry 콜백
def odom_callback(msg):
    global robot_pose
    robot_pose["x"] = msg.pose.pose.position.x
    robot_pose["y"] = msg.pose.pose.position.y
    orientation_q = msg.pose.pose.orientation
    (_, _, yaw) = euler_from_quaternion([
        orientation_q.x, orientation_q.y,
        orientation_q.z, orientation_q.w
    ])
    robot_pose["yaw"] = yaw

# 목표 좌표 계산
def calculate_target_position(angle_deg, distance):
    total_angle_rad = robot_pose["yaw"] + math.radians(angle_deg)
    target_x = robot_pose["x"] + math.cos(total_angle_rad) * distance
    target_y = robot_pose["y"] + math.sin(total_angle_rad) * distance
    return (target_x, target_y)

# 목표 위치로 이동
def move_to_goal(x, y):
    global is_moving, move_base_client

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y
    goal.target_pose.pose.orientation.w = 1.0

    rospy.loginfo(f"[→] 목표 좌표: ({x:.2f}, {y:.2f})")
    is_moving = True
    move_base_client.send_goal(goal)
    move_base_client.wait_for_result()
    is_moving = False

    result = move_base_client.get_result()
    if result:
        rospy.loginfo("이동 성공")
    else:
        rospy.logwarn("이동 실패 또는 취소됨")

# 5초마다 실행될 함수
def check_and_move(event):
    global is_moving
    if is_moving:
        rospy.loginfo("현재 이동 중입니다. 스킵합니다.")
        return

    result = get_obstacle_angle(threshold=1.5)
    if result is None:
        rospy.loginfo("[⏳] 감지된 장애물이 없습니다.")
        return

    angle_deg, distance = result
    rospy.loginfo(f"[i] 장애물 방향: {angle_deg:.1f}도, 거리: {distance:.2f}m")

    approach_distance = max(0.2, distance - 0.3)
    target_x, target_y = calculate_target_position(angle_deg, approach_distance)
    move_to_goal(target_x, target_y)

# 메인 함수
def main():
    global move_base_client
    rospy.init_node("go_to_obstacle_node")
    rospy.Subscriber("/scan", LaserScan, scan_callback)
    rospy.Subscriber("/odom", Odometry, odom_callback)

    # move_base 클라이언트 초기화
    move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    rospy.loginfo("move_base 서버 대기 중...")
    move_base_client.wait_for_server()
    rospy.loginfo("move_base 연결 완료.")

    rospy.sleep(2.0)  # 초기 센서 수신 대기
    rospy.Timer(rospy.Duration(5.0), check_and_move)
    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
