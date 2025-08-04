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

# 1. LaserScan 콜백
def scan_callback(msg):
    global latest_scan
    latest_scan = msg

# 2. 가장 가까운 장애물의 각도(도)와 거리(m)를 반환
def get_obstacle_angle(threshold=1.0):
    # param threshold: 거리 제한 (m)
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
    else:
        return None

# 3. Odometry 콜백
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

# 4. 현재 위치 기준 목표 좌표 계산
def calculate_target_position(angle_deg, distance):
    # angle_deg: 장애물 방향 (도)
    # distance: 이동 거리 (m)
    # return: (target_x, target_y)
    total_angle_rad = robot_pose["yaw"] + math.radians(angle_deg)
    target_x = robot_pose["x"] + math.cos(total_angle_rad) * distance
    target_y = robot_pose["y"] + math.sin(total_angle_rad) * distance
    return (target_x, target_y)

# 5. 목표 위치로 이동
def move_to_goal(x, y):
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    rospy.loginfo("move_base 서버 대기 중")
    client.wait_for_server()

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y
    goal.target_pose.pose.orientation.w = 1.0

    rospy.loginfo(f"[→] 목표 좌표: ({x:.2f}, {y:.2f})")
    client.send_goal(goal)
    client.wait_for_result()

    result = client.get_result()
    if result:
        rospy.loginfo("이동 성공")
    else:
        rospy.logwarn("이동 실패 또는 취소됨")

# 6. 전체 제어 흐름
def main():
    rospy.init_node("go_to_obstacle_node")
    rospy.Subscriber("/scan", LaserScan, scan_callback)
    rospy.Subscriber("/odom", Odometry, odom_callback)

    rospy.sleep(2.0)  # 초기 센서 수신 대기

    result = get_obstacle_angle(threshold=1.5)
    if result is None:
        rospy.logwarn("감지된 장애물이 없습니다.")
        return

    angle_deg, distance = result
    rospy.loginfo(f"[i] 장애물 방향: {angle_deg:.1f}도, 거리: {distance:.2f}m")

    # 최소 안전 거리 확보
    approach_distance = max(0.2, distance - 0.3)
    target_x, target_y = calculate_target_position(angle_deg, approach_distance)
    move_to_goal(target_x, target_y)

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
