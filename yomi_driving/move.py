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
move_base_client = None  # MoveBase 액션 클라이언트

# ─────────────────────────────
# [1] LaserScan 콜백
# ─────────────────────────────
def scan_callback(msg):
    global latest_scan
    latest_scan = msg

# ─────────────────────────────
# [2] Odometry 콜백 → 로봇 위치 저장
# ─────────────────────────────
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

# ─────────────────────────────
# [3] 가장 가까운 장애물의 전역 위치 계산
# ─────────────────────────────
def get_obstacle_position(threshold=1.5):
    if latest_scan is None:
        return None

    ranges = latest_scan.ranges
    angle_min = latest_scan.angle_min
    angle_increment = latest_scan.angle_increment

    min_distance = float('inf')
    best_angle = None

    for i, r in enumerate(ranges):
        if math.isinf(r) or r < 0.1 or r >= threshold:
            continue
        angle = angle_min + i * angle_increment
        if r < min_distance:
            min_distance = r
            best_angle = angle

    if best_angle is None:
        return None

    total_angle = robot_pose["yaw"] + best_angle
    obs_x = robot_pose["x"] + math.cos(total_angle) * min_distance
    obs_y = robot_pose["y"] + math.sin(total_angle) * min_distance
    return (obs_x, obs_y)

# ─────────────────────────────
# [4] 장애물 기준 상대 방향 위치 계산 (벡터 방식)
# ─────────────────────────────
def get_relative_position(obs_x, obs_y, offset=0.5, direction="left"):
    """
    장애물 기준 상대 위치 계산: 좌/우/뒤/앞 방향
    direction: "left", "right", "back", "front"
    """
    dx = obs_x - robot_pose["x"]
    dy = obs_y - robot_pose["y"]
    dist = math.hypot(dx, dy)

    if dist == 0:
        rospy.logwarn("로봇과 장애물 위치가 같습니다.")
        return obs_x, obs_y

    # 로봇 → 장애물 방향 단위 벡터
    ux = dx / dist
    uy = dy / dist

    # 방향 벡터 선택
    if direction == "front":
        vx, vy = ux, uy
    elif direction == "back":
        vx, vy = -ux, -uy
    elif direction == "left":
        vx, vy = -uy, ux
    elif direction == "right":
        vx, vy = uy, -ux
    else:
        raise ValueError("direction must be one of: front, back, left, right")

    target_x = obs_x + vx * offset
    target_y = obs_y + vy * offset
    return target_x, target_y

# ─────────────────────────────
# [5] 목표 위치로 이동
# ─────────────────────────────
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

# ─────────────────────────────
# [6] 주기적 실행: 장애물 탐지 → 상대 위치로 이동
# ─────────────────────────────
def check_and_move(event):
    global is_moving
    if is_moving:
        rospy.loginfo("이동 중. 이번 루프는 스킵.")
        return

    obs = get_obstacle_position(threshold=1.5)
    if obs is None:
        rospy.loginfo("장애물 없음")
        return

    obs_x, obs_y = obs
    rospy.loginfo(f"[i] 장애물 좌표: ({obs_x:.2f}, {obs_y:.2f})")

    # 이동 방향 선택: "left", "right", "back", "front"
    direction = "left"
    offset = 0.5  # 장애물과 떨어질 거리

    target_x, target_y = get_relative_position(obs_x, obs_y, offset=offset, direction=direction)
    rospy.loginfo(f"목표 좌표: ({target_x:.2f}, {target_y:.2f}) 방향: {direction}")
    move_to_goal(target_x, target_y)

# ─────────────────────────────
# [7] 메인 실행
# ─────────────────────────────
def main():
    global move_base_client
    rospy.init_node("go_to_obstacle_node")
    rospy.Subscriber("/scan", LaserScan, scan_callback)
    rospy.Subscriber("/odom", Odometry, odom_callback)

    move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    rospy.loginfo("move_base 서버 연결 대기 중...")
    move_base_client.wait_for_server()
    rospy.loginfo("move_base 서버 연결됨!")

    rospy.sleep(2.0)  # 초기 데이터 수신 대기
    rospy.Timer(rospy.Duration(5.0), check_and_move)
    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
