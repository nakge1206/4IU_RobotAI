import rospy
import math
import actionlib
import tf
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

class ObstacleAvoider:
    def __init__(self):
        # 센서 데이터 및 상태 변수 초기화
        self.latest_scan = None
        self.robot_pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        self.is_moving = False

        # 노드 초기화 및 토픽 구독 설정
        rospy.init_node("go_to_obstacle_node")
        rospy.Subscriber("/scan", LaserScan, self.scan_callback)    # 라이다 스캔
        rospy.Subscriber("/odom", Odometry, self.odom_callback)     # 오도메트리 정보

        # move_base 액션 서버 클라이언트 생성 및 연결
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("move_base 서버 연결 대기 중...")
        self.move_base_client.wait_for_server()
        rospy.loginfo("move_base 서버 연결됨!")

        # TF 변환 리스너 생성 (map → base_link 변환용)
        self.tf_listener = tf.TransformListener()

        # 데이터 수신 대기 후 주기적 장애물 체크 시작 (5초마다)
        rospy.sleep(2.0)
        rospy.Timer(rospy.Duration(5.0), self.check_and_move)

    # [1] LaserScan 콜백: 최신 라이다 데이터 저장
    def scan_callback(self, msg):
        self.latest_scan = msg


    # [2] Odometry 콜백: 로봇의 현재 위치 및 방향(yaw) 저장 (odom 기준)
    def odom_callback(self, msg):
        self.robot_pose["x"] = msg.pose.pose.position.x
        self.robot_pose["y"] = msg.pose.pose.position.y
        orientation_q = msg.pose.pose.orientation
        (_, _, yaw) = euler_from_quaternion([
            orientation_q.x, orientation_q.y,
            orientation_q.z, orientation_q.w
        ])
        self.robot_pose["yaw"] = yaw


    # [3] 현재 로봇의 위치를 map 좌표계 기준으로 반환 (RViz와 일치)
    def get_robot_pose_in_map(self):
        try:
            self.tf_listener.waitForTransform("map", "base_link", rospy.Time(0), rospy.Duration(1.0))
            (trans, rot) = self.tf_listener.lookupTransform("map", "base_link", rospy.Time(0))
            (_, _, yaw) = euler_from_quaternion(rot)
            return {"x": trans[0], "y": trans[1], "yaw": yaw}
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("TF 변환 실패: map → base_link")
            return None
        

    # [4] 가장 가까운 장애물의 map 기준 좌표 계산
    def get_obstacle_position(self, threshold=1.5):
        if self.latest_scan is None:
            return None

        # 현재 로봇 위치 (map 기준)
        pose_in_map = self.get_robot_pose_in_map()
        if pose_in_map is None:
            return None

        # 라이다 데이터에서 가장 가까운 유효 장애물 찾기
        ranges = self.latest_scan.ranges
        angle_min = self.latest_scan.angle_min
        angle_increment = self.latest_scan.angle_increment

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

        # 장애물의 map 기준 위치 계산
        total_angle = pose_in_map["yaw"] + best_angle
        obs_x = pose_in_map["x"] + math.cos(total_angle) * min_distance
        obs_y = pose_in_map["y"] + math.sin(total_angle) * min_distance
        return (obs_x, obs_y)
    

    # [5] 장애물 기준 특정 방향(offset 거리만큼 떨어진) 위치 계산
    def get_relative_position(self, obs_x, obs_y, offset=0.5, direction="left"):
        dx = obs_x - self.robot_pose["x"]
        dy = obs_y - self.robot_pose["y"]
        dist = math.hypot(dx, dy)

        if dist == 0:
            rospy.logwarn("로봇과 장애물 위치가 같습니다.")
            return obs_x, obs_y

        # 단위 방향 벡터 계산
        ux = dx / dist
        uy = dy / dist

        # 원하는 방향에 따른 벡터 설정
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

        # offset 거리만큼 떨어진 목표 위치 반환
        target_x = obs_x + vx * offset
        target_y = obs_y + vy * offset
        return target_x, target_y
    

    # [6] move_base를 통해 특정 위치로 이동 명령
    def move_to_goal(self, x, y):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"  # 꼭 map 기준이어야 함
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation.w = 1.0  # 방향 고정 (회전 없음)

        rospy.loginfo(f"[→] 목표 좌표: ({x:.2f}, {y:.2f})")
        self.is_moving = True
        self.move_base_client.send_goal(goal)
        self.move_base_client.wait_for_result()
        self.is_moving = False

        result = self.move_base_client.get_result()
        if result:
            rospy.loginfo("이동 성공")
        else:
            rospy.logwarn("이동 실패 또는 취소됨")


    # [7] 주기적으로 호출됨: 가장 가까운 장애물을 찾고, 지정한 방향으로 회피 이동 시도
    def check_and_move(self, event):
        if self.is_moving:
            rospy.loginfo("이동 중. 이번 루프는 스킵.")
            return

        obs = self.get_obstacle_position(threshold=1.5)
        if obs is None:
            rospy.loginfo("장애물 없음")
            return

        obs_x, obs_y = obs
        rospy.loginfo(f"[i] 장애물 좌표: ({obs_x:.2f}, {obs_y:.2f})")

        # 회피 방향 설정 및 목표 위치 계산
        direction = "left"
        offset = 0.5  # 장애물과의 거리
        target_x, target_y = self.get_relative_position(obs_x, obs_y, offset=offset, direction=direction)

        rospy.loginfo(f"목표 좌표: ({target_x:.2f}, {target_y:.2f}) 방향: {direction}")
        self.move_to_goal(target_x, target_y)


# [8] 실행부: ObstacleAvoider 클래스 실행 및 ROS 루프 유지
if __name__ == "__main__":
    try:
        ObstacleAvoider()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
