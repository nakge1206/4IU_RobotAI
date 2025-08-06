import rospy
import math
import actionlib
import tf
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import OccupancyGrid


class ObstacleAvoider:
    def __init__(self):
        """
        노드 초기화 및 주요 변수 선언, 센서 구독, move_base 액션 서버 연결 등 초기 설정 수행
        """
        if __init__(self, init_node=False):
            rospy.init_node("go_to_obstacle_node")

        self.latest_scan = None   # 최신 LaserScan 데이터
        self.robot_pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}  # 오도메트리 기반 로봇 위치 정보
        self.is_moving = False    # 이동 중 여부
        if not rospy.core.is_initialized():
            rospy.init_node("go_to_obstacle_node")
        self.map_data = None    # 맵 데이터 (OccupancyGrid)
        rospy.Subscriber("/scan", LaserScan, self.scan_callback)      # 라이다 데이터 구독
        rospy.Subscriber("/odom", Odometry, self.odom_callback)       # 오도메트리 데이터 구독
        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)    # 맵 데이터 구독


        # move_base 액션 서버 연결
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("move_base 서버 연결 대기 중...")
        # self.move_base_client.wait_for_server()
        rospy.loginfo("move_base 서버 연결됨!")

        # TF listener 생성 (map ↔ base_link 변환 추적)
        self.tf_listener = tf.TransformListener()

        rospy.sleep(2.0)  # 초기 TF 및 센서 수신 대기
        rospy.Timer(rospy.Duration(5.0), self.check_and_move)  # 5초 주기로 장애물 확인 및 이동


    def map_callback(self, msg):
        self.map_data = msg


    def scan_callback(self, msg):
        """
        라이다 센서 콜백 함수 - 최신 scan 데이터를 저장
        """
        self.latest_scan = msg


    def odom_callback(self, msg):
        """
        오도메트리 콜백 함수 - 로봇의 현재 위치와 방향(yaw)을 저장
        """
        self.robot_pose["x"] = msg.pose.pose.position.x
        self.robot_pose["y"] = msg.pose.pose.position.y
        orientation_q = msg.pose.pose.orientation
        (_, _, yaw) = euler_from_quaternion([
            orientation_q.x, orientation_q.y,
            orientation_q.z, orientation_q.w
        ])
        self.robot_pose["yaw"] = yaw


    def get_robot_pose_in_map(self):
        """
        TF를 이용해 로봇의 map 좌표계 상 위치와 yaw 각도를 반환
        """
        try:
            self.tf_listener.waitForTransform("map", "base_link", rospy.Time(0), rospy.Duration(1.0))
            (trans, rot) = self.tf_listener.lookupTransform("map", "base_link", rospy.Time(0))
            (_, _, yaw) = euler_from_quaternion(rot)
            return {"x": trans[0], "y": trans[1], "yaw": yaw}
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return None


    # 벽 좌표 를 찾는 함수
    # 맵 데이터에서 가장 가까운 벽의 좌표를 반환
    def get_nearest_wall(self):
        if self.map_data is None:
            rospy.logwarn("맵 데이터 없음")
            return None

        robot_pose = self.get_robot_pose_in_map()
        if robot_pose is None:
            return None

        map_origin = self.map_data.info.origin.position
        resolution = self.map_data.info.resolution
        width = self.map_data.info.width
        height = self.map_data.info.height
        data = self.map_data.data

        # 로봇의 맵 상 위치를 픽셀 좌표로
        rx = int((robot_pose["x"] - map_origin.x) / resolution)
        ry = int((robot_pose["y"] - map_origin.y) / resolution)

        min_dist = float('inf')
        nearest = None

        for y in range(height):
            for x in range(width):
                index = y * width + x
                if data[index] == 100:  # 벽
                    dx = x - rx
                    dy = y - ry
                    dist = math.hypot(dx, dy)
                    if dist < min_dist:
                        min_dist = dist
                        nearest = (x, y)

        if nearest is None:
            return None

        wall_x = nearest[0] * resolution + map_origin.x
        wall_y = nearest[1] * resolution + map_origin.y
        return (wall_x, wall_y)


    def go_to_nearest_wall(self):
        """
        가장 가까운 벽 앞으로 이동 (맵 기반)
        """
        wall = self.get_nearest_wall()
        if wall is None:
            rospy.loginfo("벽 좌표를 찾을 수 없음")
            return

        wall_x, wall_y = wall
        rospy.loginfo(f"가장 가까운 벽 위치: x={wall_x:.2f}, y={wall_y:.2f}")

        # 벽 앞쪽으로 약간 떨어진 좌표 계산 (로봇 쪽으로 0.5m 뒤로 이동)
        target_x, target_y = self.get_relative_position(wall_x, wall_y, offset=0.5, direction="back")
        rospy.loginfo(f"벽 앞 이동 목표: x={target_x:.2f}, y={target_y:.2f}")

        self.move_to_goal(target_x, target_y)



    def get_obstacle_position(self, threshold=1.5):
        """
        가장 가까운 장애물의 절대 좌표(map 기준)를 반환
        - threshold 미만 거리의 장애물 중 가장 가까운 장애물을 선택
        """
        if self.latest_scan is None:
            return None

        pose_in_map = self.get_robot_pose_in_map()
        if pose_in_map is None:
            return None

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

        # 로봇 yaw + scan angle을 합쳐서 장애물의 방향을 구하고, map 좌표계로 변환
        total_angle = pose_in_map["yaw"] + best_angle
        obs_x = pose_in_map["x"] + math.cos(total_angle) * min_distance
        obs_y = pose_in_map["y"] + math.sin(total_angle) * min_distance
        return (obs_x, obs_y)


    def get_relative_position(self, obs_x, obs_y, offset=0.5, direction="left"):
        """
        장애물을 기준으로 특정 방향(offset 거리만큼) 떨어진 좌표를 계산
        - direction: "front", "back", "left", "right" 중 하나
        """
        dx = obs_x - self.robot_pose["x"]
        dy = obs_y - self.robot_pose["y"]
        dist = math.hypot(dx, dy)

        if dist == 0:
            return obs_x, obs_y

        # 단위 벡터
        ux, uy = dx / dist, dy / dist

        # 방향 벡터 설정
        if direction == "front":
            vx, vy = ux, uy
        elif direction == "back":
            vx, vy = -ux, -uy
        elif direction == "left":
            vx, vy = -uy, ux
        elif direction == "right":
            vx, vy = uy, -ux
        else:
            raise ValueError("Invalid direction")

        # 장애물 위치 기준으로 offset만큼 이동한 위치
        target_x = obs_x + vx * offset
        target_y = obs_y + vy * offset
        return target_x, target_y


    def move_to_goal(self, x, y):
        """
        MoveBase를 통해 지정한 x, y 위치로 이동 시도
        """
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation.w = 1.0  # 방향은 무시 (단순 위치 이동)

        self.is_moving = True
        self.move_base_client.send_goal(goal)
        self.move_base_client.wait_for_result()
        rospy.sleep(2.0)  # 안정성 확보
        self.is_moving = False

        result = self.move_base_client.get_result()
        if result:
            rospy.loginfo("[✓] 이동 성공")
        else:
            rospy.logwarn("[×] 이동 실패 또는 취소됨")


    def check_and_move(self, event):
        """
        주기적으로 실행되는 함수 (rospy.Timer 활용)
        - 현재 위치, 장애물 위치를 파악하고, 좌/우/전/후 방향으로 목표 설정 후 이동 시도
        """
        if self.is_moving:
            rospy.loginfo("이동 중. 이번 루프는 스킵.")
            return

        pose = self.get_robot_pose_in_map()
        if pose is None:
            rospy.logwarn("로봇 위치 확인 실패")
            return

        rospy.loginfo(f"[현재 위치] x: {pose['x']:.2f}, y: {pose['y']:.2f}, yaw: {math.degrees(pose['yaw']):.1f}°")

        obs = self.get_obstacle_position(threshold=1.5)
        if obs is None:
            rospy.loginfo("탐지된 장애물 없음")
            return

        obs_x, obs_y = obs
        rospy.loginfo(f"[장애물 좌표] x: {obs_x:.2f}, y: {obs_y:.2f}")

        direction = "left"  # 기본 회피 방향
        offset = 0.5        # 장애물로부터 떨어질 거리

        target_x, target_y = self.get_relative_position(obs_x, obs_y, offset, direction)
        rospy.loginfo(f"[→] 목표 좌표: ({target_x:.2f}, {target_y:.2f})")

        self.move_to_goal(target_x, target_y)


if __name__ == "__main__":
    try:
        ObstacleAvoider()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
