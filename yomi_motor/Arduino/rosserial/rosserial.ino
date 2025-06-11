#include <ros.h>
#include <std_msgs/Int16MultiArray.h>
#include <Servo.h>

#define tx_enable 12
#define MAX485_TX digitalWrite(tx_enable, HIGH); delayMicroseconds(10);
#define MAX485_RX digitalWrite(tx_enable, LOW); delayMicroseconds(10);

// 서보 모터 객체 생성
Servo servos[8];  
int servo_pins[8] = {2, 3, 4, 5, 
                     7, 8, 9, 10};

// DC 모터 속도 및 위치 배열
int velocity_set1[10] = { 1, 1, 1, 1, 
                          1, 1, 1, 1, 
                          1, 1 };
int position_set1[10] = { 180, 180, 180, 180, 
                          180, 180, 180, 180, 
                          180, 180 };
int servo_angles[8] = { 90, 90, 90, 90, 
                        90, 90, 90, 90 };  // 초기 서보 각도

// ROS 노드 생성
ros::NodeHandle nh;

// DC 모터 속도 수신 콜백
void speedCallback(const std_msgs::Int16MultiArray &msg) {
    if (msg.data_length != 10) return;

    for (int i = 0; i < 10; i++) {
        velocity_set1[i] = msg.data[i];
    }
    publishMotorSpeed();
}

ros::Subscriber<std_msgs::Int16MultiArray> speed_sub("motor_speed_cmd", speedCallback);

// DC 모터 위치 수신 콜백
void positionCallback(const std_msgs::Int16MultiArray &msg) {
    if (msg.data_length != 10) return;

    for (int i = 0; i < 10; i++) {
        position_set1[i] = msg.data[i];
    }
    publishMotorPosition();
}

ros::Subscriber<std_msgs::Int16MultiArray> position_sub("motor_position_cmd", positionCallback);

// 서보 모터 각도 수신 콜백
void servoCallback(const std_msgs::Int16MultiArray &msg) {
    if (msg.data_length != 8) return;

    for (int i = 0; i < 8; i++) {
        servo_angles[i] = msg.data[i];
        servos[i].write(servo_angles[i]);
    }
    publishServoAngles();
}

ros::Subscriber<std_msgs::Int16MultiArray> servo_sub("servo_angle_cmd", servoCallback);

// 모터 속도 요청 콜백
void requestSpeedCallback(const std_msgs::Int16MultiArray &msg) {
    publishMotorSpeed();
}

ros::Subscriber<std_msgs::Int16MultiArray> speed_request_sub("motor_speed_request", requestSpeedCallback);

// 모터 위치 요청 콜백
void requestPositionCallback(const std_msgs::Int16MultiArray &msg) {
    publishMotorPosition();
}

ros::Subscriber<std_msgs::Int16MultiArray> position_request_sub("motor_position_request", requestPositionCallback);

// 서보 각도 요청 콜백
void requestServoAngleCallback(const std_msgs::Int16MultiArray &msg) {
    publishServoAngles();
}

ros::Subscriber<std_msgs::Int16MultiArray> servo_angle_request_sub("servo_angle_request", requestServoAngleCallback);

// Publisher for Feedback, Arduino -> PC
std_msgs::Int16MultiArray motor_speed_msg;
ros::Publisher motor_speed_pub("motor_speed_fb", &motor_speed_msg);

std_msgs::Int16MultiArray motor_position_msg;
ros::Publisher motor_position_pub("motor_position_fb", &motor_position_msg);

std_msgs::Int16MultiArray servo_angle_msg;
ros::Publisher servo_angle_pub("servo_angle_fb", &servo_angle_msg);

// 모터 속도 퍼블리시 함수
void publishMotorSpeed() {
    motor_speed_msg.data_length = 10;
    motor_speed_msg.data = velocity_set1;
    motor_speed_pub.publish(&motor_speed_msg);
}

// 모터 위치 퍼블리시 함수
void publishMotorPosition() {
    motor_position_msg.data_length = 10;
    motor_position_msg.data = position_set1;
    motor_position_pub.publish(&motor_position_msg);
}

// 서보 모터 각도 퍼블리시 함수
void publishServoAngles() {
    servo_angle_msg.data_length = 8;
    servo_angle_msg.data = servo_angles;
    servo_angle_pub.publish(&servo_angle_msg);
}

// 설정 및 초기화
void setup() {
    Serial.begin(57600);
    nh.initNode();
    
    nh.subscribe(speed_sub);
    nh.subscribe(position_sub);
    nh.subscribe(servo_sub);
    nh.subscribe(speed_request_sub);
    nh.subscribe(position_request_sub);
    nh.subscribe(servo_angle_request_sub);

    nh.advertise(motor_speed_pub);
    nh.advertise(motor_position_pub);
    nh.advertise(servo_angle_pub);
    
    // 서보 모터 핀 설정
    for (int i = 0; i < 8; i++) {
        servos[i].attach(servo_pins[i]);
    }

    Serial.println("Arduino ROS Initialized.");

    publishMotorSpeed();
    publishMotorPosition();
    publishServoAngles();
}

// 루프 함수
void loop() {
    nh.spinOnce();
}
