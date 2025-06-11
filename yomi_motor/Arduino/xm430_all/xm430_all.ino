#include <Servo.h>
#include <ros.h>
#include <std_msgs/Int16MultiArray.h>
#include <std_msgs/Bool.h>

#define tx_enable 12
#define MAX485_TX \
  digitalWrite(tx_enable, HIGH); \
  delayMicroseconds(10);
#define MAX485_RX \
  digitalWrite(tx_enable, LOW); \
  delayMicroseconds(10);

#define VELOCITY_MAX 10
// XM430 MAX VELOCITY

byte gbpTxBuffer[128];
byte gbpParameter[128];

// 서보 모터 객체 생성
Servo servos[8];
int servo_pins[8] = {2, 3, 4, 5,
                     7, 8, 9, 10
                    };
// initial 서보 각도
int servo_angles[8] = { 90, 90, 90, 90,
                        90, 90, 90, 90
                      };
//*****************************************************
int servo_min_angle[8] = { 0, 0, 0, 0,
                           0, 0, 0, 0
                         };
int servo_max_angle[8] = { 90, 130, 140, 125,
                           90, 110, 120, 145
                         };
//*****************************************************
int servo_init_angles[8] = { 90, 130, 140, 125,
                             90, 110, 120, 145
                           };

unsigned char motor_id[10] = {  0x03, 0x04, 0x05, 0x06,
                                0x07, 0x08, 0x09, 0x0A,
                                0x0B, 0x0C
                             };  // 모터 ID 리스트
//*****************************************************
// 모터별 최소 및 최대 각도 설정
int min_angle[10] = {120, 110, 90, 95,
                     60, 100, 90, 180,
                     120, 110
                    };  // 모터별 최소 각도
int max_angle[10] = {300, 260, 270, 180,
                     240, 250, 270, 265,
                     240, 250
                    };  // 모터별 최대 각도
//*****************************************************

// 사용할 스위치 핀 (A8~A13)
const int buttonPins[6] = {A8, A9, A10, A11, A12, A13};
volatile bool buttonState[6] = {HIGH, HIGH, HIGH, HIGH, HIGH, HIGH};  // 스위치 상태 저장


float velocity = 77;


// XM430 initial 속도 및 위치 배열
int velocity_set[10] = { 1, 1, 1, 1,
                         1, 1, 1, 1,
                         1, 1
                       };
int position_set1[10] = { 180, 180, 180, 180,
                          180, 180, 180, 180,
                          180, 180
                        };


unsigned short update_crc(unsigned short crc_accum, unsigned char *data_blk_ptr, unsigned short data_blk_size) {
  /*
    리턴 값 : 16bit CRC 값
    crc_accum : ‘0’으로 설정
    data_blk_ptr : Packet array pointer
    data_blk_size : CRC를 제외한 패킷의 byte 수
    data_blk_size = Header(3) + Reserved(1) + Packet ID(1) + Length(2) + Length - CRC(2) = 3 + 1 + 1 + 2 + Length - 2 = 5 + Length;
    Packet Length = (LEN_H « 8 ) + LEN_L; //Little-endian
  */
  unsigned short i, j;
  unsigned short crc_table[256] = {
    0x0000, 0x8005, 0x800F, 0x000A, 0x801B, 0x001E, 0x0014, 0x8011,
    0x8033, 0x0036, 0x003C, 0x8039, 0x0028, 0x802D, 0x8027, 0x0022,
    0x8063, 0x0066, 0x006C, 0x8069, 0x0078, 0x807D, 0x8077, 0x0072,
    0x0050, 0x8055, 0x805F, 0x005A, 0x804B, 0x004E, 0x0044, 0x8041,
    0x80C3, 0x00C6, 0x00CC, 0x80C9, 0x00D8, 0x80DD, 0x80D7, 0x00D2,
    0x00F0, 0x80F5, 0x80FF, 0x00FA, 0x80EB, 0x00EE, 0x00E4, 0x80E1,
    0x00A0, 0x80A5, 0x80AF, 0x00AA, 0x80BB, 0x00BE, 0x00B4, 0x80B1,
    0x8093, 0x0096, 0x009C, 0x8099, 0x0088, 0x808D, 0x8087, 0x0082,
    0x8183, 0x0186, 0x018C, 0x8189, 0x0198, 0x819D, 0x8197, 0x0192,
    0x01B0, 0x81B5, 0x81BF, 0x01BA, 0x81AB, 0x01AE, 0x01A4, 0x81A1,
    0x01E0, 0x81E5, 0x81EF, 0x01EA, 0x81FB, 0x01FE, 0x01F4, 0x81F1,
    0x81D3, 0x01D6, 0x01DC, 0x81D9, 0x01C8, 0x81CD, 0x81C7, 0x01C2,
    0x0140, 0x8145, 0x814F, 0x014A, 0x815B, 0x015E, 0x0154, 0x8151,
    0x8173, 0x0176, 0x017C, 0x8179, 0x0168, 0x816D, 0x8167, 0x0162,
    0x8123, 0x0126, 0x012C, 0x8129, 0x0138, 0x813D, 0x8137, 0x0132,
    0x0110, 0x8115, 0x811F, 0x011A, 0x810B, 0x010E, 0x0104, 0x8101,
    0x8303, 0x0306, 0x030C, 0x8309, 0x0318, 0x831D, 0x8317, 0x0312,
    0x0330, 0x8335, 0x833F, 0x033A, 0x832B, 0x032E, 0x0324, 0x8321,
    0x0360, 0x8365, 0x836F, 0x036A, 0x837B, 0x037E, 0x0374, 0x8371,
    0x8353, 0x0356, 0x035C, 0x8359, 0x0348, 0x834D, 0x8347, 0x0342,
    0x03C0, 0x83C5, 0x83CF, 0x03CA, 0x83DB, 0x03DE, 0x03D4, 0x83D1,
    0x83F3, 0x03F6, 0x03FC, 0x83F9, 0x03E8, 0x83ED, 0x83E7, 0x03E2,
    0x83A3, 0x03A6, 0x03AC, 0x83A9, 0x03B8, 0x83BD, 0x83B7, 0x03B2,
    0x0390, 0x8395, 0x839F, 0x039A, 0x838B, 0x038E, 0x0384, 0x8381,
    0x0280, 0x8285, 0x828F, 0x028A, 0x829B, 0x029E, 0x0294, 0x8291,
    0x82B3, 0x02B6, 0x02BC, 0x82B9, 0x02A8, 0x82AD, 0x82A7, 0x02A2,
    0x82E3, 0x02E6, 0x02EC, 0x82E9, 0x02F8, 0x82FD, 0x82F7, 0x02F2,
    0x02D0, 0x82D5, 0x82DF, 0x02DA, 0x82CB, 0x02CE, 0x02C4, 0x82C1,
    0x8243, 0x0246, 0x024C, 0x8249, 0x0258, 0x825D, 0x8257, 0x0252,
    0x0270, 0x8275, 0x827F, 0x027A, 0x826B, 0x026E, 0x0264, 0x8261,
    0x0220, 0x8225, 0x822F, 0x022A, 0x823B, 0x023E, 0x0234, 0x8231,
    0x8213, 0x0216, 0x021C, 0x8219, 0x0208, 0x820D, 0x8207, 0x0202
  };
  for (j = 0; j < data_blk_size; j++) {
    i = ((unsigned short)(crc_accum >> 8) ^ data_blk_ptr[j]) & 0xFF;
    crc_accum = (crc_accum << 8) ^ crc_table[i];
  }
  return crc_accum;
}

void TxPacket_xm430(Stream &serialport, byte bID, byte blnstruction, byte bParameterLength)  //ID값,Instruction,parameter 길이
{
  byte bCount, bPacketLength;
  unsigned short bCRC;

  gbpTxBuffer[0] = 0xff;                                    //패킷의 시작을 알리는 신호
  gbpTxBuffer[1] = 0xff;                                    //패킷의 시작을 알리는 신호
  gbpTxBuffer[2] = 0xfd;                                    //패킷의 시작을 알리는 신호
  gbpTxBuffer[3] = 0x00;                                    //Reserved(Header와 동일한 기능)
  gbpTxBuffer[4] = bID;                                     //ID
  gbpTxBuffer[5] = ((bParameterLength + 3) & 0xff);         //패킷의 길이  (Parameter0(1) + ParameterN(N) Parameter 개수(N) Instruction(1) + 2) 하위 비트
  gbpTxBuffer[6] = (((bParameterLength + 3) >> 8) & 0xff);  //패킷의 길이  (Parameter0(1) + ParameterN(N) Parameter 개수(N) Instruction(1) + 2) 상위 비트
  gbpTxBuffer[7] = blnstruction;                            //Dynamixel에게 수행하라고 지시하는 명령.

  for (bCount = 8; bCount < (bParameterLength + 8); bCount++)  //Put gbpParameter Value in gbpTxBuffer
  {
    gbpTxBuffer[bCount] = gbpParameter[bCount - 8];
  }

  //CRC
  //Packet이 통신 중에 파손되었는지를 점검하기 위한 필드 (16bit CRC)
  //하위 바이트와 상위 바이트를 Instruction Packet에서 나누어서 보냄.
  //CRC 계산 범위: Instruction Packet의 Header (FF FF FD 00)를 포함하여, CRC 필드 이전까지.
  bPacketLength = bParameterLength + 8;  //Header(3) + Reserved(1) + Packet ID(1) + Length(2) + 패킷의 길이 - CRC(2) = 패킷의 길이 + 5 = (bParameterLength+3) + 5 = bParameterLength + 8
  bCRC = update_crc(0, gbpTxBuffer, bPacketLength);
  gbpTxBuffer[bCount] = (bCRC & 0xFF);
  gbpTxBuffer[bCount + 1] = (bCRC >> 8) & 0xFF;

  MAX485_TX
  for (bCount = 0; bCount < (bPacketLength + 2); bCount++) {  //uart통신 Packet 전송
    serialport.write(gbpTxBuffer[bCount]);
    serialport.flush();
  }
  MAX485_RX
}

void xm430_position(Stream &serialport, unsigned char ID_number, float p_number)  //모터 위치값,모터 ID값
{
  unsigned int position = 11.375 * p_number;  //Change 0~360 to 0~4095

  gbpParameter[0] = 0x74;                             //goal position address_L
  gbpParameter[1] = (0x74 >> 8);                      //goal position address_H
  gbpParameter[2] = (unsigned char)(position);        //Writing Data  , goal position
  gbpParameter[3] = (unsigned char)(position >> 8);   //goal position
  gbpParameter[4] = (unsigned char)(position >> 16);  //goal position
  gbpParameter[5] = (unsigned char)(position >> 24);  //goal position

  TxPacket_xm430(serialport, ID_number, 0x03, 0x06);  // , 0x03명령, 길이
}

void send_data(Stream &serialport, byte buf_0, byte buf_1, byte buf_2) {
  gbpTxBuffer[0] = 0xff;
  gbpTxBuffer[1] = 0xff;
  gbpTxBuffer[2] = buf_0;
  gbpTxBuffer[3] = buf_1;
  gbpTxBuffer[4] = buf_2;

  char checksum = 0;
  for (int i = 2; i < 5; i++) {
    checksum += gbpTxBuffer[i];
  }
  gbpTxBuffer[5] = ~checksum;

  for (int i = 0; i < 6; i++) {
    serialport.write(gbpTxBuffer[i]);
    serialport.flush();
  }
}

void xm430_Operating_mode(Stream &serialport, unsigned char ID_number, unsigned char Operating_mode)  //Operating mode 설정 함수
{
  //Operating Mode 0~16(기본값: 3)
  //0: 전류제어 모드
  //1: 속도제어 모드
  //3: 위치제어 모드
  //4: 확장 위치제어 모드(Multi-turn)
  //5: 전류기반 위치제어 모드
  //16:PWM 제어 모드 (Voltage Control Mode)

  xm430_Torque(serialport, 0xFE, 0);
  gbpParameter[0] = 0x0B;         //Operating mode address_L
  gbpParameter[1] = (0x0B >> 8);  //Operating mode address_H
  gbpParameter[2] = Operating_mode;

  TxPacket_xm430(serialport, ID_number, 0x03, 0x03);
  xm430_Torque(serialport, 0xFE, 1);
}

void xm430_Torque(Stream &serialport, unsigned char ID_number, unsigned char Torque)  //TORQUE 설정 함수
{
  //1이면 ON
  //0이면 OFF(EEPROM할 때 0으로 설정해야함)
  gbpParameter[0] = 0x40;         //TORQUE address_L
  gbpParameter[1] = (0x40 >> 8);  //TORQUE address_H
  gbpParameter[2] = Torque;

  TxPacket_xm430(serialport, ID_number, 0x03, 0x03);
}

void xm430_profile_velocity(Stream &serialport, unsigned char ID_number, float velocity) {
  unsigned int velocity_int = 4.364 * velocity;  //veloctiy: 0.0 ~ 77.0[rev/min] -> 0 ~ 336 (velocity unit: 0.229)

  gbpParameter[0] = 0x70;                                 //profile_velocity address_L
  gbpParameter[1] = (0x70 >> 8);                          //profile_velocity address_H
  gbpParameter[2] = (unsigned char)(velocity_int);        //Writing Data  , profile_velocity
  gbpParameter[3] = (unsigned char)(velocity_int >> 8);   //Writing Data  , profile_velocity
  gbpParameter[4] = (unsigned char)(velocity_int >> 16);  //Writing Data  , profile_velocity
  gbpParameter[5] = (unsigned char)(velocity_int >> 24);  //Writing Data  , profile_velocity

  TxPacket_xm430(serialport, ID_number, 0x03, 0x06);
}

void xm430_profile_acceleration(Stream &serialport, unsigned char ID_number, int32_t acceleration) {
  acceleration = 0.0046 * acceleration;  //acceleration: 0 ~ 6000[rev/min^2] -> 28 (acceleration unit: 214.577)

  gbpParameter[0] = 0x6C;                                 //profile_acceleration address_L
  gbpParameter[1] = (0x6C >> 8);                          //profile_acceleration address_H
  gbpParameter[2] = (unsigned char)(acceleration);        //Writing Data  , profile_acceleration
  gbpParameter[3] = (unsigned char)(acceleration >> 8);   //Writing Data  , profile_acceleration
  gbpParameter[4] = (unsigned char)(acceleration >> 16);  //Writing Data  , profile_acceleration
  gbpParameter[5] = (unsigned char)(acceleration >> 24);  //Writing Data  , profile_acceleration

  TxPacket_xm430(serialport, ID_number, 0x03, 0x06);
}

void xm430_init(Stream &serialport, int operating_mode, float velocity, int32_t acceleration) {
  // veloctiy: 0.0 ~ 77.0[rev/min] -> 0 ~ 336 (velocity unit: 0.229)
  // acceleration: 0 ~ 6000[rev/min^2] -> 28 (acceleration unit: 214.577)

  // Operating Mode(11) : 위치제어 모드, value : 3
  // Drive Mode(10) : Profile Configuration([0] Velocity-based Profile: 속도를 기준으로 Profile 생성), value : 0
  // Profile Velocity(112) : Drive Mode(10)에서 Velocity-based Profile이 선택된 경우, Profile Velocity(112)는 Profile의 최대 속도를 설정
  //                          단위: 0.229 [rev/min], 4 byte
  // Profile Acceleration(108) : Drive Mode(10)에서 Velocity-based Profile이 선택된 경우, Profile Acceleration(108)은 Profile의 가속도를 설정
  //                          단위: 214.577 [rev/min^2], 4 byte
  xm430_Torque(serialport, 0xFE, 0);

  xm430_Operating_mode(serialport, 0xfe, operating_mode);
  xm430_profile_velocity(serialport, 0xfe, velocity);
  xm430_profile_acceleration(serialport, 0xfe, acceleration);

  xm430_Torque(serialport, 0xFE, 1);
}



void set_all_motors(Stream &serialport, int velocity[10], int position[10]) {
  int position_limited[10];

  xm430_Torque(serialport, 0xFE, 0);
  xm430_Operating_mode(serialport, 0xFE, 3);

  for (int i = 0; i < 10; i++) {
    xm430_profile_velocity(serialport, motor_id[i], velocity[i]);
  }

  xm430_profile_acceleration(serialport, 0xFE, 30000);
  xm430_Torque(serialport, 0xFE, 1);

  for (int i = 0; i < 10; i++) {
    // 최소 및 최대 각도 제한 적용
    position_limited[i] = constrain(position[i], min_angle[i], max_angle[i]);
    xm430_position(serialport, motor_id[i], position_limited[i]);
  }
}


/////////////////////////////////////////////////////////////////////////////
// 콜백이 호출되었는지 확인하는 플래그
volatile bool speed_updated = false;
volatile bool position_updated = false;

// ROS 노드 생성
ros::NodeHandle nh;

// XM430 모터 속도 수신 콜백
void speedCallback(const std_msgs::Int16MultiArray &msg) {
  if (msg.data_length != 10) return;

  for (int i = 0; i < 10; i++) {
    velocity_set[i] = constrain(msg.data[i], 0, VELOCITY_MAX); // 0~VELOCITY_MAX 사이로 제한
  }

  speed_updated = true;  // 속도가 업데이트됨을 표시
  publishMotorSpeed();
  checkAndExecuteMotorUpdate();  // 상태 확인 후 `set_all_motors()` 실행
}

ros::Subscriber<std_msgs::Int16MultiArray> speed_sub("motor_speed_cmd", speedCallback);

// XM430 모터 위치 수신 콜백
void positionCallback(const std_msgs::Int16MultiArray &msg) {
  if (msg.data_length != 10) return;

  for (int i = 0; i < 10; i++) {
    position_set1[i] = msg.data[i];
  }

  position_updated = true;  // 위치가 업데이트됨을 표시
  publishMotorPosition();
  checkAndExecuteMotorUpdate();  // 상태 확인 후 `set_all_motors()` 실행
}

// 속도와 위치가 모두 업데이트되었을 때 한 번만 `set_all_motors()` 실행
void checkAndExecuteMotorUpdate() {
  if (speed_updated && position_updated) {
    set_all_motors(Serial1, velocity_set, position_set1);

    // 실행 후 플래그 초기화
    speed_updated = false;
    position_updated = false;
  }
}

ros::Subscriber<std_msgs::Int16MultiArray> position_sub("motor_position_cmd", positionCallback);

// 서보 모터 각도 수신 콜백
void servoCallback(const std_msgs::Int16MultiArray &msg) {
  if (msg.data_length != 8) return;

  for (int i = 0; i < 8; i++) {
    servo_angles[i] = constrain(msg.data[i], servo_min_angle[i], servo_max_angle[i]);
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

// 개별적인 ROS 퍼블리셔 추가 (bool 타입 사용)
std_msgs::Bool switch_msgs[6];  // 각 스위치 상태 메시지
ros::Publisher switch_pubs[6] = {
  ros::Publisher("switch_1_state", &switch_msgs[0]),
  ros::Publisher("switch_2_state", &switch_msgs[1]),
  ros::Publisher("switch_3_state", &switch_msgs[2]),
  ros::Publisher("switch_4_state", &switch_msgs[3]),
  ros::Publisher("switch_5_state", &switch_msgs[4]),
  ros::Publisher("switch_6_state", &switch_msgs[5])
};


// 모터 속도 퍼블리시 함수
void publishMotorSpeed() {
  motor_speed_msg.data_length = 10;
  motor_speed_msg.data = velocity_set;
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

// PCINT2_vect: A8~A13 핀 상태 변화 감지
ISR(PCINT2_vect) {
  for (int i = 0; i < 6; i++) {
    bool newState = digitalRead(buttonPins[i]);

    if (newState != buttonState[i]) {  // 상태 변화 감지
      buttonState[i] = newState;

      // 상태 변경 시 개별 토픽에 퍼블리시
      switch_msgs[i].data = (newState == LOW);  // LOW = 버튼 눌림 (true), HIGH = 버튼 해제 (false)
      switch_pubs[i].publish(&switch_msgs[i]);

    }
  }
}

void setup() {
  delay(50);//initialize wait

  Serial.begin(57600); // ROS SERIAL
  Serial1.begin(57600);  // XM430-W350-R
  Serial1.setTimeout(20);

  const int32_t acceleration = 30000;

  pinMode(tx_enable, OUTPUT);

  // 서보 모터 핀 설정
  for (int i = 0; i < 8; i++) {
    servos[i].attach(servo_pins[i]);
  }

  for (int i = 0; i < 6; i++) {
    pinMode(buttonPins[i], INPUT_PULLUP);
  }

  // PCICR: 핀 체인지 인터럽트 활성화 (PCINT2 그룹)
  PCICR |= (1 << PCIE2);  // PCINT2 그룹(A8~A15) 활성화

  // PCMSK2: A8~A11에 해당하는 PCINT 활성화 (PCINT24~27)
  PCMSK2 |= 0x3F;  // PCINT24~PCINT27 활성화 (A8~A13)

  // ROS 노드에 스위치 퍼블리셔 등록
  for (int i = 0; i < 6; i++) {
    nh.advertise(switch_pubs[i]);
  }

  for (int i = 0; i < 8; i++) {
    servos[i].write(servo_init_angles[i]);
  }

  //Serial.println("Arduino ROS Initialized.");

  const int motor_id = 0xFE;
  int angle_min, angle_max, motor_move_time;
  bool motor_toggle = true;
  xm430_init(Serial1, 3, velocity, acceleration);
  long _cnt = 0;

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


  publishMotorSpeed();
  publishMotorPosition();
  publishServoAngles();
}



void loop() {
  nh.spinOnce();




  /*
    float velocity_set1[10] = { 5, 5, 5, 5, 5, 5, 5, 5, 5, 5};

    //if - angle up
    //->
    //-> back, down, hand back, up
    //->
    float position_set1[10] = { 180, 180, 180, 180,
                              180, 260, 180, 180,
                              180, 180};
    set_all_motors(Serial1, velocity_set1, position_set1);
    delay(10000);
    //set_all_motors(Serial1, velocity_set2, position_set2);
    //delay(1000);*/

}
