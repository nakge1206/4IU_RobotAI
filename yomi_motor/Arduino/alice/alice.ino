#include <Servo.h>
#include <ros.h>
#include <std_msgs/Int16MultiArray.h>
#include <std_msgs/Bool.h>
#define tx_enable 3
#define SN74LV_TX \
  digitalWrite(tx_enable, HIGH); \
  delayMicroseconds(10);
#define SN74LV_RX \
  digitalWrite(tx_enable, LOW); \
  delayMicroseconds(10);


Servo servo_left;
Servo servo_right;
byte gbpTxBuffer[128];
byte gbpTxBuffer_ax[128];
byte gbpParameter[128];



int speed = 300;
int init_angle[6] = { 300, 90, 100, 80, 150, 90 };  //1, 2, 3, 45, wrist, servo

int init_PLA_left_angle[4] = { 100, 130, 130, 100 };   //1, 2, 3, 45
int init_PLA_right_angle[4] = { 100, 130, 130, 100 };  //1, 2, 3, 45

int init_TPU_left_angle[4] = { 100, 120, 140, 110 };   //1, 2, 3, 45
int init_TPU_right_angle[4] = { 100, 140, 140, 110 };  //1, 2, 3, 45

int position_set[12] = { 1,1,1,1,1,1,
                         1,1,1,1,1,1
                        };

char bt_data = 0;

//  TxPacket (ID_number,0x06,0);
//  0xff 0xff 0x00 0x02 0x06
byte TxPacket(byte bID, byte blnstruction, byte bParameterLength) {
  byte bCount, bCheckSum, bPacketLength;
  gbpTxBuffer[0] = 0xff;                  //패킷의 시작을 알리는 신호
  gbpTxBuffer[1] = 0xff;                  //패킷의 시작을 알리는 신호
  gbpTxBuffer[2] = bID;                   //ID
  gbpTxBuffer[3] = bParameterLength + 2;  //패킷의 길이  Instruction(1) + Parameter0(1) + ParameterN(N) Parameter 개수(N) + 2
  gbpTxBuffer[4] = blnstruction;          //Dynamixel에게 수행하라고 지시하는 명령.

  for (bCount = 0; bCount < bParameterLength; bCount++)  //Put gbpParameter Value in gbpTxBuffer
  {
    gbpTxBuffer[bCount + 5] = gbpParameter[bCount];
  }
  bCheckSum = 0;
  bPacketLength = bParameterLength + 6;                   //StartByte(2) + IDByte(1) + LengthByte(1) + Instruction(1) + Parameter(N)+Checksum(1)
  for (bCount = 2; bCount < bPacketLength - 1; bCount++)  //StartByte 제외, Parameter0 포함, CheckSum 계산
  {
    bCheckSum += gbpTxBuffer[bCount];
  }
  gbpTxBuffer[bCount] = ~bCheckSum;  // Check Sum = ~( ID + Length + Instruction + Parameter1 + … Parameter N )

  for (bCount = 0; bCount < bPacketLength; bCount++)  //uart통신 Packet 전송
  {
    Serial1.write(gbpTxBuffer[bCount]);
  }
  Serial1.flush();
}

void id_set(unsigned int ID_number) {
  gbpParameter[0] = 0x03;  //ID address
  gbpParameter[1] = ID_number;
  TxPacket(0xFE, 0x03, 2);
}

void baud_set(unsigned int borate) {
  gbpParameter[0] = 0x04;  //Baud Rate Address
  gbpParameter[1] = borate;

  TxPacket(0xFE, 0x03, 2);  //모터 전체 , 0X03명령 , 길이
}
void mode_set(unsigned int mode) {  //이 함수 안쓰는거 같은데;
  //mode:0 서보제어, mode:1 바퀴 //
  if (mode == 0) {
    gbpParameter[0] = 0x06;                         //Angle Limit Address?????????????????????????????????
    gbpParameter[1] = (unsigned char)(0x00);        //CW Angle Limit(L)
    gbpParameter[2] = (unsigned char)(0x00 >> 8);   //CW Angle Limit(H)
    gbpParameter[3] = (unsigned char)(0x3FF);       //CCW Angle Limit(L)
    gbpParameter[4] = (unsigned char)(0x3FF >> 8);  //CCW Angle Limit(H)
    TxPacket(0xFE, 0x03, 5);
  } else if (mode == 1) {
    gbpParameter[0] = 0x06;                        //Angle Limit Address
    gbpParameter[1] = (unsigned char)(0x00);       //CW Angle Limit(L)
    gbpParameter[2] = (unsigned char)(0x00 >> 8);  //CW Angle Limit(H)
    gbpParameter[3] = (unsigned char)(0x00);       //CCW Angle Limit(L)
    gbpParameter[4] = (unsigned char)(0x00 >> 8);  //CCW Angle Limit(H)
    TxPacket(0xFE, 0x03, 5);
  }
}

byte dynamixel_position(unsigned int p_number, unsigned int ID_number)  //모터 위치값,모터 ID값
{
  unsigned int position = 3.41 * p_number;           //Change 0~300 to 0~1023
  gbpParameter[0] = 0x1E;                            //goal position(L) address
  gbpParameter[1] = (unsigned char)(position);       //Writing Data  , goal position(L)
  gbpParameter[2] = (unsigned char)(position >> 8);  //goal position(H)
  TxPacket(ID_number, 0x03, 3);                      // , 0x03명령, 길이
}


void ax_motor_left(unsigned int speed_1, unsigned int p_num_1, unsigned int speed_2,
                   unsigned int p_num_2, unsigned int speed_3, unsigned int p_num_3, unsigned int speed_4,
                   unsigned int p_num_4, unsigned int speed_5, unsigned int p_num_5) {
  byte bCheckSum = 0;
  unsigned int motor_speed[5] = { 3.41 * speed_1, 3.41 * speed_2, 3.41 * speed_3, 3.41 * speed_4, 3.41 * speed_5 };
  unsigned int motor_angle[5] = { 3.41 * p_num_1, 3.41 * p_num_2, 3.41 * p_num_3, 3.41 * p_num_4, 3.41 * p_num_5 };

  ///AX-12
  gbpTxBuffer_ax[0] = 0xFF;
  gbpTxBuffer_ax[1] = 0xFF;  //시작
  gbpTxBuffer_ax[2] = 0xFE;  //ID (0xFE : 전체 모터 대상)

  gbpTxBuffer_ax[3] = ((4 + 1) * 5) + 4;  //길이 (L:Dynamixel별 Data Length, N:Dynamixel의 개수)

  gbpTxBuffer_ax[4] = 0x83;  //명령

  gbpTxBuffer_ax[5] = 0x1E;  //Data를 쓰고자 하는 곳의 시작 Address
  gbpTxBuffer_ax[6] = 4;     //길이: Dynamixel별 Data Length

  gbpTxBuffer_ax[7] = 0x01;                                   //ID
  gbpTxBuffer_ax[8] = (unsigned char)(motor_angle[0]);        //0x04 goal position(L)
  gbpTxBuffer_ax[9] = (unsigned char)(motor_angle[0] >> 8);   //0x04 goal position(H)
  gbpTxBuffer_ax[10] = (unsigned char)(motor_speed[0]);       //0x04 goal Speed(L)
  gbpTxBuffer_ax[11] = (unsigned char)(motor_speed[0] >> 8);  //0x04 goal Speed(H)

  gbpTxBuffer_ax[12] = 0x02;                                  //ID
  gbpTxBuffer_ax[13] = (unsigned char)(motor_angle[1]);       //0x05 goal position(L)
  gbpTxBuffer_ax[14] = (unsigned char)(motor_angle[1] >> 8);  //0x05 goal position(H)
  gbpTxBuffer_ax[15] = (unsigned char)(motor_speed[1]);       //0x05 goal Speed(L)
  gbpTxBuffer_ax[16] = (unsigned char)(motor_speed[1] >> 8);  //0x05 goal Speed(H)

  gbpTxBuffer_ax[17] = 0x03;                                  //ID
  gbpTxBuffer_ax[18] = (unsigned char)(motor_angle[2]);       //0x05 goal position(L)
  gbpTxBuffer_ax[19] = (unsigned char)(motor_angle[2] >> 8);  //0x05 goal position(H)
  gbpTxBuffer_ax[20] = (unsigned char)(motor_speed[2]);       //0x05 goal Speed(L)
  gbpTxBuffer_ax[21] = (unsigned char)(motor_speed[2] >> 8);  //0x05 goal Speed(H)

  gbpTxBuffer_ax[22] = 0x04;                                  //ID
  gbpTxBuffer_ax[23] = (unsigned char)(motor_angle[3]);       //0x05 goal position(L)
  gbpTxBuffer_ax[24] = (unsigned char)(motor_angle[3] >> 8);  //0x05 goal position(H)
  gbpTxBuffer_ax[25] = (unsigned char)(motor_speed[3]);       //0x05 goal Speed(L)
  gbpTxBuffer_ax[26] = (unsigned char)(motor_speed[3] >> 8);  //0x05 goal Speed(H)

  gbpTxBuffer_ax[27] = 0x05;                                  //ID
  gbpTxBuffer_ax[28] = (unsigned char)(motor_angle[4]);       //0x05 goal position(L)
  gbpTxBuffer_ax[29] = (unsigned char)(motor_angle[4] >> 8);  //0x05 goal position(H)
  gbpTxBuffer_ax[30] = (unsigned char)(motor_speed[4]);       //0x05 goal Speed(L)
  gbpTxBuffer_ax[31] = (unsigned char)(motor_speed[4] >> 8);  //0x05 goal Speed(H)

  bCheckSum = 0;

  for (int bCount = 2; bCount <= 31; bCount++) {
    bCheckSum += gbpTxBuffer_ax[bCount];
  }
  gbpTxBuffer_ax[32] = ~bCheckSum;
  SN74LV_TX
  for (int i = 0; i < 10; i++) {
    for (int bCount = 0; bCount <= 32; bCount++)  //uart통신 Packet 전송
    {
      Serial1.write(gbpTxBuffer_ax[bCount]);
    }
    Serial1.flush();
  }
  SN74LV_RX
}

void ax_motor_right(unsigned int speed_1, unsigned int p_num_1, unsigned int speed_2, unsigned int p_num_2, unsigned int speed_3, unsigned int p_num_3, unsigned int speed_4, unsigned int p_num_4, unsigned int speed_5, unsigned int p_num_5) {
  byte bCheckSum = 0;
  unsigned int motor_speed[5] = { 3.41 * speed_1, 3.41 * speed_2, 3.41 * speed_3, 3.41 * speed_4, 3.41 * speed_5 };
  unsigned int motor_angle[5] = { 3.41 * p_num_1, 3.41 * p_num_2, 3.41 * p_num_3, 3.41 * p_num_4, 3.41 * p_num_5 };

  ///AX-12
  gbpTxBuffer_ax[0] = 0xFF;
  gbpTxBuffer_ax[1] = 0xFF;  //시작
  gbpTxBuffer_ax[2] = 0xFE;  //ID

  gbpTxBuffer_ax[3] = ((4 + 1) * 5) + 4;  //길이 (L:Dynamixel별 Data Length, N:Dynamixel의 개수)

  gbpTxBuffer_ax[4] = 0x83;  //명령

  gbpTxBuffer_ax[5] = 0x1E;  //
  // Data를 쓰고자 하는 곳의 시작 Address (Goal Position으로 할거니까)-AX-12A Control table 참고
  gbpTxBuffer_ax[6] = 4;  //길이: Dynamixel별 Data Length

  gbpTxBuffer_ax[7] = 0x06;                                   //ID
  gbpTxBuffer_ax[8] = (unsigned char)(motor_angle[0]);        //0x04 goal position(L)
  gbpTxBuffer_ax[9] = (unsigned char)(motor_angle[0] >> 8);   //0x04 goal position(H)
  gbpTxBuffer_ax[10] = (unsigned char)(motor_speed[0]);       //0x04 goal Speed(L)
  gbpTxBuffer_ax[11] = (unsigned char)(motor_speed[0] >> 8);  //0x04 goal Speed(H)

  gbpTxBuffer_ax[12] = 0x07;                                  //ID
  gbpTxBuffer_ax[13] = (unsigned char)(motor_angle[1]);       //0x05 goal position(L)
  gbpTxBuffer_ax[14] = (unsigned char)(motor_angle[1] >> 8);  //0x05 goal position(H)
  gbpTxBuffer_ax[15] = (unsigned char)(motor_speed[1]);       //0x05 goal Speed(L)
  gbpTxBuffer_ax[16] = (unsigned char)(motor_speed[1] >> 8);  //0x05 goal Speed(H)

  gbpTxBuffer_ax[17] = 0x08;                                  //ID
  gbpTxBuffer_ax[18] = (unsigned char)(motor_angle[2]);       //0x05 goal position(L)
  gbpTxBuffer_ax[19] = (unsigned char)(motor_angle[2] >> 8);  //0x05 goal position(H)
  gbpTxBuffer_ax[20] = (unsigned char)(motor_speed[2]);       //0x05 goal Speed(L)
  gbpTxBuffer_ax[21] = (unsigned char)(motor_speed[2] >> 8);  //0x05 goal Speed(H)

  gbpTxBuffer_ax[22] = 0x09;                                  //ID
  gbpTxBuffer_ax[23] = (unsigned char)(motor_angle[3]);       //0x05 goal position(L)
  gbpTxBuffer_ax[24] = (unsigned char)(motor_angle[3] >> 8);  //0x05 goal position(H)
  gbpTxBuffer_ax[25] = (unsigned char)(motor_speed[3]);       //0x05 goal Speed(L)
  gbpTxBuffer_ax[26] = (unsigned char)(motor_speed[3] >> 8);  //0x05 goal Speed(H)

  gbpTxBuffer_ax[27] = 0x10;                                  //ID
  gbpTxBuffer_ax[28] = (unsigned char)(motor_angle[4]);       //0x05 goal position(L)
  gbpTxBuffer_ax[29] = (unsigned char)(motor_angle[4] >> 8);  //0x05 goal position(H)
  gbpTxBuffer_ax[30] = (unsigned char)(motor_speed[4]);       //0x05 goal Speed(L)
  gbpTxBuffer_ax[31] = (unsigned char)(motor_speed[4] >> 8);  //0x05 goal Speed(H)

  bCheckSum = 0;
  for (int bCount = 2; bCount <= 31; bCount++) {
    bCheckSum += gbpTxBuffer_ax[bCount];
  }
  gbpTxBuffer_ax[32] = ~bCheckSum;
  SN74LV_TX
  for (int i = 0; i < 10; i++) {
    for (int bCount = 0; bCount <= 32; bCount++)  //uart통신 Packet 전송
    {
      Serial1.write(gbpTxBuffer_ax[bCount]);
    }
    Serial1.flush();
  }
  SN74LV_RX
}



void ax_init_angle_test() {
  ax_motor_left(300, 300,
                300, 90,
                300, 100,
                300, 100,
                300, 150);
  ax_motor_right(300, 300,
                 300, 90,
                 300, 100,
                 300, 100,
                 300, 150);
  servo_left.write(90);
  servo_right.write(90);
  delay(3000);

  ax_motor_left(300, 220,
                300, 190,
                300, 200,
                300, 200,
                300, 150);
  ax_motor_right(300, 220,
                 300, 190,
                 300, 200,
                 300, 200,
                 300, 150);
  delay(1000);
  ax_motor_left(300, 220,
                300, 190,
                300, 200,
                300, 200,
                300, 100);
  ax_motor_right(300, 220,
                 300, 190,
                 300, 200,
                 300, 200,
                 300, 100);
  delay(1000);
  ax_motor_left(300, 220,
                300, 190,
                300, 200,
                300, 200,
                300, 150);
  ax_motor_right(300, 220,
                 300, 190,
                 300, 200,
                 300, 200,
                 300, 150);
  delay(1000);
}

void clamp(int *value) {
  if (*value < 0) {
    *value = 0;
  } else if (*value > 5) {
    *value = 5;
  }
}
void handle(int left_1, int left_2, int left_3, int left_45, int left_wrist, int left_servo,
            int right_1, int right_2, int right_3, int right_45, int right_wrist, int right_servo) {

  clamp(&left_1);
  clamp(&left_2);
  clamp(&left_3);
  clamp(&left_45);
  clamp(&left_wrist);
  clamp(&left_servo);
  clamp(&right_1);
  clamp(&right_2);
  clamp(&right_3);
  clamp(&right_45);
  clamp(&right_wrist);
  clamp(&right_servo);



  ax_motor_left(speed, init_angle[0] - (init_TPU_left_angle[0] / 5) * left_1,
                speed, init_angle[1] + (init_TPU_left_angle[1] / 5) * left_2,
                speed, init_angle[2] + (init_TPU_left_angle[2] / 5) * left_3,
                speed, init_angle[3] + (init_TPU_left_angle[3] / 5) * left_45,
                speed, init_angle[4] - (90 / 5) * left_wrist);  //안쪽으로 돌면 ++, 바깥쪽으로돌면 --
  servo_left.write(init_angle[5] - (50 / 5) * left_servo);      //추가예정


  ax_motor_right(speed, init_angle[0] - (init_TPU_right_angle[1] / 5) * right_2,
                 speed, init_angle[1] + (init_TPU_right_angle[0] / 5) * right_1,
                 speed, init_angle[2] + (init_TPU_right_angle[3] / 5) * right_45,
                 speed, init_angle[3] + (init_TPU_right_angle[2] / 5) * right_3,
                 speed, init_angle[4] + (90 / 5) * right_wrist);  //안쪽으로 돌면 ++, 바깥쪽으로돌면 --
  servo_right.write(init_angle[5] + (50 / 5) * right_servo);      //추가예정


}

void handle_ros(int position[12]) {
  for (int i = 0; i < 12; i++) {
    clamp(&position[i]);
  }

  ax_motor_left(speed, init_angle[0] - (init_TPU_left_angle[0] / 5) * position[0],
                speed, init_angle[1] + (init_TPU_left_angle[1] / 5) * position[1],
                speed, init_angle[2] + (init_TPU_left_angle[2] / 5) * position[2],
                speed, init_angle[3] + (init_TPU_left_angle[3] / 5) * position[3],
                speed, init_angle[4] - (90 / 5) * position[4]);

  servo_left.write(init_angle[5] - (50 / 5) * position[5]);

  ax_motor_right(speed, init_angle[0] - (init_TPU_right_angle[1] / 5) * position[7],
                 speed, init_angle[1] + (init_TPU_right_angle[0] / 5) * position[6],
                 speed, init_angle[2] + (init_TPU_right_angle[3] / 5) * position[9],
                 speed, init_angle[3] + (init_TPU_right_angle[2] / 5) * position[8],
                 speed, init_angle[4] + (90 / 5) * position[10]);

  servo_right.write(init_angle[5] + (50 / 5) * position[11]);
}
//////////////////////////////////////////////
// 콜백이 호출되었는지 확인하는 플래그
volatile bool position_updated = false;


// ROS 노드 생성
ros::NodeHandle nh;

// XM430 모터 위치 수신 콜백
void positionCallback(const std_msgs::Int16MultiArray &msg) {
  if (msg.data_length != 12) return;

  for (int i = 0; i < 12; i++) {
    position_set[i] = msg.data[i];
  }

  position_updated = true;  // 위치가 업데이트됨을 표시
  publishMotorPosition();
  handle_ros(position_set);
}
// 모터 위치 요청 콜백
void requestPositionCallback(const std_msgs::Int16MultiArray &msg) {
  publishMotorPosition();
}

ros::Subscriber<std_msgs::Int16MultiArray> position_sub("motor_position_cmd", positionCallback);
ros::Subscriber<std_msgs::Int16MultiArray> position_request_sub("motor_position_request", requestPositionCallback);

std_msgs::Int16MultiArray motor_position_msg;
ros::Publisher motor_position_pub("motor_position_fb", &motor_position_msg);


// 모터 위치 퍼블리시 함수
void publishMotorPosition() {
  motor_position_msg.data_length = 12;
  motor_position_msg.data = position_set;
  motor_position_pub.publish(&motor_position_msg);
}



void setup() {
  // servo1.attach(6);
  // servo.write(90);
  pinMode(tx_enable, OUTPUT);
  servo_left.attach(2);   //left
  servo_right.attach(4);  // right

  servo_left.write(init_angle[5]);
  servo_right.write(init_angle[5]);

  Serial.begin(57600); // ROS SERIAL

  Serial1.begin(1000000, SERIAL_8N1);
  Serial2.begin(9600);
  delay(500);
  //id_set(0x05);
  //dynamixel_position(290, 0xfe);

  //LEFT:  1, 2, 3, 45, wrist, servo
  //RIGHT: 1, 2, 3, 45, wrist, servo
  handle(0, 0, 0, 0, 3, 3,
         0, 0, 0, 0, 3, 3);  //init 자세



  nh.initNode();

  nh.subscribe(position_sub);
  nh.subscribe(position_request_sub);

  nh.advertise(motor_position_pub);

  Serial.println("Arduino ROS Initialized.");

  publishMotorPosition();
}
/*
unsigned long previousMillis = 0;                              // 이전 시간 저장
int step = 0;                                                  // 현재 단계
const unsigned long intervals[] = { 1000, 4000, 6000, 2000 };  // 각 단계별 시간 간격
/*
ang_mat_11 = [0   2   3   4   5   6     7   8   9   10  11 12   13;    % time
              ssr 0   0   0   0   0     -5  -5  -5  -5  -5  -5  -5;     % sag_sh_l
              ser 130 130 130 130 130   40   40   40   40   40   40   40;    % sag_el_l
              tsr 0   -30  0  -30 0     0   0   0   0   0   0   0;     % tran_sh_l
              ssr 0   0   0   0   0     30  0   30  0   30  0   0;     % sag_sh_r   
              ser 130 130 130 130 130   70  70  70  70  70  70 ser];    % sag_el_r
*/

void loop() {

  nh.spinOnce();
  /*
  if (Serial2.available()) {
    bt_data = Serial2.read();
    //Serial.print(bt_data);
  }
  if (bt_data == 'n') {
    handle(0, 0, 0, 0, 3, 3,
           0, 0, 0, 0, 3, 3);  //init 자세
    Serial2.println("stop");
  }
  if (bt_data == 'o') {
    unsigned long currentMillis = millis();  // 현재 시간 확인

    if (currentMillis - previousMillis >= intervals[step]) {
      previousMillis = currentMillis;  // 이전 시간 갱신

      switch (step) {
        case 0:
          Serial2.println("start");
          // 시작 initial 팔목 각도, 손 펴기
          handle(0, 0, 0, 0, 3, 3,
                 0, 0, 0, 0, 3, 3);
          break;

        case 1:
          // 손 흔들때 팔목 각도
          handle(0, 0, 0, 0, 1, 3,
                 0, 0, 0, 0, 1, 3);
          break;

        case 2:
          // 왼팔: initial 팔목 각도, 주먹자세, 오른팔: 악수 팔목 각도, 약간 주먹
          handle(4, 4, 4, 4, 5, 3,
                 4, 4, 4, 4, 5, 3);
          break;

        case 3:
          // 손 펼침
          handle(0, 0, 0, 0, 5, 3,
                 0, 0, 0, 0, 5, 3);
          Serial2.println("end");
          step = -1;  // 모든 단계 완료 후 리셋
          break;
      }

      step++;  // 다음 단계로 이동
    }
  }*/
}
