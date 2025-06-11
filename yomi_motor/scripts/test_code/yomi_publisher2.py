import rospy
import tkinter as tk
from std_msgs.msg import Int16MultiArray
from yomi_msgs.msg import SetMotor  # 직접 만든 메시지 사용

rospy.init_node('gui_publisher', anonymous=True)

# 퍼블리셔 생성
speed_pub = rospy.Publisher('/set_motor_speed', SetMotor, queue_size=10)
position_pub = rospy.Publisher('/set_motor_position', SetMotor, queue_size=10)
servo_pub = rospy.Publisher('/set_servo_angle', SetMotor, queue_size=10)

def send_motor_speed(index, value):
    msg = SetMotor()
    msg.index = index
    msg.value = value
    speed_pub.publish(msg)
    print(f"Sent Motor Speed {index}: {value}")

def send_motor_position(index, value):
    msg = SetMotor()
    msg.index = index
    msg.value = value
    position_pub.publish(msg)
    print(f"Sent Motor Position {index}: {value}")

def send_servo_angle(index, value):
    msg = SetMotor()
    msg.index = index
    msg.value = value
    servo_pub.publish(msg)
    print(f"Sent Servo Angle {index}: {value}")

# GUI 생성
root = tk.Tk()
root.title("Motor Control")

tk.Label(root, text="Motor Index").grid(row=0, column=0)
tk.Label(root, text="Value").grid(row=0, column=1)

index_entry = tk.Entry(root)
index_entry.grid(row=1, column=0)
value_entry = tk.Entry(root)
value_entry.grid(row=1, column=1)

tk.Button(root, text="Set Speed", command=lambda: send_motor_speed(int(index_entry.get()), int(value_entry.get()))).grid(row=2, column=0)
tk.Button(root, text="Set Position", command=lambda: send_motor_position(int(index_entry.get()), int(value_entry.get()))).grid(row=2, column=1)
tk.Button(root, text="Set Servo", command=lambda: send_servo_angle(int(index_entry.get()), int(value_entry.get()))).grid(row=3, column=0)

root.mainloop()
