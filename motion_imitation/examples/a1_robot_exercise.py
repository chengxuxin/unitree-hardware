"""Commands A1 robot to raise and lower its legs so it crouches and stands up.

Can be run in sim by setting --real_robot=False.
"""
import sys
sys.path.append("../")
import numpy as np
from robot_interface import RobotInterface
import time
from tqdm import tqdm
from robots.robot_config import MotorControlMode

class A1Robot():
  def __init__(self) -> None: 
    # Robot state variables
    self._init_complete = False
    self._base_position = np.zeros((3,))
    self._base_orientation = None
    self._last_position_update_time = time.time()
    self._raw_state = None
    self._last_raw_state = None
    self._motor_angles = np.zeros(12)
    self._motor_velocities = np.zeros(12)
    self._motor_temperatures = np.zeros(12)
    self._joint_states = None
    self._last_reset_time = time.time()

    self.commands = np.zeros(60, dtype=np.float32)
    self.default_dof_pos = np.array([-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5])
    # self.Kp = 40
    # self.Kd = 5

    # Initiate UDP for robot state and actions
    self._robot_interface = RobotInterface()
    for i in range(100):
      self.ReceiveObservation()
      time.sleep(0.01)
    # self._robot_interface.send_command(np.zeros(60, dtype=np.float32))
  
  def Step(self, actions, Kp, Kd):
    actions = self._clip_actions(actions)
    for motor_id in range(12):
      self.commands[5*motor_id] = actions[motor_id]
      self.commands[5*motor_id+1] = Kp
      self.commands[5*motor_id+2] = 0.
      self.commands[5*motor_id+3] = Kd
      self.commands[5*motor_id+4] = 0.
    # self.torques = self._compute_torques(actions)
    self._robot_interface.send_command(self.commands)

  def _clip_actions(self, actions):
    return np.clip(actions, -1+self.default_dof_pos, 1+self.default_dof_pos)
  
  def ReceiveObservation(self):
      """Receives observation from robot.

      Synchronous ReceiveObservation is not supported in A1,
      so changging it to noop instead.
      """
      state = self._robot_interface.receive_observation()
      self._raw_state = state
      # Convert quaternion from wxyz to xyzw, which is default for Pybullet.
      q = state.imu.quaternion
      rpy = state.imu.rpy
      print(state.tick)
      self._base_quat = np.array(q[0:4])
      self._base_rpy = np.array(rpy[0:3])
      self._ang_vel = np.array(state.imu.gyroscope[0:3])
      self._accelerometer_reading = np.array(state.imu.accelerometer)
      self._motor_angles = np.array([motor.q for motor in state.motorState[:12]])
      self._motor_velocities = np.array([motor.dq for motor in state.motorState[:12]])
      # self._joint_states = np.array(list(zip(self._motor_angles, self._motor_velocities)))
      # self._observed_motor_torques = np.array([motor.tauEst for motor in state.motorState[:12]])
      # self._motor_temperatures = np.array([motor.temperature for motor in state.motorState[:12]])

  def GetTrueMotorAngles(self):
    return self._motor_angles.copy()
  
  def GetMotorVel(self):
    return self._motor_velocities.copy()

  def GetBaseQuat(self):
    return self._base_quat.copy()
  
  def GetBaseRPY(self):
    return self._base_rpy.copy()
  
  def GetAngVel(self):
    return self._ang_vel.copy()

KP = 40
KD = 5
def main():
  robot = A1Robot()
  # Move the motors slowly to initial position
  robot.ReceiveObservation()
  init_motor_angle = np.array(robot.GetTrueMotorAngles())
  desired_motor_angle = robot.default_dof_pos.copy()
  print(init_motor_angle)

  for t in tqdm(range(400)):
    # print(robot.GetBaseOrientation())
    current_motor_angle = np.array(robot.GetTrueMotorAngles())
    robot.ReceiveObservation()
    blend_ratio = np.minimum(t / 250., 1)
    action = (1 - blend_ratio) * current_motor_angle + blend_ratio * desired_motor_angle
    robot.Step(action, Kp=KP, Kd=KD)
    time.sleep(0.02)

  for t in tqdm(range(900)):
    # print(robot.GetBaseOrientation())
    current_motor_angle = np.array(robot.GetTrueMotorAngles())
    robot.ReceiveObservation()
    blend_ratio = np.minimum(t / 800., 1)
    action = (1 - blend_ratio) * current_motor_angle + blend_ratio * init_motor_angle
    robot.Step(action, Kp=KP, Kd=KD)
    time.sleep(0.02)


if __name__ == "__main__":
  main()
