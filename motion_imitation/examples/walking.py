from a1_robot_exercise import *
import torch
from collections import deque
import argparse

def quatToEuler(quat):
    eulerVec = np.zeros(3)
    qw = quat[0], qx = quat[1], qy = quat[2], qz = quat[3]
    # roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    eulerVec[0] = np.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        eulerVec[1] = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        eulerVec[1] = np.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    eulerVec[2] = np.atan2(siny_cosp, cosy_cosp)
    
    return eulerVec


class A1Policy():
    def __init__(self, policy_file, Kp, Kd) -> None:
        self.action_scale = 0.25
        self.default_dof_pos = np.array([-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5])

        self.Kp = Kp
        self.Kd = Kd

        self.scales_ang_vel = 0.25
        self.scales_commands = np.array([2.0, 2.0, 0.25])
        self.scales_dof_pos = 1.0
        self.scales_dof_vel = 0.05

        self.n_priv = 3+3+3
        self.n_proprio = 3+2+3+36+1+1+1+1+3+4
        self.history_len = 4
        self.num_observations = self.n_proprio + self.history_len * self.n_proprio + self.n_priv
        self.obs = np.zeros((1, self.num_observations))
        self.last_action = np.zeros(12)
        self.proprio_history_buf = deque(maxlen=self.history_len)
        for i in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_proprio))
        
        self.commands = np.array([0.2, 0, 0])
        print("Loading jit...")
        self.policy = torch.jit.load(policy_file)
        print("Loaded jit from: {}".format(policy_file))
        self.robot = A1Robot()

    def step(self):
        start = time.time()
        self.robot.ReceiveObservation()
        obs = self.get_observation()
        # print(time.time() - start)
        # start = time.time()
        raw_actions = self.policy(torch.from_numpy(obs[None, :])).detach().numpy().squeeze()
        # print(raw_actions*self.actions_scale)
        # print(time.time() - start)
        PD_target = self.action_scale * raw_actions + self.default_dof_pos
        # self.robot.Step(PD_target, self.Kp, self.Kd)
        
        self.last_action = raw_actions.copy()
        self.proprio_history_buf.append(obs[:self.n_proprio])
        
    
    
    def get_observation(self):
        return np.concatenate([0.25*np.ones((1, )),  # dis2wall
                         self.robot.GetAngVel() * self.scales_ang_vel,
                         self.robot.GetBaseRPY()[:2],
                         0.3*np.ones(1, ), # env slope angle
                         -0.5*np.ones((1, )),  # hit flags
                         np.zeros(4),  # foot ref 
                         self.commands * self.scales_commands, 
                         (self.robot.GetTrueMotorAngles() - self.default_dof_pos) * self.scales_dof_pos,
                         self.robot.GetMotorVel() * self.scales_dof_vel,
                         self.last_action,
                         np.zeros((4, )), # foot contact
                         np.array(self.proprio_history_buf).flatten()]).astype(np.float32)

    def stand_up(self, sit_afterwards=True):
        KP = 40
        KD = 5
        self.robot.ReceiveObservation()
        init_motor_angle = np.array(self.robot.GetTrueMotorAngles())
        desired_motor_angle = self.robot.default_dof_pos.copy()
        print(init_motor_angle)
        for t in tqdm(range(400)):
            # print(self.robot.GetBaseOrientation())
            current_motor_angle = np.array(self.robot.GetTrueMotorAngles())
            self.robot.ReceiveObservation()
            blend_ratio = np.minimum(t / 250., 1)
            action = (1 - blend_ratio) * current_motor_angle + blend_ratio * desired_motor_angle
            self.robot.Step(action, Kp=KP, Kd=KD)
            time.sleep(0.02)
            # print("RPY:", self.robot.GetBaseRPY())
            # print("dof_pos:", self.robot.GetTrueMotorAngles())
            # warm up torch by doing inference
            # self.robot.ReceiveObservation()
            # obs = self.get_observation()
            # raw_actions = self.policy(torch.from_numpy(obs[None, :])).detach().numpy().squeeze()

        if sit_afterwards:
            for t in tqdm(range(900)):
                # print(robot.GetBaseOrientation())
                current_motor_angle = np.array(self.robot.GetTrueMotorAngles())
                self.robot.ReceiveObservation()
                blend_ratio = np.minimum(t / 800., 1)
                action = (1 - blend_ratio) * current_motor_angle + blend_ratio * init_motor_angle
                self.robot.Step(action, Kp=KP, Kd=KD)
                time.sleep(0.02)

if __name__ == "__main__":
    FREQ = 50
    STEP_TIME = 1/FREQ

    parser = argparse.ArgumentParser()
    parser.add_argument("--exptid", type=str)
    parser.add_argument("--kp", type=float)
    parser.add_argument("--kd", type=float)
    args = parser.parse_args()
    model_file = "models/" + args.exptid + "_jit.pt"

    hardware_pipeline = A1Policy(model_file, args.kp, args.kd)
    hardware_pipeline.stand_up(sit_afterwards=True)
    # hardware_pipeline.stand_up(sit_afterwards=False)
    exit()
    start = time.time()
    while True:
        # print(time.time()-start)
        start = time.time()  # 10e-6 s precision
        hardware_pipeline.step()
        duration = time.time() - start
        if duration < STEP_TIME:
            time.sleep(STEP_TIME - duration)
    
    # python3 walking.py --exptid rand_dis-1-delay-1 --kp 5 --kd 0.5