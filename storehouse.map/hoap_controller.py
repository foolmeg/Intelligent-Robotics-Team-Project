from controller import Supervisor
import math


class HoapWalker:
    def __init__(self):
        # 用 Supervisor 方便读取自身位置
        self.robot = Supervisor()
        self.time_step = int(self.robot.getBasicTimeStep())
        self.dt = self.time_step / 1000.0

        # 逻辑关节名 -> Webots 设备名
        motor_name_map = {
            # 腰
            'WaistPitch':      'body_joint_1',

            # 左腿
            'LHipYaw':         'lleg_joint_1',
            'LHipRoll':        'lleg_joint_2',
            'LHipPitch':       'lleg_joint_3',
            'LKneePitch':      'lleg_joint_4',
            'LAnklePitch':     'lleg_joint_5',
            'LAnkleRoll':      'lleg_joint_6',

            # 右腿
            'RHipYaw':         'rleg_joint_1',
            'RHipRoll':        'rleg_joint_2',
            'RHipPitch':       'rleg_joint_3',
            'RKneePitch':      'rleg_joint_4',
            'RAnklePitch':     'rleg_joint_5',
            'RAnkleRoll':      'rleg_joint_6',

            # 左臂
            'LShoulderPitch':  'larm_joint_1',
            'LShoulderRoll':   'larm_joint_2',
            'LElbowYaw':       'larm_joint_3',
            'LElbowPitch':     'larm_joint_4',  # 极限很小
            'LWristYaw':       'larm_joint_5',

            # 右臂
            'RShoulderPitch':  'rarm_joint_1',
            'RShoulderRoll':   'rarm_joint_2',
            'RElbowYaw':       'rarm_joint_3',
            'RElbowPitch':     'rarm_joint_4',  # 极限很小
            'RWristYaw':       'rarm_joint_5',

            # 头
            'HeadYaw':         'head_joint_1',
            'HeadPitch':       'head_joint_2',
        }

        self.motors = {}
        self.limits = {}

        for logical_name, device_name in motor_name_map.items():
            motor = self.robot.getDevice(device_name)
            self.motors[logical_name] = motor

            max_vel = motor.getMaxVelocity()
            # 腿用 30% 速度，其它 15%（比较慢，保证柔和）
            if 'Hip' in logical_name or 'Knee' in logical_name or 'Ankle' in logical_name:
                motor.setVelocity(max_vel * 0.3)
            else:
                motor.setVelocity(max_vel * 0.15)

            min_pos = motor.getMinPosition()
            max_pos = motor.getMaxPosition()
            self.limits[logical_name] = (min_pos, max_pos)

            print(f"[INFO] Ready motor: {logical_name} -> {device_name}, "
                  f"min={min_pos:.4f}, max={max_pos:.4f}")

        print("[INFO] HoapWalker initialized (extra safe joint limits).")

        # 读取初始 X，用于 5 m 范围内往返
        self.self_node = self.robot.getSelf()
        self.translation_field = self.self_node.getField('translation')
        pos = self.translation_field.getSFVec3f()
        self.start_x = pos[0]
        self.max_offset = 2.5  # ±2.5 m
        self.direction = 1
        print(f"[INFO] Start X = {self.start_x}, walking within ±{self.max_offset} m along X.")

        # 步态相位
        self.phase = 0.0
        self.step_freq = 0.15  # Hz，比之前的 0.05 更快一些
        self.omega = 2.0 * math.pi * self.step_freq

        # 先站稳再走
        self.init_stand_time = 3.0   # 秒
        self.init_stand_steps = int(self.init_stand_time / self.dt)
        self.step_counter = 0

        # 记录开始走路的步数，用于渐变增大步幅
        self.walk_start_step = None

    # 安全设置关节角：自动夹在 min/max 之间
    def set_joint(self, name, target_angle):
        motor = self.motors.get(name)
        if motor is None:
            return

        min_pos, max_pos = self.limits[name]

        if min_pos is not None and max_pos is not None:
            if target_angle < min_pos:
                target_angle = min_pos
            if target_angle > max_pos:
                target_angle = max_pos

        motor.setPosition(target_angle)

    # 初始站姿：非常接近直立，只稍微弯一点膝盖
    def stand_posture(self):
        stand_angles = {
            'WaistPitch': 0.0,

            'LHipYaw': 0.0, 'RHipYaw': 0.0,
            'LHipRoll': 0.0, 'RHipRoll': 0.0,

            # 轻微下蹲
            'LHipPitch': -0.08,
            'RHipPitch': -0.08,
            'LKneePitch': 0.16,
            'RKneePitch': 0.16,
            'LAnklePitch': -0.08,
            'RAnklePitch': -0.08,
            'LAnkleRoll': 0.0,
            'RAnkleRoll': 0.0,

            # 手臂 / 头：保持接近 0，不碰肘 Pitch
            'LShoulderPitch': 0.0,
            'RShoulderPitch': 0.0,
            'LShoulderRoll': 0.0,
            'RShoulderRoll': 0.0,
            'LElbowYaw': 0.0,
            'RElbowYaw': 0.0,
            'LElbowPitch': 0.0,
            'RElbowPitch': 0.0,
            'LWristYaw': 0.0,
            'RWristYaw': 0.0,
            'HeadYaw': 0.0,
            'HeadPitch': 0.0,
        }

        for name, angle in stand_angles.items():
            self.set_joint(name, angle)

    def update_direction_by_x(self):
        pos = self.translation_field.getSFVec3f()
        x = pos[0]
        offset = x - self.start_x

        if offset > self.max_offset:
            self.direction = -1
            print(f"[INFO] Reversing direction to -X (x={x:.3f})")
        elif offset < -self.max_offset:
            self.direction = 1
            print(f"[INFO] Reversing direction to +X (x={x:.3f})")

    def walk_step(self):
        # 根据位置决定是否掉头
        self.update_direction_by_x()

        # 更新相位
        self.phase += self.direction * self.omega * self.dt

        # 左右腿相位差 180°
        phase_L = self.phase
        phase_R = self.phase + math.pi

        sL = math.sin(phase_L)
        sR = math.sin(phase_R)
        s = math.sin(self.phase)  # 用于左右重心摆动

        # === 渐变增大步幅：前 2 秒从 0 慢慢涨到 1 ===
        if self.walk_start_step is not None:
            t_walk = (self.step_counter - self.walk_start_step) * self.dt
        else:
            t_walk = 0.0
        gain = min(1.0, t_walk / 2.0)   # 2 秒内从 0 -> 1

        # === 基础姿态（接近直立） ===
        base_hip   = -0.05
        base_knee  =  0.10
        base_ankle = -0.05

        # === 腿部步态参数 ===
        step_amp_hip_base   = 0.06    # 可以再往上调到 0.08 试
        step_amp_knee_base  = 0.10    # 可以再往上调到 0.14 试
        roll_amp_base       = 0.025   # 可以再往上调到 0.03 试

        step_amp_hip  = step_amp_hip_base  * gain
        step_amp_knee = step_amp_knee_base * gain
        roll_amp      = roll_amp_base      * gain

        # 左腿
        hip_L = base_hip + step_amp_hip * sL
        knee_L = base_knee + step_amp_knee * max(0.0, sL)
        ankle_L = base_ankle - 0.5 * (knee_L - base_knee)

        # 右腿
        hip_R = base_hip + step_amp_hip * sR
        knee_R = base_knee + step_amp_knee * max(0.0, sR)
        ankle_R = base_ankle - 0.5 * (knee_R - base_knee)

        # 写回腿部关节
        self.set_joint('LHipPitch', hip_L)
        self.set_joint('RHipPitch', hip_R)
        self.set_joint('LKneePitch', knee_L)
        self.set_joint('RKneePitch', knee_R)
        self.set_joint('LAnklePitch', ankle_L)
        self.set_joint('RAnklePitch', ankle_R)

        # 轻微左右侧倾
        self.set_joint('LHipRoll',  roll_amp * s)
        self.set_joint('RHipRoll', -roll_amp * s)
        self.set_joint('LAnkleRoll', -roll_amp * s)
        self.set_joint('RAnkleRoll', roll_amp * s)

        # 腰保持竖直
        self.set_joint('WaistPitch', 0.0)

        # === 手臂摆动（和对侧腿反相） ===
        # 使用和腿同一个 gain，步伐变大时手臂摆动也慢慢变大
        arm_amp_pitch_base = 0.20   # 手臂前后摆幅，视情况可以调到 0.25
        arm_amp_roll_base  = 0.05   # 轻微侧摆

        arm_amp_pitch = arm_amp_pitch_base * gain
        arm_amp_roll  = arm_amp_roll_base  * gain

        # 这里用腿的相位做一个简单但能看的摆臂：
        # （大致效果：左腿向前时，右臂向前，左臂向后）
        LShoulderPitch = -arm_amp_pitch * sL   # sL>0 时左腿往“一个方向”摆，手臂反相
        RShoulderPitch = -arm_amp_pitch * sR

        # 轻微左右展开一点点，配合重心摆动
        LShoulderRoll =  arm_amp_roll * s
        RShoulderRoll = -arm_amp_roll * s

        self.set_joint('LShoulderPitch', LShoulderPitch)
        self.set_joint('RShoulderPitch', RShoulderPitch)
        self.set_joint('LShoulderRoll',  LShoulderRoll)
        self.set_joint('RShoulderRoll',  RShoulderRoll)

        # 肘只保持在 0，避免触碰到极小的 Pitch 限制
        self.set_joint('LElbowYaw', 0.0)
        self.set_joint('RElbowYaw', 0.0)
        self.set_joint('LElbowPitch', 0.0)
        self.set_joint('RElbowPitch', 0.0)
        self.set_joint('LWristYaw', 0.0)
        self.set_joint('RWristYaw', 0.0)

        # 头保持不动
        self.set_joint('HeadYaw', 0.0)
        self.set_joint('HeadPitch', 0.0)

    def run(self):
        print("[INFO] HoapWalker faster gait controller with arm swing started.")

        while self.robot.step(self.time_step) != -1:
            if self.step_counter < self.init_stand_steps:
                # 先站稳
                self.stand_posture()
            else:
                # 记录刚开始走路的 step，用来算行走时间
                if self.walk_start_step is None:
                    self.walk_start_step = self.step_counter
                    print("[INFO] Start walking...")

                # 开始行走
                self.walk_step()

            self.step_counter += 1


if __name__ == "__main__":
    walker = HoapWalker()
    walker.run()
