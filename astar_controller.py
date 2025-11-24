#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
from controller import Robot

# 让 from lib.xxx 能导入
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from lib.grid_map import GridMap
from lib.astar import a_star


class AStarController(Robot):
    TIME_STEP = 32
    WHEEL_RADIUS = 0.0205   # e-puck 轮半径（m）
    AXLE_LENGTH = 0.053     # e-puck 两轮间距（m）
    PATH_TOLERANCE = 0.30   # 路径点容差（m）

    def __init__(self):
        super().__init__()

        # ====== 电机 ======
        self.left_motor = self.getDevice("left wheel motor")
        self.right_motor = self.getDevice("right wheel motor")
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # ====== 编码器（PositionSensor） ======
        self.left_ps = self.getDevice("left wheel sensor")
        self.right_ps = self.getDevice("right wheel sensor")
        self.left_ps.enable(self.TIME_STEP)
        self.right_ps.enable(self.TIME_STEP)

        self._first_step = True
        self._prev_left = 0.0
        self._prev_right = 0.0

        # ====== 可选 IMU（没有就用里程计角度） ======
        try:
            self.imu = self.getDevice("inertial unit")
            self.imu.enable(self.TIME_STEP)
        except Exception:
            self.imu = None
            print("可选设备未找到：['inertial unit']")

        # ====== 地图和路径 ======
        self.cell_size = 0.2
        self.map = GridMap(width=20, height=20, cell_size=self.cell_size)
        self._add_basic_obstacles(self.map)

        # 固定的起点/终点演示 A*
        self.start = (2, 2)      # 左下偏里一点
        self.goal = (18, 18)     # 右上区域

        self.path = a_star(self.map.grid, self.start, self.goal)
        print("Path length:", len(self.path), "first 5:", self.path[:5])
        self.path_index = 0

        # 把里程计初始位置放在起点格中心
        self.pose_x, self.pose_z = self.map.grid_to_world(*self.start)
        self.pose_theta = 0.0

    # ---------- 地图障碍 ----------
    def _add_basic_obstacles(self, gridMap: GridMap):
        """
        在网格里造一堵中间有缺口的“竖直墙”，
        让从 (5,5) 走到 (35,30) 必须绕一下。
        """
        width = gridMap.width
        height = gridMap.height

        wall_x = width // 2       # 墙的 x 位置，大概是 20
        gap_start = 15            # 缺口开始的 y（j）
        gap_end = 20              # 缺口结束的 y（j）

        for j in range(0, height):
            # 中间留一个走廊
            if gap_start <= j <= gap_end:
                continue
            gridMap.set_obstacle(wall_x, j)

        # 额外再放一点斜着的障碍，让路线更弯
        for k in range(8):
            gridMap.set_obstacle(10 + k, 10 + k)

    # ---------- 角度归一化 ----------
    def _normalize_angle(self, a: float) -> float:
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a

    # ---------- 里程计 ----------
    def _update_odometry(self):
        left = self.left_ps.getValue()
        right = self.right_ps.getValue()
        print(f"encoders: L={left:.3f}, R={right:.3f}")

        if self._first_step:
            self._prev_left = left
            self._prev_right = right
            self._first_step = False
            return

        dl = (left - self._prev_left) * self.WHEEL_RADIUS
        dr = (right - self._prev_right) * self.WHEEL_RADIUS
        self._prev_left = left
        self._prev_right = right

        d_center = (dl + dr) / 2.0
        d_theta = (dr - dl) / self.AXLE_LENGTH

        self.pose_x += d_center * math.cos(self.pose_theta + 0.5 * d_theta)
        self.pose_z += d_center * math.sin(self.pose_theta + 0.5 * d_theta)
        self.pose_theta = self._normalize_angle(self.pose_theta + d_theta)

    # ---------- 当前朝向（IMU 优先，否则用里程计角度） ----------
    def get_yaw(self) -> float:
        if self.imu:
            roll, pitch, yaw = self.imu.getRollPitchYaw()
            return yaw
        return self.pose_theta

    # ---------- 朝当前路点运动 ----------
    def move_towards(self, wx: float, wz: float) -> bool:
        yaw = self.get_yaw()  # 当前朝向
        dx = wx - self.pose_x
        dz = wz - self.pose_z
        distance = math.hypot(dx, dz)  # 与目标点的距离
        target_theta = math.atan2(dz, dx)
        angle_error = self._normalize_angle(target_theta - yaw)

        # 距离阈值，达到就停止
        if distance < 0.10:
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)
            return True

        # 轮子最大角速度（Webots e-puck 常见上限）
        MAX_WHEEL_SPEED = 6.28  # rad/s
        # 线速度/角速度限幅（物理单位）
        MAX_V = 0.10  # m/s
        MAX_W = 2.0   # rad/s

        # Curvature drive 参数；可按实际微调
        k_lin = 1.5   # 线速度增益（距离越大越快前进）
        k_ang = 2.0   # 角速度增益

        # 基于距离与角度误差计算 v/w，并限幅
        v = max(-MAX_V, min(MAX_V, k_lin * distance))
        w = max(-MAX_W, min(MAX_W, k_ang * angle_error))

        # 将线速度/角速度转换为左右轮角速度（rad/s）
        omega_L = (v - w * (self.AXLE_LENGTH / 2.0)) / self.WHEEL_RADIUS
        omega_R = (v + w * (self.AXLE_LENGTH / 2.0)) / self.WHEEL_RADIUS

        # 限制到电机可达范围
        omega_L = max(-MAX_WHEEL_SPEED, min(MAX_WHEEL_SPEED, omega_L))
        omega_R = max(-MAX_WHEEL_SPEED, min(MAX_WHEEL_SPEED, omega_R))

        # 调试打印
        print(
            f"pose=({self.pose_x:.3f},{self.pose_z:.3f},{self.pose_theta:.3f}) "
            f"target=({wx:.3f},{wz:.3f}) angle_err={angle_error:.3f} dist={distance:.3f} "
            f"cmd=(ωL={omega_L:.2f}, ωR={omega_R:.2f})"
        )

        self.left_motor.setVelocity(omega_L)
        self.right_motor.setVelocity(omega_R)
        return False

    # ---------- 主循环 ----------
    def run(self):
        while self.step(self.TIME_STEP) != -1:
            self._update_odometry()

            # 先把距离当前路点很近的点都跳过去，避免卡在边界
            while self.path_index < len(self.path):
                i, j = self.path[self.path_index]
                wx, wz = self.map.grid_to_world(i, j)
                dist = math.hypot(wx - self.pose_x, wz - self.pose_z)
                if dist <= self.PATH_TOLERANCE:
                    self.path_index += 1
                else:
                    break

            if self.path_index < len(self.path):
                i, j = self.path[self.path_index]
                wx, wz = self.map.grid_to_world(i, j)
                reached = self.move_towards(wx, wz)
                if reached:
                    self.path_index += 1
            else:
                self.left_motor.setVelocity(0.0)
                self.right_motor.setVelocity(0.0)
                print("Path finished")
                break


def main():
    controller = AStarController()
    controller.run()


if __name__ == "__main__":
    main()
