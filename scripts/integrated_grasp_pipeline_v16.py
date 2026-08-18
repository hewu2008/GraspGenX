import os
import sys
import time
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as R, Slerp
import threading
import queue

# ================= 导入 SDK 及环境配置 =================
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)

# 机器人 SDK
from lib.lib_h1_sdk_python import (
    H1Robot, MotorControlMode, EtherCAT_Motor_Index,
    ArmAction, ArmPose, ArmEndPose, Motor_Control
)

# 相机 gRPC 客户端
from camera_client import CameraClient

# ---------------------------------------------------------
# 直接复用 zerith_client.py 中的客户端及完整处理函数。
# 客户端代码保持不变；Integrated 只负责调用并消费输出位姿。
# ---------------------------------------------------------
from zerith.zerith_client import (
    create_client,
    detect_parts,
    process_label,
    save_detection_debug,
)

# ================= 系统控制参数 =================
RATE_HZ = 500             # 底层控制频率 500Hz
DT = 1.0 / RATE_HZ        # 控制周期 0.002s

# 腰部放料动作参数：正常高度 0.67m，松爪前下降到 0.57m。
WAIST_NORMAL_Z = 0.67
WAIST_RELEASE_Z = WAIST_NORMAL_Z - 0.18
WAIST_PITCH = 1.2
WAIST_MOVE_DURATION = 2.0
GRIPPER_RELEASE_WAIT = 2.0  # 松爪后等待夹爪真正打开，再恢复腰部

# FoundationPose & Zerith 配置
GRPC_TARGET = "localhost:50051"
CAMERA_NAME = "rs/cam_high"
#ZMQ_SERVER_ADDR = "tcp://172.31.200.245:5555"
ZMQ_SERVER_ADDR = "tcp://192.168.3.28:5555"
CLIENT_DEBUG_DIR = "./client_debug"
REGISTER_ITERATIONS = 5
RETRY_COUNT = 3600

# zerith_client 会把每个检测实例保存为：
# client_debug/<category_id>_<instance_index>/ob_in_cam/0.txt
#
# 根据当前服务端类别映射，默认：
#   左手抓取 cat2 的第 0 个实例
#   右手抓取 cat4 的第 0 个实例
# 若服务端类别映射不同，只需修改下面四个值，不需要改检测逻辑。
LEFT_TARGET_CATEGORY_ID = "cat3"
LEFT_TARGET_INSTANCE_INDEX = [0, 1, 2]
RIGHT_TARGET_CATEGORY_ID = "cat2"
RIGHT_TARGET_INSTANCE_INDEX = [0, 1, 2]

K_COLOR = np.array([
    [607.62, 0.00, 329.68],
    [0.00, 608.40, 243.36],
    [0.00, 0.00, 1.00],
], dtype=np.float64)
# ================= 1. 机器人姿态准备模块 =================
def prepare_robot_posture(robot, cur_waist_z, cur_waist_pitch, tar_waist_z, tar_waist_pitch):
    print("\n[流程 A] 正在调整机器人初始观测姿态...")
    waist_steps = int(3.0 * RATE_HZ)

    waist_pose = ArmPose()
    waist_pose.x = waist_pose.y = 0.0
    waist_pose.z = cur_waist_z
    waist_pose.roll = 0.0
    waist_pose.pitch = cur_waist_pitch
    waist_pose.yaw = 0.0

    diff_z = tar_waist_z - cur_waist_z
    diff_pitch = tar_waist_pitch - cur_waist_pitch

    def send():
        end = robot.armPoseToArmEndPose(waist_pose)
        robot.setWaist_high(end)

    for _ in range(1, waist_steps + 1):
        waist_pose.z += diff_z / waist_steps
        send(); time.sleep(DT)    
    
    for _ in range(1, waist_steps + 1):
        waist_pose.pitch += diff_pitch / waist_steps
        send(); time.sleep(DT)

    time.sleep(1.5)

    # 动作 2：头部俯视
    # print(" -> 2/2 头部云台俯视中...")
    # head_pitch = Motor_Control()
    # head_pitch.Position = 0.0
    # target_pitch_angle = -0.2  
    # head_steps = int(1.5 * RATE_HZ) 
    
    # for i in range(1, head_steps + 1):
    #     head_pitch.Position = target_pitch_angle * (i / head_steps)
    #     robot.setHead_high(EtherCAT_Motor_Index.MOTOR_HEAD_UP, head_pitch)
    #     time.sleep(DT)
        
    # time.sleep(2.0)
    # print(" -> 观测姿态调整完毕。")

# def arm_move_pre(robot, cur_xyz, cur_quat, dest_xyz, dest_quat):
#     print("\n[流程 E] 控制手臂从相对零点平滑逼近目标...")
#     duration_arm = 4.0
#     steps_arm = int(duration_arm * RATE_HZ)

#     start_x, start_y, start_z = cur_xyz[0], cur_xyz[1], cur_xyz[2]
#     start_quat = cur_quat

#     dest_x, dest_y, dest_z = dest_xyz[0], dest_xyz[1], dest_xyz[2] 

#     key_rots = R.from_quat([start_quat, dest_quat])
#     slerp = Slerp([0, 1], key_rots)

#     for i in range(1, steps_arm + 1):
#         ratio = i / steps_arm

#         x = start_x + (dest_x - start_x) * ratio
#         y = start_y + (dest_y - start_y) * ratio
#         z = start_z + (dest_z - start_z) * ratio

#         interp_quat = slerp(ratio).as_quat()

#         target_end_pose = ArmEndPose()
#         target_end_pose.position = [x, y, z]
#         target_end_pose.rotation = [interp_quat[0], interp_quat[1], interp_quat[2], interp_quat[3]]

#         robot.setArm_high(ArmAction.LEFT_ARM, target_end_pose)
#         robot.setArm_high(ArmAction.RIGHT_ARM, target_end_pose)
#         time.sleep(DT)

#     time.sleep(0.5)
#     print(" -> 已平滑到达目标点！")

def _move_arm(robot, arm, start_xyz, start_quat, dest_xyz, dest_quat,
              duration=3.0, rate=RATE_HZ, dt=DT):
    """单条手臂的平滑插补，供线程调用"""
    steps = int(duration * rate)
    sx, sy, sz = start_xyz
    dx, dy, dz = dest_xyz

    key_rots = R.from_quat([start_quat, dest_quat])
    slerp = Slerp([0, 1], key_rots)

    for i in range(1, steps + 1):
        ratio = i / steps
        x = sx + (dx - sx) * ratio
        y = sy + (dy - sy) * ratio
        z = sz + (dz - sz) * ratio
        quat = slerp(ratio).as_quat()

        pose = ArmEndPose()
        pose.position = [x, y, z]
        pose.rotation = [quat[0], quat[1], quat[2], quat[3]]

        robot.setArm_high(arm, pose)
        time.sleep(dt)


def arm_move_pre(robot, cur_xyz, cur_quat, dest_xyz, dest_quat):
    print("\n[流程 E] 控制双臂从相对零点同时平滑逼近目标...")

    # 左右臂不同的目标位置（y 互为相反数）
    left_dest = [dest_xyz[0],  dest_xyz[1], dest_xyz[2]]
    right_dest = [dest_xyz[0], -dest_xyz[1], dest_xyz[2]]   # y 取反

    # 创建线程：左臂与右臂同时运动
    t_left = threading.Thread(
        target=_move_arm,
        args=(robot, ArmAction.LEFT_ARM, cur_xyz, cur_quat,
              left_dest, dest_quat, 2)
    )
    t_right = threading.Thread(
        target=_move_arm,
        args=(robot, ArmAction.RIGHT_ARM, cur_xyz, cur_quat,
              right_dest, dest_quat, 2)
    )

    t_left.start()
    t_right.start()

    # 等待两个线程都结束
    t_left.join()
    t_right.join()

    time.sleep(0.5)
    print(" -> 双臂已同时平滑到达各自目标！")

def arm_move_rec(robot, arm, dx, dy, dz):

    duration_arm = 2.0
    steps_arm = int(duration_arm * RATE_HZ)

    ok_arm, arm_state = robot.getHandRelative(arm)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)
    start_x, start_y, start_z = arm_pos_rel[0], arm_pos_rel[1], arm_pos_rel[2]
    start_quat = arm_quat_rel

    # dest_x, dest_y, dest_z = arm_pos_rel[0], arm_pos_rel[1]+0.2, arm_pos_rel[2]
    dest_x, dest_y, dest_z = arm_pos_rel[0]+dx, arm_pos_rel[1]+dy, arm_pos_rel[2]+dz
    dest_quat = arm_quat_rel
    #dest_quat = target_quat

    key_rots = R.from_quat([start_quat, dest_quat])
    slerp = Slerp([0, 1], key_rots)

    for i in range(1, steps_arm + 1):
        ratio = i / steps_arm

        x = start_x + (dest_x - start_x) * ratio
        y = start_y + (dest_y - start_y) * ratio
        z = start_z + (dest_z - start_z) * ratio

        interp_quat = slerp(ratio).as_quat()

        target_end_pose = ArmEndPose()
        target_end_pose.position = [x, y, z]
        target_end_pose.rotation = [interp_quat[0], interp_quat[1], interp_quat[2], interp_quat[3]]

        robot.setArm_high(arm, target_end_pose)
        time.sleep(DT)

    time.sleep(0.5)
    print(" -> 已平滑到达目标点！")

def arm_move_left(robot, target_pos, target_quat):

    ok_arm, arm_state = robot.getHandRelative(ArmAction.LEFT_ARM)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)

    temp_xyz = [arm_pos_rel[0]+target_pos[0]-0.10, arm_pos_rel[1]+target_pos[1]+0.02, arm_pos_rel[2]+target_pos[2]]
        
    #temp_xyz = [arm_pos_rel[0]+target_pos[0]-0.10, arm_pos_rel[1]+target_pos[1], arm_pos_rel[2]+target_pos[2]]
    temp_quat = [0.0000, 0.0, 0.0000, 1]
    _move_arm(robot, ArmAction.LEFT_ARM, arm_pos_rel, arm_quat_rel, temp_xyz, temp_quat, 2)

    time.sleep(0.5)

    ok_arm, arm_state = robot.getHandRelative(ArmAction.LEFT_ARM)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)
    
    temp_xyz = [arm_pos_rel[0], arm_pos_rel[1], arm_pos_rel[2]]
    temp_quat = target_quat
    _move_arm(robot, ArmAction.LEFT_ARM, arm_pos_rel, arm_quat_rel, temp_xyz, temp_quat, 1)

    time.sleep(0.5)

    ok_arm, arm_state = robot.getHandRelative(ArmAction.LEFT_ARM)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)

    dest_xyz = [arm_pos_rel[0]+0.05, arm_pos_rel[1], arm_pos_rel[2]-0.01]
    #dest_xyz = [arm_pos_rel[0]+0.03, arm_pos_rel[1], arm_pos_rel[2]-0.02]
    dest_quat = arm_quat_rel

    _move_arm(robot, ArmAction.LEFT_ARM, arm_pos_rel, arm_quat_rel, dest_xyz, dest_quat, 1)

    time.sleep(0.5)
    print(" -> 已平滑到达目标点！")

def arm_move_right(robot, target_pos, target_quat):
    ok_arm, arm_state = robot.getHandRelative(ArmAction.RIGHT_ARM)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)
    
    temp_xyz = [arm_pos_rel[0]+target_pos[0]-0.10, arm_pos_rel[1]+target_pos[1]+0.02, arm_pos_rel[2]+target_pos[2]]

    #temp_xyz = [arm_pos_rel[0]+target_pos[0]-0.10, arm_pos_rel[1]+target_pos[1], arm_pos_rel[2]+target_pos[2]]
    temp_quat = [0.0000, 0.0, 0.0000, 1]
    # temp_quat = [0.0000, 0.3827, 0.0000, 0.9239]
    _move_arm(robot, ArmAction.RIGHT_ARM, arm_pos_rel, arm_quat_rel, temp_xyz, temp_quat, 2)

    time.sleep(0.5)

    ok_arm, arm_state = robot.getHandRelative(ArmAction.RIGHT_ARM)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)
    
    temp_xyz = [arm_pos_rel[0], arm_pos_rel[1], arm_pos_rel[2]]
    temp_quat = target_quat
    # temp_quat = [0.0000, 0.3827, 0.0000, 0.9239]
    _move_arm(robot, ArmAction.RIGHT_ARM, arm_pos_rel, arm_quat_rel, temp_xyz, temp_quat, 1)

    time.sleep(0.5)

    ok_arm, arm_state = robot.getHandRelative(ArmAction.RIGHT_ARM)
    arm_pos_rel = getattr(arm_state, "position", None)
    arm_quat_rel = getattr(arm_state, "rotation", None)

    # dest_x, dest_y, dest_z = arm_pos_rel[0]+target_pos[0]-0.09, arm_pos_rel[1]+target_pos[1]+0.04, arm_pos_rel[2]+target_pos[2]-0.02
    dest_xyz = [arm_pos_rel[0]+0.05, arm_pos_rel[1], arm_pos_rel[2]-0.01]
    #dest_xyz = [arm_pos_rel[0]+0.03, arm_pos_rel[1], arm_pos_rel[2]-0.02]
    dest_quat = arm_quat_rel
    # dest_quat = target_quat

    _move_arm(robot, ArmAction.RIGHT_ARM, arm_pos_rel, arm_quat_rel, dest_xyz, dest_quat, 1)

    time.sleep(0.5)
    print(" -> 已平滑到达目标点！")

# ================= 2. RGB-D 获取模块 =================
def capture_rgbd_data():
    """获取 gRPC 图像并保存"""
    print(f"\n[流程 B] 正在连接相机服务获取 RGB-D 数据 ({GRPC_TARGET})...")
    client = CameraClient(grpc_target=GRPC_TARGET, enable_depth=True)
    client.start()
    
    rgb_path = "zerith_rgb.png"
    depth_path = "zerith_depth.npy"
    
    try:
        max_retries = 50
        for i in range(max_retries):
            depth_data = client.get_latest_depth(CAMERA_NAME)
            color_data = client.get_latest_frame(CAMERA_NAME) 
            
            if depth_data is not None and color_data is not None:
                depth_raw_mm, _ = depth_data
                color_raw, _ = color_data
                
                # 转换为算法需要的 float32 (米)
                depth_raw_m = depth_raw_mm.astype(np.float32) / 1000.0
                
                cv2.imwrite(rgb_path, color_raw)
                np.save(depth_path, depth_raw_m)
                print(f" -> 捕获成功！已保存 {rgb_path} 和 {depth_path}")
                return rgb_path, depth_path
                
            time.sleep(0.1)
        raise TimeoutError("获取图像超时！请检查 gRPC 节点。")
    finally:
        client.stop()

# ================= 3. 感知端逻辑（处理所有检测物体） =================
def build_pose_path(debug_dir, category_id, instance_index):
    """构造 zerith_client 保存的单个物体位姿文件路径。"""
    return os.path.join(
        debug_dir,
        f"{category_id}_{instance_index}",
        "ob_in_cam",
        "0.txt",
    )


def run_perception_client(rgb_path, depth_path, debug_dir=CLIENT_DEBUG_DIR, result_queue=None):
    """
    调用 zerith_client 的原有处理流程，对 Detection 返回的所有物体逐一注册。

    不根据 label 做筛选，也不额外写根目录下的 0.txt。每个成功物体的位姿由
    zerith_client.process_label 保存到：
        <debug_dir>/<category_id>_<instance_index>/ob_in_cam/0.txt

    当 result_queue 不为 None 时，每个物体注册成功后立即将
    (category_id, instance_index, pose_path) push 到队列中，
    全部处理完毕后再 push 一个 None 作为哨兵值。

    Returns:
        tuple[dict, dict] | None:
            若 result_queue 为 None，返回 (pose_files, category_counts)；
            否则返回 None（结果通过队列传递）。
    """
    print("\n[流程 C] 请求 Zerith 对所有检测物体执行 Detection + Register...")
    os.makedirs(debug_dir, exist_ok=True)

    client = create_client(ZMQ_SERVER_ADDR)
    if client is None:
        print(" -> [错误] Zerith 服务端连接失败。")
        if result_queue is not None:
            result_queue.put(None)
        return ({}, {}) if result_queue is None else None

    pose_files = {}

    try:
        # 1. Detection：直接使用 zerith_client.detect_parts，保留全部检测结果。
        color, boxes = detect_parts(client, rgb_path)
        if color is None or boxes is None:
            print(" -> [错误] Detection 调用失败。")
            if result_queue is not None:
                result_queue.put(None)
            return ({}, {}) if result_queue is None else None

        if len(boxes) == 0:
            print(" -> [警告] 当前画面未检测到任何物体。")
            if result_queue is not None:
                result_queue.put(None)
            return ({}, {}) if result_queue is None else None

        save_detection_debug(debug_dir, color, boxes)
        print(f" -> Detection 返回 {len(boxes)} 个物体，开始逐个 Register。")

        # 2. 读取与 zerith_client.main 相同的深度数据。
        depth = np.load(depth_path)

        # 同一 category_id 可能出现多个实例，编号规则与 zerith_client.main 一致。
        category_counts = {}

        for detection_index, box_dict in enumerate(boxes):
            label = box_dict["label"]
            category_id = str(box_dict["category_id"])

            instance_index = category_counts.get(category_id, 0)
            category_counts[category_id] = instance_index + 1

            box = [
                int(box_dict["x1"]),
                int(box_dict["y1"]),
                int(box_dict["x2"]),
                int(box_dict["y2"]),
            ]

            label_output_dir = os.path.join(
                debug_dir,
                f"{category_id}_{instance_index}",
            )

            print(
                f" -> [{detection_index + 1}/{len(boxes)}] "
                f"处理 {category_id}_{instance_index}: {label}"
            )

            # process_label 内部会调用 client.register，并将结果保存到
            # <label_output_dir>/ob_in_cam/0.txt。
            success = process_label(
                client=client,
                K=K_COLOR,
                color=color,
                depth=depth,
                label=label,
                category_id=category_id,
                box=box,
                mesh_bbox=None,
                to_origin=None,
                label_output_dir=label_output_dir,
                register_iterations=REGISTER_ITERATIONS,
                show=False,
            )

            if not success:
                print(f" -> [警告] {category_id}_{instance_index} 注册失败，继续处理其他物体。")
                continue

            pose_path = build_pose_path(
                debug_dir,
                category_id,
                instance_index,
            )

            if not os.path.isfile(pose_path):
                print(f" -> [警告] 注册返回成功，但未找到位姿文件: {pose_path}")
                continue

            pose_files[(category_id, instance_index)] = pose_path
            print(f" -> 位姿已保存: {pose_path}")

            if result_queue is not None:
                result_queue.put((category_id, instance_index, pose_path))

        print(f"\n -> 本轮共成功获得 {len(pose_files)} 个物体位姿：")
        for (category_id, instance_index), pose_path in pose_files.items():
            print(f"    {category_id}_{instance_index}: {pose_path}")

        if result_queue is not None:
            result_queue.put(None)
        else:
            return pose_files, category_counts

    except Exception as e:
        print(f" -> [异常] 感知端运行崩溃: {e}")
        if result_queue is not None:
            result_queue.put(None)
        else:
            return ({}, {})

    finally:
        client.close()


# ================= 4. 抓取运算与执行模块 =================
def load_pose_matrix_pre(filepath):
    matrix = []
    with open(filepath, 'r') as f:
        for line in f:
            row = [float(x) for x in line.strip().split()]
            if row: matrix.append(row)
    return np.array(matrix)

def load_pose_matrix(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    matrix_lines = [line.strip().split() for line in lines[:4]]
    pose = np.array(matrix_lines, dtype=np.float64).reshape(4,4)
    angle = float(lines[4].strip())
    return pose, angle

def calculate_target_relative_pose(cam_pos_rel, cam_quat_rel, arm_pos_rel, arm_quat_rel, T_obj_cam, flag):
    """坐标系变换：感知位姿 -> 手臂相对零点目标位姿"""
    T1 = np.eye(4)
    T1[:3, :3] = R.from_quat(cam_quat_rel).as_matrix()
    T1[:3, 3] = cam_pos_rel

    T2 = np.eye(4)
    T2[:3, :3] = R.from_euler('xyz', [-1.7802, 0.0, -1.5708], degrees=False).as_matrix()
    T2[:3, 3] = [0.2194, 0.0325, 0.6075]

    T3 = np.eye(4)
    if flag == 0:
        T3[:3, 3] = [-0.5743, -0.1800, -0.1208]
    else:
        T3[:3, 3] = [-0.5743, 0.1800, -0.1208]

    T4_inv = np.eye(4)
    T4_inv[:3, :3] = R.from_quat(arm_quat_rel).as_matrix()
    T4_inv[:3, 3] = arm_pos_rel
    T_comp = np.eye(4)
    
    T4_inv = np.dot(T4_inv, T_comp)
    T4 = np.linalg.inv(T4_inv)

    T_obj_in_arm = np.dot(T4, np.dot(T3, np.dot(T2, np.dot(T1, T_obj_cam))))

    T_grasp_local = np.eye(4)
    T_grasp_local[:3, :3] = R.from_euler('xyz', [0.0, 0.0, 0.0], degrees=True).as_matrix()
    T_grasp_local[:3, 3] = [0.0, 0.0, 0.0] 

    T_final = np.dot(T_obj_in_arm, T_grasp_local)

    target_pos = T_final[:3, 3]
    target_quat = R.from_matrix(T_final[:3, :3]).as_quat()

    return target_pos, target_quat

def chassis_move(robot: H1Robot, dist):
    DT = 0.2  
    SPEED = 0.2  
    DISTANCE = abs(dist)
    direction = 1 if dist >= 0 else -1
    speed = SPEED * direction
    DURATION = DISTANCE / SPEED  
    
    print(f"[Chassis] 开始前进: 速度={SPEED}m/s, 距离={DISTANCE}m, 持续{DURATION}s")
    start_time = time.time()
    
    try:
        while time.time() - start_time < DURATION:
            loop_start = time.perf_counter()
            robot.setChassis_high(speed, 0.0)  
            
            elapsed_time = time.time() - start_time
            remaining_distance = DISTANCE - (SPEED * elapsed_time)
            print(f"[Chassis] 已过{elapsed_time:.2f}s, 剩余距离{remaining_distance:.2f}m")
            
            elapsed = time.perf_counter() - loop_start
            sleep_time = DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        robot.setChassis_high(0.0, 0.0)
        print(f"[Chassis] 已到达目标位置，前进{DISTANCE}米完成")
                
    except KeyboardInterrupt:
        print("\n[Chassis] 手动停止")
        robot.setChassis_high(0.0, 0.0)

def rotate_90_degrees(robot: H1Robot, direction="left"):
    DT = 0.2
    ANGULAR_SPEED = 0.5  
    TARGET_ANGLE = 1.5708  
    DURATION = TARGET_ANGLE / ANGULAR_SPEED + 1.2  
    
    if direction == "left":
        yaw_speed = ANGULAR_SPEED  
    else:
        yaw_speed = -ANGULAR_SPEED  
    
    print(f"[Chassis] 旋转90度{direction}, 角速度{ANGULAR_SPEED}rad/s, 持续{DURATION:.2f}s")
    start_time = time.time()
    
    try:
        while time.time() - start_time < DURATION:
            loop_start = time.perf_counter()
            robot.setChassis_high(0.0, yaw_speed)
            
            elapsed = time.time() - start_time
            rotated_degrees = (ANGULAR_SPEED * elapsed) * 180 / 3.14159
            print(f"[Chassis] 已旋转: {rotated_degrees:.1f}°")
            
            elapsed_loop = time.perf_counter() - loop_start
            sleep_time = DT - elapsed_loop
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        robot.setChassis_high(0.0, 0.0)
        print(f"[Chassis] 旋转90度完成")
                
    except KeyboardInterrupt:
        print("\n[Chassis] 中断")
        robot.setChassis_high(0.0, 0.0)

# ================= 主控制流 =================

def move_waist_z(robot, start_z, target_z, duration=WAIST_MOVE_DURATION):
    """腰部 Z 方向平滑移动。

    当前流程会在每组抓取结束后恢复到 WAIST_NORMAL_Z，因此这里使用已知的
    start_z 和 target_z，不依赖 SDK 中不存在的腰部状态读取接口。
    """
    steps = max(1, int(duration * RATE_HZ))

    waist_pose = ArmPose()
    waist_pose.x = 0.0
    waist_pose.y = 0.0
    waist_pose.z = start_z
    waist_pose.roll = 0.0
    waist_pose.pitch = WAIST_PITCH
    waist_pose.yaw = 0.0

    print(f"[腰部] Z: {start_z:.2f}m -> {target_z:.2f}m")
    for i in range(1, steps + 1):
        ratio = i / steps
        waist_pose.z = start_z + (target_z - start_z) * ratio
        waist_end_pose = robot.armPoseToArmEndPose(waist_pose)
        robot.setWaist_high(waist_end_pose)
        time.sleep(DT)

    time.sleep(0.5)


def _wait_all_events(events, description, timeout=120.0):
    """等待一组线程事件全部完成，避免无限阻塞。"""
    deadline = time.time() + timeout
    for event in events:
        remaining = deadline - time.time()
        if remaining <= 0 or not event.wait(timeout=remaining):
            raise TimeoutError(f"等待{description}超时")


def wait_for_waist_before_release(robot, thread_index, sync_state):
    """所有机械臂到达放料位后，仅由 thread_0 下降腰部。"""
    thread_id = f"thread_{thread_index}"

    # 当前机械臂已经到达放料位置。
    sync_state["ready_events"][thread_index].set()
    print(f"[{thread_id}] 已到达放料位")

    if thread_index == 0:
        print(f"[{thread_id}] 等待本组所有机械臂到达放料位...")
        _wait_all_events(sync_state["ready_events"], "所有机械臂到达放料位")

        print(f"[{thread_id}] 松爪前执行腰部下降 0.10m")
        move_waist_z(robot, WAIST_NORMAL_Z, WAIST_RELEASE_Z)
        sync_state["waist_down_event"].set()
        return

    print(f"[{thread_id}] 等待 thread_0 完成腰部下降...")
    if not sync_state["waist_down_event"].wait(timeout=120.0):
        raise TimeoutError(f"{thread_id} 等待腰部下降超时，取消松爪")
    print(f"[{thread_id}] 腰部下降完成，允许松爪")


def restore_waist_before_arm_recovery(robot, thread_index, sync_state):
    """两只夹爪都松开后，由 thread_0 恢复腰部，之后才允许手臂撤回。"""
    thread_id = f"thread_{thread_index}"

    # 调用本函数前已经发送松爪指令，并等待夹爪完成打开。
    sync_state["released_events"][thread_index].set()
    print(f"[{thread_id}] 夹爪已松开")

    if thread_index == 0:
        print(f"[{thread_id}] 等待本组所有夹爪完成松开...")
        _wait_all_events(sync_state["released_events"], "所有夹爪完成松开")

        print(f"[{thread_id}] 两只手已放下零件，先恢复腰部到 {WAIST_NORMAL_Z:.2f}m")
        move_waist_z(robot, WAIST_RELEASE_Z, WAIST_NORMAL_Z)
        sync_state["waist_restored_event"].set()
        return

    print(f"[{thread_id}] 等待 thread_0 恢复腰部...")
    if not sync_state["waist_restored_event"].wait(timeout=120.0):
        raise TimeoutError(f"{thread_id} 等待腰部恢复超时，禁止手臂撤回")
    print(f"[{thread_id}] 腰部已恢复，允许手臂撤回")

def grasp_by_left(robot, target_pos, target_quat, thread_index, sync_state):
    # ------------------------------------------------
    # 4. 读取手臂/相机的当前状态并解算最终目标点
    # ------------------------------------------------
    print(f" -> 🎯 解算目标相对平移: X={target_pos[0]:.4f}, Y={target_pos[1]:.4f}, Z={target_pos[2]:.4f}")

    # ------------------------------------------------
    # 5. 执行手臂平移逼近目标
    # ------------------------------------------------
    print("\n[流程 E] 控制手臂从相对零点平滑逼近目标...")

    arm_move_left(robot, target_pos, target_quat)

    # ------------------------------------------------
    # 6. 闭合夹爪进行抓取
    # ------------------------------------------------
    print("\n[流程 F] 正在闭合左手夹爪进行抓取...")
    close_cmd = Motor_Control()
    close_cmd.Position = 1.5
    robot.setGripper_high(EtherCAT_Motor_Index.MOTOR_LEFT_ARM_8, close_cmd)

    time.sleep(2.0)

    arm_move_rec(robot, ArmAction.LEFT_ARM, -0.2, 0, 0.05)
    #prepare_robot_posture(robot, 0.67, 1.2, 0.75, 1.0)
    #arm_move_rec(robot, ArmAction.LEFT_ARM, 0, 0.25, 0)
    ok_arm, arm_state = robot.getHandRelative(ArmAction.LEFT_ARM)
    arm_pos_rel = getattr(arm_state, "position", None) 
    arm_quat_rel = getattr(arm_state, "rotation", None)
    _move_arm(robot, ArmAction.LEFT_ARM, arm_pos_rel, arm_quat_rel, [0.0, 0.30, 0.30], [0, 0, 0, 1], 1)
    time.sleep(1.0)
    #arm_move_rec(robot, ArmAction.LEFT_ARM, 0.2, 0, -0.1)
    ok_arm, arm_state = robot.getHandRelative(ArmAction.LEFT_ARM)
    arm_pos_rel = getattr(arm_state, "position", None) 
    arm_quat_rel = getattr(arm_state, "rotation", None)
    _move_arm(robot, ArmAction.LEFT_ARM, arm_pos_rel, arm_quat_rel, [0.17, 0.30, 0.30], [0, 0, 0, 1], 1)
    time.sleep(1.0)

    # 松爪前：只有 thread_0 下降腰部；thread_1 等待 thread_0 完成。
    wait_for_waist_before_release(robot, thread_index, sync_state)

    time.sleep(1.0)
    close_cmd.Position = 0.0
    robot.setGripper_high(EtherCAT_Motor_Index.MOTOR_LEFT_ARM_8, close_cmd)
    time.sleep(GRIPPER_RELEASE_WAIT)

    # 必须先等双手都松开并恢复腰部，之后才能执行手臂撤回动作。
    restore_waist_before_arm_recovery(robot, thread_index, sync_state)

    ok_arm, arm_state = robot.getHandRelative(ArmAction.LEFT_ARM)
    arm_pos_rel = getattr(arm_state, "position", None) 
    arm_quat_rel = getattr(arm_state, "rotation", None)
    _move_arm(robot, ArmAction.LEFT_ARM, arm_pos_rel, arm_quat_rel, [0.0, 0.30, 0.30], [0, 0, 0, 1], 1)
    time.sleep(1.0)

    ok_arm, arm_state = robot.getHandRelative(ArmAction.LEFT_ARM)
    arm_pos_rel = getattr(arm_state, "position", None) 
    arm_quat_rel = getattr(arm_state, "rotation", None)
    _move_arm(robot, ArmAction.LEFT_ARM, arm_pos_rel, arm_quat_rel, [-0.1, 0.0, 0.30], [0, 0, 0, 1], 1)
    time.sleep(1.0)
    # prepare_robot_posture(robot, 0.75, 1.0, 0.67, 1.2)
    # time.sleep(1.0)


def grasp_by_right(robot, target_pos, target_quat, thread_index, sync_state):
    print(f" -> 🎯 解算目标相对平移: X={target_pos[0]:.4f}, Y={target_pos[1]:.4f}, Z={target_pos[2]:.4f}")

    #grasp
    print("\n[流程 E] 控制手臂从相对零点平滑逼近目标...")
    arm_move_right(robot, target_pos, target_quat)

    print("\n[流程 F] 正在闭合左手夹爪进行抓取...")
    close_cmd = Motor_Control()
    close_cmd.Position = 1.5
    robot.setGripper_high(EtherCAT_Motor_Index.MOTOR_RIGHT_ARM_8, close_cmd)

    time.sleep(2.0)

    arm_move_rec(robot, ArmAction.RIGHT_ARM, -0.2, 0, 0.05)
    #prepare_robot_posture(robot, 0.67, 1.2, 0.75, 1.0)
    #arm_move_rec(robot, ArmAction.RIGHT_ARM, 0, -0.25, 0)
    ok_arm, arm_state = robot.getHandRelative(ArmAction.RIGHT_ARM)
    arm_pos_rel = getattr(arm_state, "position", None) 
    arm_quat_rel = getattr(arm_state, "rotation", None)
    _move_arm(robot, ArmAction.RIGHT_ARM, arm_pos_rel, arm_quat_rel, [0.0, -0.30, 0.30], [0, 0, 0, 1], 1)
    time.sleep(1.0)
    #arm_move_rec(robot, ArmAction.RIGHT_ARM, 0.2, 0, -0.1)
    ok_arm, arm_state = robot.getHandRelative(ArmAction.RIGHT_ARM)
    arm_pos_rel = getattr(arm_state, "position", None) 
    arm_quat_rel = getattr(arm_state, "rotation", None)
    _move_arm(robot, ArmAction.RIGHT_ARM, arm_pos_rel, arm_quat_rel, [0.17, -0.30, 0.30], [0, 0, 0, 1], 1)
    time.sleep(1.0)

    # 松爪前：只有 thread_0 下降腰部；thread_1 等待 thread_0 完成。
    wait_for_waist_before_release(robot, thread_index, sync_state)
    time.sleep(1.0)

    close_cmd.Position = 0.0
    robot.setGripper_high(EtherCAT_Motor_Index.MOTOR_RIGHT_ARM_8, close_cmd)
    time.sleep(GRIPPER_RELEASE_WAIT)

    # 必须先等双手都松开并恢复腰部，之后才能执行手臂撤回动作。
    restore_waist_before_arm_recovery(robot, thread_index, sync_state)

    ok_arm, arm_state = robot.getHandRelative(ArmAction.RIGHT_ARM)
    arm_pos_rel = getattr(arm_state, "position", None) 
    arm_quat_rel = getattr(arm_state, "rotation", None)
    _move_arm(robot, ArmAction.RIGHT_ARM, arm_pos_rel, arm_quat_rel, [0.0, -0.30, 0.30], [0, 0, 0, 1], 1)
    time.sleep(1.0)
    
    ok_arm, arm_state = robot.getHandRelative(ArmAction.RIGHT_ARM)
    arm_pos_rel = getattr(arm_state, "position", None) 
    arm_quat_rel = getattr(arm_state, "rotation", None)
    _move_arm(robot, ArmAction.RIGHT_ARM, arm_pos_rel, arm_quat_rel, [-0.1, 0.0, 0.30], [0, 0, 0, 1], 1)
    time.sleep(1.0)
    # prepare_robot_posture(robot, 0.75, 1.0, 0.67, 1.2)
    # time.sleep(1.0)

def grasp_by_right1(robot, target_pos, target_quat, cat):
    print(f" -> 🎯 解算目标相对平移: X={target_pos[0]:.4f}, Y={target_pos[1]:.4f}, Z={target_pos[2]:.4f}")

    #grasp
    print("\n[流程 E] 控制手臂从相对零点平滑逼近目标...")
    arm_move_right(robot, target_pos, target_quat)

    print("\n[流程 F] 正在闭合左手夹爪进行抓取...")
    close_cmd = Motor_Control()
    close_cmd.Position = 1.5
    robot.setGripper_high(EtherCAT_Motor_Index.MOTOR_RIGHT_ARM_8, close_cmd)

    time.sleep(2.0)

    arm_move_rec(robot, ArmAction.RIGHT_ARM, -0.2, 0, 0.05)
    ok_arm, arm_state = robot.getHandRelative(ArmAction.RIGHT_ARM)
    arm_pos_rel = getattr(arm_state, "position", None) 
    arm_quat_rel = getattr(arm_state, "rotation", None)
    _move_arm(robot, ArmAction.RIGHT_ARM, arm_pos_rel, arm_quat_rel, [0.0, -0.25, 0.30], [0, 0, 0, 1], 1)
    time.sleep(1.0)
    ok_arm, arm_state = robot.getHandRelative(ArmAction.RIGHT_ARM)
    arm_pos_rel = getattr(arm_state, "position", None) 
    arm_quat_rel = getattr(arm_state, "rotation", None)
    _move_arm(robot, ArmAction.RIGHT_ARM, arm_pos_rel, arm_quat_rel, [0.20, -0.25, 0.38], [0, 0, 0, 1], 1)
    time.sleep(1.0)

    close_cmd.Position = 0.0
    robot.setGripper_high(EtherCAT_Motor_Index.MOTOR_RIGHT_ARM_8, close_cmd)

    ok_arm, arm_state = robot.getHandRelative(ArmAction.RIGHT_ARM)
    arm_pos_rel = getattr(arm_state, "position", None) 
    arm_quat_rel = getattr(arm_state, "rotation", None)
    _move_arm(robot, ArmAction.RIGHT_ARM, arm_pos_rel, arm_quat_rel, [0.0, -0.25, 0.30], [0, 0, 0, 1], 1)
    time.sleep(1.0)
    
    ok_arm, arm_state = robot.getHandRelative(ArmAction.RIGHT_ARM)
    arm_pos_rel = getattr(arm_state, "position", None) 
    arm_quat_rel = getattr(arm_state, "rotation", None)
    _move_arm(robot, ArmAction.RIGHT_ARM, arm_pos_rel, arm_quat_rel, [0.0, -0.02, 0.30], [0, 0, 0, 1], 1)
    time.sleep(1.0)

def select_arm(robot, pose_path):
    print("\n[流程 D] 读取位姿状态并进行矩阵解算...")
    ok_cam, cam_state = robot.getHeadCameraRelative()
    cam_pos_rel = getattr(cam_state, "position", None) 
    cam_quat_rel = getattr(cam_state, "rotation", None) 

    ok_arm, arm_state = robot.getHandRelative(ArmAction.LEFT_ARM)
    arm_pos_rel = getattr(arm_state, "position", None) 
    arm_quat_rel = getattr(arm_state, "rotation", None) 

    if not (ok_cam and ok_arm):
        print("传感器位姿获取失败！")
        return

    T_obj_cam, angle = load_pose_matrix(pose_path)
    left_target_pos, left_target_quat = calculate_target_relative_pose(
        cam_pos_rel, cam_quat_rel, arm_pos_rel, arm_quat_rel, T_obj_cam, 0
    )

    ok_arm, arm_state = robot.getHandRelative(ArmAction.RIGHT_ARM)
    arm_pos_rel = getattr(arm_state, "position", None) 
    arm_quat_rel = getattr(arm_state, "rotation", None) 

    if not (ok_cam and ok_arm):
        print("传感器位姿获取失败！")
        return

    right_target_pos, right_target_quat = calculate_target_relative_pose(
        cam_pos_rel, cam_quat_rel, arm_pos_rel, arm_quat_rel, T_obj_cam, 1
    )
    if np.sum(np.array(left_target_pos)**2) < np.sum(np.array(right_target_pos)**2):
        return 0, left_target_pos, angle
    else:
        return 1, right_target_pos, angle


def split_tasks_by_arm(robot, result_queue):
    left_arm_queue = queue.Queue()
    right_arm_queue = queue.Queue()
    category_item_list = list()
    while not result_queue.empty():
        item = result_queue.get()
        if item is None:
            break
        category_id, _, pose_path = item
        if category_id == "cat4":
            category_item_list.append(item)
            continue
        flag, _, _ = select_arm(robot, pose_path)
        if flag == 0:
            left_arm_queue.put(item)
        else:
            right_arm_queue.put(item)
    return left_arm_queue, right_arm_queue, category_item_list


def compute_euclidean_distance(left_pose_path, right_pose_path):
    left_obj_matrix, _ = load_pose_matrix(left_pose_path)
    right_obj_matrix, _ = load_pose_matrix(right_pose_path)

    left_xyz = left_obj_matrix[:3, 3]
    right_xyz = right_obj_matrix[:3, 3]

    return np.linalg.norm(np.array(left_xyz) - np.array(right_xyz))


def max_weight_matching(weight_matrix):
    row_ind, col_ind = linear_sum_assignment(-weight_matrix)
    return list(zip(row_ind, col_ind))


def match_grasp_items_by_distance(left_arm_queue, right_arm_queue):
    left_items = []
    right_items = []
    while not left_arm_queue.empty():
        left_items.append(left_arm_queue.get())
    while not right_arm_queue.empty():
        right_items.append(right_arm_queue.get())

    m = len(left_items)
    n = len(right_items)
    result = []

    if m == 0 and n == 0:
        return result

    if m == 0:
        for r in right_items:
            result.append((None, r))
        return result

    if n == 0:
        for l in left_items:
            result.append((l, None))
        return result

    weight_matrix = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            weight_matrix[i, j] = compute_euclidean_distance(left_items[i][2], right_items[j][2])

    pairs = max_weight_matching(weight_matrix)

    matched_left = set()
    matched_right = set()
    for i, j in pairs:
        result.append((left_items[i], right_items[j]))
        matched_left.add(i)
        matched_right.add(j)

    for i in range(m):
        if i not in matched_left:
            result.append((left_items[i], None))

    for j in range(n):
        if j not in matched_right:
            result.append((None, right_items[j]))

    return result


def execute_arm_grasp(robot, item, target_flag, thread_index, sync_state):
    """执行一只机械臂的抓取；thread_0 负责腰部下降与恢复。"""
    thread_id = f"thread_{thread_index}"
    print(f"[{thread_id}] 开始执行，目标机械臂 flag={target_flag}")

    _, _, pose_path = item
    flag, target_pos, angle = select_arm(robot, pose_path)
    if flag != target_flag:
        print(f"[抓取失败] 期望机械臂 flag={target_flag}，实际 select_arm 返回 flag={flag}")
        return False

    target_quat = R.from_euler("xyz", [angle, 0, 0], degrees=True).as_quat()

    if flag == 0:
        grasp_by_left(robot, target_pos, target_quat, thread_index, sync_state)
    else:
        grasp_by_right(robot, target_pos, target_quat, thread_index, sync_state)
    time.sleep(1.0)
    return True


def execute_grasp_group(robot, left_item=None, right_item=None, parallel=True):
    """
    执行一组单臂或双臂任务。

    线程按任务加入顺序编号为 thread_0、thread_1：
    - thread_0 负责腰部下降和腰部恢复。
    - 所有手臂到达放料位后，thread_0 才下降腰部。
    - 所有夹爪都松开后，thread_0 立即恢复腰部。
    - 腰部恢复到 0.67m 后，各手臂才执行撤回动作。

    parallel=False 时，为避免近距离抓取冲突：
    thread_0 先抓取并到达放料位；随后才启动 thread_1。两只手都到达
    放料位以后，再统一下降、松爪、恢复腰部和撤回。
    """
    tasks = []
    if left_item is not None:
        tasks.append((left_item, 0))
    if right_item is not None:
        tasks.append((right_item, 1))

    if not tasks:
        return

    task_count = len(tasks)
    sync_state = {
        "ready_events": [threading.Event() for _ in range(task_count)],
        "released_events": [threading.Event() for _ in range(task_count)],
        "waist_down_event": threading.Event(),
        "waist_restored_event": threading.Event(),
    }

    threads = []
    for index, (item, target_flag) in enumerate(tasks):
        thread_id = f"thread_{index}"
        thread = threading.Thread(
            name=thread_id,
            target=execute_arm_grasp,
            args=(robot, item, target_flag, index, sync_state),
        )
        threads.append(thread)

    try:
        if parallel or len(threads) == 1:
            for thread in threads:
                thread.start()
        else:
            # 近距离双臂任务：先让 thread_0 抓取并停在放料位，
            # 再启动 thread_1，避免两只手在取料区域同时运动。
            threads[0].start()
            if not sync_state["ready_events"][0].wait(timeout=120.0):
                raise TimeoutError("thread_0 未能到达放料位，无法启动 thread_1")
            print("[主线程] thread_0 已到达放料位，开始启动 thread_1")
            for thread in threads[1:]:
                thread.start()

        for thread in threads:
            thread.join()
    finally:
        # 正常情况下，腰部已在线程内部、手臂撤回之前完成恢复。
        # 若某线程异常退出，则在本组退出前做一次安全兜底恢复。
        if (
            sync_state["waist_down_event"].is_set()
            and not sync_state["waist_restored_event"].is_set()
        ):
            print("[腰部][安全兜底] 检测到腰部尚未恢复，立即恢复到 0.67m")
            move_waist_z(robot, WAIST_RELEASE_Z, WAIST_NORMAL_Z)
            sync_state["waist_restored_event"].set()

def main():
    robot = H1Robot()
    try:
        print("[INIT] 实例化机器人并连接...")
        if not robot.robot_connect():
            print("连接机器人失败！")
            return

        robot.switchControlMode(MotorControlMode.HIGH_LEVEL)
        robot.robot_init()
        chassis_move(robot, 0.8)
        time.sleep(1.0)
        prepare_robot_posture(robot, 0, 0, 0.67, 1.2)
        arm_move_pre(robot, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [-0.1, 0.0, 0.30],
                [0.0, 0.0, 0.0, 1.0])
        chassis_move(robot, 0.3)
        time.sleep(2.0)

        #rgb_path, depth_path = capture_rgbd_data()

        for attempt in range(RETRY_COUNT):
            print(f"\n===== 抓取尝试 {attempt}/{RETRY_COUNT} =====")
            # ------------------------------------------------
            # 2. 拍摄 RGB-D 照片
            # ------------------------------------------------
            rgb_path, depth_path = capture_rgbd_data()
        
            # ------------------------------------------------
            # 3. 对当前画面中的所有检测物体执行注册并保存各自位姿
            # ------------------------------------------------
            result_queue = queue.Queue()

            perception_thread = threading.Thread(
                target=run_perception_client,
                args=(rgb_path, depth_path, CLIENT_DEBUG_DIR, result_queue),
                daemon=True,
            )
            perception_thread.start()
            time.sleep(0.1)

            while True:
                if not perception_thread.is_alive():
                    print("感知线程已结束，开始进入双臂操作模式...")
                    break
                item = result_queue.get()
                if item is None:
                    print("主线程收到 None准备退出...")
                    break
                category_id, instance_index, pose_path = item
                print(f"\n[主线程] 收到物体位姿: {category_id}_{instance_index}")
                print(pose_path)
                flag, _, _ = select_arm(robot, pose_path)
                if flag == 0:
                    execute_grasp_group(robot, left_item=item)
                else:
                    execute_grasp_group(robot, right_item=item)
                time.sleep(1.0)
            
            left_arm_queue, right_arm_queue, category_item_list = split_tasks_by_arm(robot, result_queue)
            pairs = match_grasp_items_by_distance(left_arm_queue, right_arm_queue)
            for pair in pairs:
                left_item, right_item = pair
                print(f"左臂: {left_item}, 右臂: {right_item}")

                both_present = left_item is not None and right_item is not None
                if both_present:
                    dist = compute_euclidean_distance(left_item[2], right_item[2])
                    if dist < 0.1:
                        print(f"物体距离 {dist:.4f}m < 0.1m，采用双线程串行执行")
                        execute_grasp_group(
                            robot,
                            left_item=left_item,
                            right_item=right_item,
                            parallel=False,
                        )
                        continue

                execute_grasp_group(
                    robot,
                    left_item=left_item,
                    right_item=right_item,
                    parallel=True,
                )

            print("开始单独处理第4类零件...")
            for category_item in  category_item_list:
                category_id, instance_index, pose_path = category_item
                print(f"\n[主线程] 收到物体位姿: {category_id}_{instance_index}")
                print(pose_path)
                flag, target_pos, angle = select_arm(robot, pose_path)
                target_quat = R.from_euler("xyz",[angle, 0, 0],degrees=True).as_quat()
                grasp_by_right1(robot, target_pos, target_quat, 'cat4')
                time.sleep(1.0)

            print("抓取流程在第 {attempt} 次尝试完成")
            # time.sleep(1.0)
            import pdb; pdb.set_trace()  

        while True:
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n[!] 捕获到 Ctrl+C 中断，准备执行安全下线流程...")
    except Exception as e:
        print(f"\n[!] 运行时发生异常: {e}")

    finally:
        print("[清理] 回收机器人控制权...")
        if 'robot' in locals() and hasattr(robot, "robot_deinit"):
            robot.robot_deinit()


if __name__ == "__main__":
    main()
