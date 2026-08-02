# Franka ACT 单按钮项目代码模块说明



## 1. 文档范围



本文档说明当前单电梯按钮项目实际使用的代码模块、调用关系和输入输出。



当前主线：



- 直接使用RealSense SDK读取L515

- 使用SpaceMouse采集示教动作

- 使用LeRobotDataset保存数据

- 使用无楼层条件的ACT训练

- 使用unconditioned rollout

- HTTP负责状态和低频操作

- ZMQ负责高频笛卡尔速度命令



多楼层、YOLO和手眼标定不属于当前主线。ROS2图像桥接代码已从当前开发分支删除。



## 2. 当前主调用链



```text

start_franka_control_server.sh

└── defaults.py

      └── 远程VITA Franka控制服务



run_record.py

├── RealSenseCameraConfig

├── FrankaRobot

│   ├── RealSenseCamera

│   ├── FrankaStateCache

│   └── LatestZmqVelocitySender

├── FrankaSpaceMouseTeleop

└── record_lerobot_dataset.py

      └── LeRobotDataset



lerobot_train.py

├── datasets/factory.py

├── lerobot_dataset.py

├── configuration_act.py

├── modeling_act.py

├── processor_act.py

├── optim/factory.py

└── common/train_utils.py



run_act_rollout_realsense_unconditioned.py

├── ACT checkpoint

├── policy preprocessor/postprocessor

├── RealSenseCamera

├── FrankaStateCache

└── FrankaRobot

      ├── ZMQ笛卡尔速度

      └── HTTP状态与维护操作

```



## 3. 控制服务模块



### `hardware_test/franka/defaults.py`



作用：



- 统一提供Franka控制电脑地址

- 优先读取环境变量`FRANKA_CONTROL_HOST`

- 未设置环境变量时使用项目默认地址



主要调用者：



- `start_franka_control_server.sh`

- `run_record.py`

- rollout和维护脚本



### `hardware_test/franka/scripts/start_franka_control_server.sh`



作用：



- 通过SSH连接Franka控制电脑

- 启动远程VITA Docker控制服务

- 检查HTTP控制服务状态

- 提供启动、状态查询和停止命令



输出：



- HTTP控制端点：`http://<control-host>:29000/ctl`

- ZMQ速度端点：`tcp://<control-host>:29010`



## 4. 数据采集模块



### `hardware_test/franka/run_record.py`



作用：



- 单按钮数据采集总入口

- 解析录制、相机、机械臂和SpaceMouse参数

- 创建机器人、遥操作器和LeRobotDataset

- 管理episode录制、保存、重置和资源释放



当前默认：



- `camera_backend=realsense`

- 不要求`target_floor`

- 不写入楼层条件`environment_state`



### `hardware_test/franka/franka_robot.py`



作用：



- 实现LeRobot的Franka机器人适配器

- 定义观测和动作格式

- 管理相机、状态缓存、HTTP客户端和ZMQ发送器

- 将末端增量动作转换为受限笛卡尔速度

- 管理夹爪、Home、停止及断开



连接顺序：



```text

连接相机

→ 启动FrankaStateCache

→ 初始化ZMQ速度发送器

→ 标记机器人已连接

```



观测输出：



- 7维关节角

- 1维归一化夹爪宽度

- L515 RGB图像



### `hardware_test/franka/state_cache.py`



作用：



- 后台轮询机械臂状态

- 顺序调用`get_curr()`和`gripper_get_state()`

- 缓存关节角、夹爪状态、时间戳和错误

- 为录制和rollout提供非阻塞状态读取



### `hardware_test/franka/franka_spacemouse_teleop.py`



作用：



- 读取SpaceMouse六轴输入

- 完成死区、坐标映射、符号转换和缩放

- 生成六维末端增量动作

- 管理夹爪开关和Home/reset输入



### `hardware_test/franka/record_lerobot_dataset.py`

作用：

- 定义数据集features和单帧数据格式
- 创建LeRobotDataset
- 执行固定FPS的`record_lerobot_episode()`录制循环
- 同步状态、图像和实际发送动作
- 每帧调用`dataset.add_frame()`

episode录制完成后，由`run_record.py`调用`dataset.save_episode()`；
整个采集程序退出时，由`run_record.py`调用`dataset.finalize()`。



## 5. 相机模块



### `src/lerobot/cameras/realsense/configuration_realsense.py`



作用：



- 定义RealSense设备、序列号、分辨率、FPS和数据流配置



### `src/lerobot/cameras/utils.py`



作用：



- 根据`RealSenseCameraConfig`创建实际相机对象



### `src/lerobot/cameras/realsense/camera_realsense.py`



作用：



- 通过RealSense SDK连接L515

- 启动后台图像读取线程

- 保存最新RGB帧

- 为采集和rollout提供最新图像



## 6. 数据集模块



### `src/lerobot/datasets/lerobot_dataset.py`



作用：



- 创建和加载LeRobotDataset

- 管理episode缓冲区

- 添加训练帧

- 保存episode

- 训练时作为PyTorch Dataset返回样本



### `src/lerobot/datasets/dataset_writer.py`



作用：



- 将状态、动作和索引写入Parquet

- 将RGB图像编码或写入视频

- 更新episode元数据和统计信息



### `src/lerobot/datasets/factory.py`

作用：

- 根据训练配置加载LeRobotDataset
- 从ACT配置的`action_delta_indices`计算`delta_timestamps`
- 创建训练集和可选验证集
- LeRobotDataset根据这些时间偏移返回未来动作序列



## 7. ACT训练模块



### `src/lerobot/scripts/lerobot_train.py`



作用：



- LeRobot通用策略训练入口
- 当前项目通过`policy.type=act`选择ACTPolicy
- 创建数据集、策略、处理器、优化器和DataLoader
- 执行训练、可选验证和checkpoint保存



### `src/lerobot/policies/act/configuration_act.py`



作用：



- 定义ACT输入输出特征

- 配置动作块长度、ResNet、Transformer和VAE

- 配置归一化方式



### `src/lerobot/policies/act/modeling_act.py`



作用：



- 定义ACTPolicy和ACT模型网络

- 编码图像与机器人状态

- 预测动作块

- 计算L1和KL损失

- 执行策略推理



不负责独立读取`model.safetensors`。



### `src/lerobot/policies/act/processor_act.py`



作用：
- 定义ACT默认preprocessor和postprocessor
- preprocessor执行重命名、增加batch维、设备迁移和归一化
- postprocessor执行动作反归一化并迁移到CPU
- 训练时由`policies/factory.py`调用本模块创建pipeline
- pipeline的checkpoint保存由`common/train_utils.py`负责
- rollout由`policies/factory.py`加载checkpoint中保存的pipeline



### `src/lerobot/optim/factory.py`



作用：


- 根据`TrainPipelineConfig`构建优化器和可选学习率调度器
- 当前ACT预设使用AdamW
- 当前ACT默认`get_scheduler_preset()`返回None，因此默认没有学习率调度器



### `src/lerobot/common/train_utils.py`



作用：

- `save_checkpoint()`保存policy权重、训练配置、前后处理器和训练状态
- `load_training_state()`恢复训练步数、随机数状态、优化器和可选scheduler
- policy权重由`PreTrainedPolicy.from_pretrained()`恢复，不由`load_training_state()`加载



## 8. Rollout模块



### `hardware_test/franka/run_act_rollout_realsense_unconditioned.py`



作用：



- 当前单按钮实机rollout唯一主入口

- 验证checkpoint不包含楼层条件

- 加载ACT模型及前后处理器

- 读取L515图像与Franka状态

- 执行ACT推理

- 将动作发送给FrankaRobot

- 正常结束或异常退出时发送零速度



当前动作处理：



```text

7维ACT输出

→ 只取前6维末端动作

→ 乘以action_scale

→ FrankaRobot.send_action()

→ ZMQ发送笛卡尔速度

```



第7维夹爪动作在当前rollout中被主动抑制。



### `src/lerobot/policies/pretrained.py`



作用：



- 从`model.safetensors`加载ACT模型权重

- 将权重恢复到ACTPolicy



### `src/lerobot/policies/factory.py`



作用：



- 根据`config.type`选择ACTPolicy

- 创建或加载策略前后处理器



### `src/lerobot/processor/pipeline.py`

作用：

- 定义通用的ProcessorStep和DataProcessorPipeline框架
- 串联多个数据转换步骤
- 支持pipeline的执行、序列化、保存和恢复
- rollout中由`policies/factory.py`调用`from_pretrained()`恢复前后处理器
- 不执行ACT模型推理


## 9. 维护模块



### `hardware_test/franka/go_home.py`



作用：



- 请求机械臂移动到预设七关节Home位置



### `hardware_test/franka/recover_fault.py`



作用：


- 调用`client.recover()`清除Franka和速度循环故障
- 随后调用`velocity_loop_status()`查询恢复后的速度循环状态



### `hardware_test/franka/maintenance_cli.py`

作用：

- 为维护脚本提供公共命令行参数
- 创建HTTP模式的FrankaControlClient
- 统一执行维护请求、输出JSON结果、处理异常并关闭客户端
- 被`go_home.py`和`recover_fault.py`共同调用



## 10. 兼容模块



### `hardware_test/franka/franka_zmq_http_robot.py`



作用：



- 兼容旧导入路径

- 重新导出`franka_robot.py`中的配置和类

- 不包含第二套独立的Franka控制逻辑



## 11. 当前非主线模块



以下模块保留在仓库，但不属于当前单按钮运行链：

- `run_record_ui.py`

- `handeye/`

- `run_teleop.py`：不保存数据的手动遥操作入口
- `franka_recording_controller.py`：录制UI使用的后台控制器
- `scripts/start_franka_record_ui.sh`：录制UI启动脚本
