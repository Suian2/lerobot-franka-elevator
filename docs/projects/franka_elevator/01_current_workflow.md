# Franka ACT 单电梯按钮项目流程



当前有效基线：

* 只按一个固定电梯按钮
* 不使用楼层条件和 environment\_state
* L515 通过 RealSense SDK 直接读取
* 使用 unconditioned ACT
* HTTP负责状态、夹爪、回零和故障恢复
* ZMQ负责高频末端速度命令



## 阶段一：启动Franka控制服务

start\_franka\_control\_server.sh
（通过defaults.py获取IP，SSH启动远程VITA Docker；start-control检查HTTP服务，status检查HTTP和ZMQ状态。）

defaults.py
（统一提供机械臂控制主机IP，优先读取FRANKA\_CONTROL\_HOST，否则使用默认地址。）

## 阶段二：采集单按钮示教数据

1. run\_record.py 单按钮示教采集总入口。
（负责：解析录制参数；创建相机、机器人和SpaceMouse配置；连接全部硬件；创建LeRobotDataset；控制episode录制和保存；结束时释放硬件资源。）
2. configuration\_realsense.py 定义L515相机参数
（包括：相机设备名称或序列号；分辨率；FPS；是否使用RGB；是否使用深度；相机预热时间。）
3. franka\_robot.py 创建 FrankaRobotConfig 和 FrankaRobot，是LeRobot与Franka实机之间的核心适配器。
（主要负责：定义观测和动作格式；管理相机；启动状态缓存；读取机械臂状态；把末端动作转换成速度；通过ZMQ/HTTP下发控制命令；管理夹爪、Home、停止和断开。）
4. cameras/utils.py 根据 RealSenseCameraConfig 动态创建实际的 RealSenseCamera 对象。
(调用关系：RealSenseCameraConfig→ make\_cameras\_from\_configs()→ RealSenseCamera)
5. camera\_realsense.py L515相机的实际设备实现。
(负责：通过RealSense SDK连接L515；配置960×540、30 FPS RGB流；启动相机后台读取线程；保存最新RGB图像；向主循环提供最新图像。在 FrankaRobot.connect() 中，相机会首先连接。)
6. state\_cache.py 相机连接完成后，启动机械臂状态后台缓存线程。
(负责循环调用：HTTP GET /ctl/get\_curr
HTTP GET /ctl/gripper\_state
并缓存：七个关节角；夹爪状态；状态时间戳；HTTP状态错误。
主录制循环从缓存读取状态，避免每帧同步等待HTTP请求。)
7. franka\_spacemouse\_teleop.py SpaceMouse示教输入模块。
(负责：读取SpaceMouse六轴数据；进行死区处理；进行坐标轴映射和正负号转换；缩放成末端六维增量；生成夹爪开关命令；生成Home/reset命令。
输出形式：Δx、Δy、Δz、Δrx、Δry、Δrz、夹爪、reset)
8. record\_lerobot\_dataset.py 负责固定FPS的实际录制循环，把硬件数据组成同步训练帧。
(每帧顺序：FrankaRobot读取机械臂状态和图像→ SpaceMouse产生示教动作→ FrankaRobot发送机械臂动作→ 将观测和实际发送的动作组合成训练帧→ 添加到LeRobotDataset)
9. lerobot\_dataset.py LeRobotDataset的主要接口。
(负责：根据机器人特征创建数据集；管理当前episode缓冲区；添加每一帧数据；保存episode；管理数据集元数据；结束时完成视频和数据写入。)
10. dataset\_writer.py 数据集的底层写入实现。
(负责：将状态、动作和索引写入Parquet；将RGB图像编码成MP4；更新episode元数据；更新数据集统计信息；管理文件和chunk编号。)

## 阶段二输出：LeRobotDataset

dataset\_root/
├── data/*.parquet
├── videos/observation.images.l515/*.mp4
└── meta/
├── info.json
├── stats.json
├── tasks.parquet
└── episodes/*.parquet
各文件功能：
data/*.parquet：保存状态、动作、时间戳和帧索引；
videos/*.mp4：保存L515 RGB图像；
info.json：保存字段、shape、dtype和FPS；
stats.json：保存状态和动作的归一化统计量；
tasks.parquet：保存任务文本；
episodes/*.parquet：保存每个episode的范围和文件位置。

## 阶段三：ACT训练

1. lerobot\_train.py
（训练总入口；创建数据集、模型、优化器和DataLoader，并运行训练、验证和保存流程。）
2. datasets/factory.py
（根据训练配置加载LeRobotDataset，构造ACT动作块，可选划分训练集和验证集。）
3. lerobot\_dataset.py 训练期间作为PyTorch Dataset使用。
(负责根据样本索引返回：{
"observation.state": ...,
"observation.images.l515": ...,
"action": ...,
"action\_is\_pad": ...
}
其中：状态和动作来自Parquet；图像来自MP4；action包含未来一个动作块；action\_is\_pad标记episode末尾的填充动作。)
4. configuration\_act.py
（定义ACT的输入输出、动作块长度、ResNet主干、Transformer、VAE及归一化参数。）
5. modeling\_act.py：ACT模型本体
（定义并执行ACT模型网络，负责图像和状态编码、动作块预测及推理；模型权重由PreTrainedPolicy.from\_pretrained()加载。）
6. processor\_act.py：训练前后处理
（输入预处理和输出后处理；负责增加batch维、放入GPU、归一化和动作反归一化。）
7. optim/factory.py
（根据ACT训练配置创建AdamW优化器；当前ACT默认不配置学习率调度器。）
8. lerobot\_train.py中的训练循环
（每个训练step执行：DataLoader读取batch→图像转换为float32→processor归一化→ACTPolicy.forward()→计算L1和KL损失→backward()→梯度裁剪→optimizer.step()。仅在配置scheduler时更新学习率，当前ACT默认无scheduler。）
9. common/train\_utils.py
（`save_checkpoint()`保存policy权重、训练配置、processor和训练状态；`load_training_state()`恢复训练步数、随机数状态、优化器及可选scheduler。）

## 阶段三输出：ACT checkpoint
checkpoint/
├── pretrained\_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── train\_config.json
│   ├── policy\_preprocessor.json
│   └── policy\_postprocessor.json
└── training\_state/
├── optimizer\_state.safetensors
├── scheduler\_state.json（仅配置scheduler时生成；当前ACT默认不生成）
├── rng\_state.safetensors
└── training\_step.json
Rollout主要使用 pretrained\_model，不需要优化器状态。

## 阶段四：单按钮实机Rollout

1、当前单按钮无条件入口：
run\_act\_rollout\_realsense\_unconditioned.py
（当前单按钮无楼层条件的实机推理入口；加载checkpoint，构造模型输入，运行ACT并发送动作。）

2. configuration\_act.py
（从checkpoint的 config.json 恢复：ACT类型；输入输出特征；动作块长度；ResNet和Transformer结构；VAE和归一化配置。
无条件rollout还会检查：checkpoint必须是ACT；不能包含楼层条件；状态必须是8维；图像必须是3×540×960；动作必须是7维。）
3. modeling\_act.py
（定义ACTPolicy的网络及推理逻辑，根据图像和机器人状态预测动作块；模型权重由policies/pretrained.py中的from\_pretrained()加载。）
4. policies/pretrained.py
（从checkpoint的：model.safetensors 加载ACT模型权重，并恢复到 ACTPolicy。）
5. policies/factory.py
（根据：config.type = "act" 选择对应的 ACTPolicy，并调用统一接口加载ACT前后处理器。）
6. processor/pipeline.py
（从checkpoint加载policy\_preprocessor和policy\_postprocessor。
preprocessor负责观测归一化、增加batch维和移动到GPU；
postprocessor负责动作反归一化并移动到CPU。
ACT推理由modeling\_act.py中的ACTPolicy完成。）
7. configuration\_realsense.py
（创建rollout使用的L515配置：分辨率960×540、帧率30 FPS、启用RGB、不使用深度。）
8. franka\_robot.py
（获取实机观测，将ACT末端增量转换成受速度限制的笛卡尔速度；内部通过LatestZmqVelocitySender以ZMQ PUSH方式发送最新速度命令，地址为tcp://控制电脑:29010，并避免历史动作积压。）
9. cameras/utils.py
（根据RealSense配置创建实际的 RealSenseCamera 对象。）
10. camera\_realsense.py
（在 robot.connect() 时首先连接L515，并启动图像读取线程，为rollout持续提供最新RGB图像。）
11. state\_cache.py
（rollout期间持续更新关节和夹爪状态。）
12. 机械臂状态和L515图像→ preprocessor→ ACTPolicy.select\_action()→ postprocessor→ 取动作前6维并乘以action\_scale→ FrankaRobot.send\_action()→ ZMQ发送笛卡尔末端速度
当前unconditioned rollout虽然要求checkpoint输出7维动作，但只下发前6维末端动作，第7维夹爪动作被主动抑制。



## 运行结束、回零与故障恢复

* rollout正常结束或发生异常时，通过finally发送零笛卡尔速度。
* 断开相机、状态缓存和ZMQ连接。
* go\_home.py：机械臂回到预设关节Home位置。
* recover\_fault.py：清除机械臂故障并恢复控制状态。
* maintenance\_cli.py：状态检查、故障恢复和维护操作入口。



## 暂不进入当前单按钮主流程

* 多楼层条件编码
* 旧数据增加 target\_floor
* 多楼层数据合并与验证
* run\_act\_rollout\_realsense.py
* YOLO按钮识别
* 手眼标定
* UI及远程采集替代入口

