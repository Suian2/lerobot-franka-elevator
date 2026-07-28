from __future__ import annotations

import argparse
import queue
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware_test.franka.franka_recording_controller import (  # noqa: E402
    DatasetSpec,
    FrankaRecorderSession,
    RecorderCommand,
    RecorderEvent,
    RecorderOptions,
    RecorderState,
    RecorderWorker,
)
from hardware_test.franka.franka_robot import FrankaRobot  # noqa: E402
from hardware_test.franka.franka_spacemouse_teleop import FrankaSpaceMouseTeleop  # noqa: E402
from hardware_test.franka.record_lerobot_dataset import (  # noqa: E402
    create_lerobot_dataset,
    make_lerobot_frame,
)
from hardware_test.franka.run_record import (  # noqa: E402
    build_arg_parser,
    build_cameras,
    build_robot_config,
    build_teleop_config,
    describe_config,
)


@dataclass(frozen=True)
class ButtonPolicy:
    start: bool
    end: bool
    home: bool
    clear_fault: bool


@dataclass(frozen=True)
class UiRuntime:
    worker: RecorderWorker


def button_policy(state: RecorderState) -> ButtonPolicy:
    if state is RecorderState.IDLE:
        return ButtonPolicy(True, False, True, True)
    if state is RecorderState.RECORDING:
        return ButtonPolicy(False, True, True, True)
    if state in {RecorderState.HOMING_IDLE, RecorderState.HOMING_RECORDING, RecorderState.FAULTED}:
        return ButtonPolicy(False, False, False, True)
    return ButtonPolicy(False, False, False, False)


def build_ui_arg_parser() -> argparse.ArgumentParser:
    parser = build_arg_parser()
    parser.description = "Franka + L515 SpaceMouse LeRobot recorder UI"
    parser.set_defaults(duration_s=0.0, num_episodes=0, reset_time_s=0.0)
    return parser


def _validate_ui_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.action_mode != "delta_ee_pose":
        parser.error("the recorder UI supports only --action-mode delta_ee_pose")
    if args.push_to_hub:
        parser.error("the recorder UI writes locally; upload the finalized dataset separately")


def _build_runtime(args: argparse.Namespace) -> UiRuntime:
    robot = FrankaRobot(build_robot_config(args), cameras=build_cameras(args))
    teleop = FrankaSpaceMouseTeleop(build_teleop_config(args))

    def dataset_factory(spec: DatasetSpec):
        return create_lerobot_dataset(
            repo_id=spec.repo_id,
            root=spec.root,
            fps=args.fps,
            robot=robot,
            teleop=teleop,
            task=spec.task,
            use_videos=args.use_videos,
            image_writer_processes=args.image_writer_processes,
            image_writer_threads_per_camera=args.image_writer_threads_per_camera,
            streaming_encoding=args.streaming_encoding,
            encoder_queue_maxsize=args.encoder_queue_maxsize,
            encoder_threads=args.encoder_threads,
        )

    session = FrankaRecorderSession(
        robot=robot,
        teleop=teleop,
        dataset_factory=dataset_factory,
        options=RecorderOptions(
            fps=args.fps,
            duration_s=args.duration_s,
            num_episodes=args.num_episodes,
            cartesian_action_units=args.cartesian_action_units,
            max_consecutive_state_misses=args.state_max_consecutive_misses,
            max_state_wait_s=args.max_state_wait_s,
            state_retry_sleep_s=args.state_retry_sleep_s,
        ),
        frame_builder=make_lerobot_frame,
    )
    return UiRuntime(worker=RecorderWorker(session, fps=args.fps))


class FrankaRecorderApp:
    def __init__(self, root: Any, runtime: UiRuntime, args: argparse.Namespace) -> None:
        import tkinter as tk
        from tkinter import scrolledtext, ttk

        self.root = root
        self.runtime = runtime
        self.worker = runtime.worker
        self.args = args
        self._tk = tk
        self._ttk = ttk
        self._closing = False
        self._close_dialog_open = False
        self._last_log_line = ""
        self._after_id: str | None = None

        root.title("Franka LeRobot 数据录制")
        root.minsize(760, 600)
        root.protocol("WM_DELETE_WINDOW", self.request_close)

        self.repo_var = tk.StringVar(value=args.repo_id)
        self.root_var = tk.StringVar(value=args.root)
        self.task_var = tk.StringVar(value=args.task)
        self.connection_var = tk.StringVar(value="● 正在连接")
        self.state_var = tk.StringVar(value="启动录制后端…")
        self.episode_var = tk.StringVar(value="已保存 Episode: 0")
        self.frame_var = tk.StringVar(value="当前帧: 0")
        self.elapsed_var = tk.StringVar(value="录制: 0.0s  墙钟: 0.0s")

        style = ttk.Style(root)
        style.configure("Primary.TButton", font=("Sans", 14, "bold"), padding=(16, 14))
        style.configure("Secondary.TButton", font=("Sans", 11), padding=(12, 10))
        style.configure("Status.TLabel", font=("Sans", 12, "bold"))

        outer = ttk.Frame(root, padding=18)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 14))
        ttk.Label(header, textvariable=self.connection_var, style="Status.TLabel").pack(side="left")
        ttk.Label(header, textvariable=self.state_var).pack(side="right")

        config = ttk.LabelFrame(outer, text="数据集", padding=12)
        config.pack(fill="x", pady=(0, 14))
        self.repo_entry = self._entry_row(config, 0, "Repo ID", self.repo_var)
        self.root_entry = self._entry_row(config, 1, "保存目录", self.root_var)
        self.task_entry = self._entry_row(config, 2, "任务描述", self.task_var)
        config.columnconfigure(1, weight=1)

        primary = ttk.Frame(outer)
        primary.pack(fill="x", pady=(0, 12))
        primary.columnconfigure((0, 1), weight=1)
        self.start_button = ttk.Button(
            primary,
            text="开始录制",
            style="Primary.TButton",
            command=self._start_recording,
        )
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.end_button = ttk.Button(
            primary,
            text="结束录制",
            style="Primary.TButton",
            command=lambda: self.worker.submit(RecorderCommand.end()),
        )
        self.end_button.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        secondary = ttk.Frame(outer)
        secondary.pack(fill="x", pady=(0, 14))
        secondary.columnconfigure((0, 1), weight=1)
        self.home_button = ttk.Button(
            secondary,
            text="回到原位",
            style="Secondary.TButton",
            command=lambda: self.worker.submit(RecorderCommand.home()),
        )
        self.home_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.fault_button = ttk.Button(
            secondary,
            text="清除 Fault",
            style="Secondary.TButton",
            command=lambda: self.worker.submit(RecorderCommand.clear_fault()),
        )
        self.fault_button.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        stats = ttk.Frame(outer)
        stats.pack(fill="x", pady=(0, 12))
        ttk.Label(stats, textvariable=self.episode_var).pack(side="left", padx=(0, 24))
        ttk.Label(stats, textvariable=self.frame_var).pack(side="left", padx=(0, 24))
        ttk.Label(stats, textvariable=self.elapsed_var).pack(side="left")

        ttk.Label(outer, text="运行日志").pack(anchor="w")
        self.log = scrolledtext.ScrolledText(outer, height=14, state="disabled", wrap="word")
        self.log.pack(fill="both", expand=True, pady=(4, 0))

        self._apply_policy(ButtonPolicy(False, False, False, False))
        self._append_log("UI 已启动；硬件连接由 RecorderWorker 建立")
        self.worker.start()
        self._schedule_poll()

    def _entry_row(self, parent: Any, row: int, label: str, variable: Any):
        self._ttk.Label(parent, text=label, width=12).grid(row=row, column=0, sticky="w", pady=4)
        entry = self._ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        return entry

    def _start_recording(self) -> None:
        repo_id = self.repo_var.get().strip()
        root = self.root_var.get().strip()
        task = self.task_var.get().strip()
        if not repo_id or not root or not task:
            from tkinter import messagebox

            messagebox.showerror("参数缺失", "Repo ID、保存目录和任务描述不能为空。", parent=self.root)
            return
        self.worker.submit(RecorderCommand.start(DatasetSpec(repo_id, root, task)))

    def request_close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._apply_policy(ButtonPolicy(False, False, False, False))
        if not self.worker.is_alive():
            self.root.destroy()
            return
        self.worker.submit(RecorderCommand.prepare_close())

    def _schedule_poll(self) -> None:
        self._after_id = self.root.after(50, self._poll_worker)

    def _poll_worker(self) -> None:
        try:
            while True:
                self._apply_event(self.worker.get_event_nowait())
        except queue.Empty:
            pass
        worker_alive = self.worker.is_alive()
        if worker_alive or not self._closing:
            self._schedule_poll()
        else:
            self.root.after_idle(self.root.destroy)

    def _apply_event(self, event: RecorderEvent) -> None:
        snapshot = event.snapshot
        connected = snapshot.state not in {RecorderState.CONNECTING, RecorderState.FATAL_ERROR}
        self.connection_var.set("● 已连接" if connected else "● 未连接")
        self.state_var.set(snapshot.message)
        self.episode_var.set(f"已保存 Episode: {snapshot.saved_episodes}")
        self.frame_var.set(f"当前帧: {snapshot.frame_count}")
        self.elapsed_var.set(
            f"录制: {snapshot.frame_count / max(1, self.args.fps):.1f}s  墙钟: {snapshot.wall_elapsed_s:.1f}s"
        )
        policy = button_policy(snapshot.state)
        if self.args.num_episodes and snapshot.saved_episodes >= self.args.num_episodes:
            policy = ButtonPolicy(False, policy.end, policy.home, policy.clear_fault)
        self._apply_policy(policy)
        self._set_dataset_locked(snapshot.dataset_locked)
        if event.message and event.message != self._last_log_line:
            self._append_log(event.message, level=event.level)
            self._last_log_line = event.message
        if self._closing and event.level == "error" and snapshot.state is not RecorderState.CLOSED:
            self._closing = False
        if self._closing and snapshot.state is RecorderState.PAUSED_CLOSE:
            self._show_close_decision(snapshot)
        elif snapshot.state is RecorderState.CLOSED:
            self.root.after_idle(self.root.destroy)

    def _show_close_decision(self, snapshot: Any) -> None:
        if self._close_dialog_open:
            return
        self._close_dialog_open = True
        from tkinter import messagebox

        try:
            if not snapshot.pending_valid and snapshot.frame_count == 0:
                self.worker.submit(RecorderCommand.close(save_pending=False))
                return
            if snapshot.can_save_on_close:
                decision = messagebox.askyesnocancel(
                    "关闭录制器",
                    "当前 episode 尚未结束。\n是：保存并关闭\n否：丢弃并关闭\n取消：继续录制",
                    parent=self.root,
                )
                if decision is None:
                    self._closing = False
                    self.worker.submit(RecorderCommand.cancel_close())
                else:
                    self.worker.submit(RecorderCommand.close(save_pending=bool(decision)))
                return
            discard = messagebox.askokcancel(
                "归位尚未完成",
                "当前 episode 不能安全保存。确定丢弃并关闭吗？",
                parent=self.root,
            )
            if discard:
                self.worker.submit(RecorderCommand.close(save_pending=False))
            else:
                self._closing = False
                self.worker.submit(RecorderCommand.cancel_close())
        finally:
            self._close_dialog_open = False

    def _apply_policy(self, policy: ButtonPolicy) -> None:
        self.start_button.configure(state="normal" if policy.start else "disabled")
        self.end_button.configure(state="normal" if policy.end else "disabled")
        self.home_button.configure(state="normal" if policy.home else "disabled")
        self.fault_button.configure(state="normal" if policy.clear_fault else "disabled")

    def _set_dataset_locked(self, locked: bool) -> None:
        state = "disabled" if locked else "normal"
        self.repo_entry.configure(state=state)
        self.root_entry.configure(state=state)
        self.task_entry.configure(state=state)

    def _append_log(self, message: str, *, level: str = "info") -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log.configure(state="normal")
        self.log.insert("end", f"[{timestamp}] {level.upper():5s} {message}\n")
        self.log.see("end")
        self.log.configure(state="disabled")


def _install_signal_handlers(root: Any, app: FrankaRecorderApp) -> None:
    def request_close(signum: int, frame: Any) -> None:  # noqa: ARG001
        root.after(0, app.request_close)

    signal.signal(signal.SIGINT, request_close)
    signal.signal(signal.SIGTERM, request_close)


def main(argv: list[str] | None = None) -> int:
    parser = build_ui_arg_parser()
    args = parser.parse_args(argv)
    _validate_ui_args(parser, args)
    robot_config = build_robot_config(args)
    print(describe_config(args, robot_config), flush=True)
    if args.dry_run_config:
        print("dry-run-config: no hardware connection or Tk window attempted", flush=True)
        return 0

    import tkinter as tk

    root = tk.Tk()
    app = FrankaRecorderApp(root, _build_runtime(args), args)
    _install_signal_handlers(root, app)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
