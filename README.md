# G1 Control

Control and AI for the Unitree G1 29-DoF (dummy hands) from this server.

```
browser ──► server (UI + relay, :8000) ──HTTP──► Agent on robot (:8100) ──DDS──► G1
```

All hosts, ports, paths, limits and models are in `config/config.yaml`. Secrets go in `.env`.

## Setup (once)

```bash
scripts/setup_server.sh          # conda env g1ui_server
scripts/setup_robot.sh           # Miniforge + g1ui_robot env on the robot (over SSH)
scripts/robot.sh install         # deploy the Agent and install it as a systemd service
```

## Run

```bash
scripts/run_server.sh            # then open http://<server-ip>:8000
scripts/robot.sh deploy && scripts/robot.sh restart   # after changing robot code
scripts/robot.sh logs            # Agent logs
```

## Agent API

`POST /cmd` with `{"id", "action", "params"}`. The response is `{"id", "ok", "status", "result", "error", "ms"}`.
The status is `done`, `running`, `failed` or `rejected`. A command id never runs twice. Poll `GET /cmd/{id}` while it is `running`.

| action | params |
|---|---|
| forward, backward, left_aside, right_aside | velocity (m/s, ≤ 0.6), duration (s, ≤ 30) |
| turn_left, turn_right | velocity (rad/s, ≤ 0.6), duration (s, ≤ 30) |
| set_mode | mode: zero_torque, damping, stand_up, locomotion, manipulation |
| set_upperbody | q: 17 angles in degrees (motor index 12..28), duration (optional) |
| play_upperbody | keyframes: `[{"t": 1.2, "q": {"18": 70}}]`, interp: smooth or linear |
| release_upperbody | — |
| speak | text (≤ 200 chars) |
| play_audio / stop_audio | pcm_b64 (16 kHz mono int16) |
| headlight | hex |
| processing | source, on (blinks the head light red while any source is on) |
| set_volume / get_volume | volume 0..100 |
| get_mode, get_29_states, get_battery_status, get_imu, get_end_effector_state | — |
