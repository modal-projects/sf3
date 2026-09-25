import asyncio
import mimetypes
import os
import queue
import threading
import time
from collections import deque
from fractions import Fraction
from pathlib import Path

import modal

from src.env import EnvironmentConfig, create_environment
from src.serve import MODELS
from src.utils import (
    COMBOS,
    CONTAINER_REGION,
    MAX_CPU_DIFFICULTY,
    DEFAULT_PLAYER1_PARTICIPANT,
    DEFAULT_PLAYER2_PARTICIPANT,
    MINUTES,
    PARTICIPANT_LABELS,
    RECENT_MOVE_LIMIT,
    ROUTING_REGION,
    SPECIAL_MOVES,
    FrameEncoder,
    create_gameplay_image,
    generate_move,
    is_model_participant,
    normalize_game_participants,
    player_state,
)

mimetypes.add_type("image/webp", ".webp")

# Modal setup

app = modal.App(name="sf3")
for model in MODELS.values():
    app.include(model.app)

CONTROL_MESSAGE_QUEUE_LIMIT = 128
SESSION_TASK_SHUTDOWN_TIMEOUT_SECONDS = 1.0
ENV_INIT_TIMEOUT_SECONDS = 1 * MINUTES
GAMEPLAY_PORT = 8000
GAMEPLAY_SESSION_IDLE_TIMEOUT_SECONDS = 10 * MINUTES
MAX_GAME_SESSIONS = 6


local_assets_dir = Path(__file__).parent.parent / "assets"

remote_frontend_dir = "/root/frontend"
remote_icons_dir = "/root/icons"
remote_logos_dir = "/root/logos"
remote_sounds_dir = "/root/sounds"

launcher_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "fastapi[standard]==0.116.1",
)

gameplay_image = (
    create_gameplay_image(
        extra_python_packages=(
            "aiortc",
            "av",
            "fastapi[standard]==0.116.1",
            "websockets==15.0.1",
        )
    )
    .add_local_dir(Path(__file__).parent / "frontend", remote_frontend_dir)
    .add_local_dir(local_assets_dir / "icons", remote_icons_dir)
    .add_local_dir(local_assets_dir / "logos", remote_logos_dir)
    .add_local_dir(local_assets_dir / "sounds", remote_sounds_dir)
)


@app.server(
    image=gameplay_image,
    compute_region=CONTAINER_REGION,
    routing_region=ROUTING_REGION,
    min_containers=1,
    max_concurrency=3,
    target_concurrency=2,
    secrets=[modal.Secret.from_name("turn-credentials")],
    port=GAMEPLAY_PORT,
    startup_timeout=5 * MINUTES,
)
@modal.sessioned()
class Web:
    @modal.enter()
    def enter(
        self,
    ):
        import uvicorn

        self.participant_servers = {}
        self.participant_boot_tasks = {}
        self.game_sessions = {}
        self.http_server = uvicorn.Server(
            uvicorn.Config(
                self.build_app(),
                host="0.0.0.0",
                port=GAMEPLAY_PORT,
                log_level="warning",
            )
        )
        threading.Thread(target=self.http_server.run, daemon=True).start()
        if os.environ.get("SF3_WARM_MODELS") != "1":
            return
        for participant, model in MODELS.items():
            try:
                model.server().update_autoscaler(min_containers=1)
            except Exception as exc:
                label = PARTICIPANT_LABELS.get(participant, participant)
                print(f"Could not keep {label} warm: {exc!r}")

    @modal.exit()
    def exit(self):
        self.http_server.should_exit = True

    async def create_participant_server(self, participant: str):
        model = MODELS.get(participant)
        if model is None:
            raise ValueError(f"Unsupported participant: {participant}")

        server = self.participant_servers.get(participant)
        if server is not None:
            return server

        boot_task = self.participant_boot_tasks.get(participant)
        if boot_task is None:

            async def boot():
                label = PARTICIPANT_LABELS.get(participant, participant)
                print(f"Creating {label}...")
                server = model.server()
                await asyncio.wait_for(
                    server.boot.remote.aio(),
                    timeout=60 * MINUTES,
                )
                self.participant_servers[participant] = server
                print(f"{label} created")
                return server

            boot_task = asyncio.create_task(boot())
            self.participant_boot_tasks[participant] = boot_task
            boot_task.add_done_callback(
                lambda _: self.participant_boot_tasks.pop(participant, None)
            )

        return await asyncio.shield(boot_task)

    def build_app(self):
        import json
        import traceback

        import numpy as np
        from aiortc import (
            RTCConfiguration,
            RTCIceServer,
            RTCPeerConnection,
            RTCSessionDescription,
            VideoStreamTrack,
        )
        from aiortc.sdp import candidate_from_sdp
        from av import VideoFrame
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.middleware.gzip import GZipMiddleware
        from fastapi.responses import FileResponse, JSONResponse, Response
        from fastapi.staticfiles import StaticFiles
        from starlette.websockets import WebSocketState

        web_app = FastAPI()
        web_app.add_middleware(GZipMiddleware, minimum_size=1024)

        # helper fns

        stun_ice_servers = [{"urls": "stun:stun.l.google.com:19302"}]

        def build_ice_servers() -> list[dict]:
            username = os.environ.get("TURN_USERNAME")
            credential = os.environ.get("TURN_CREDENTIAL")
            if not username or not credential:
                return stun_ice_servers

            creds = {"username": username, "credential": credential}
            return [
                {"urls": "stun:stun.relay.metered.ca:80"},
                {"urls": "turn:standard.relay.metered.ca:80"} | creds,
                {"urls": "turn:standard.relay.metered.ca:80?transport=tcp"} | creds,
                {"urls": "turn:standard.relay.metered.ca:443"} | creds,
                {"urls": "turns:standard.relay.metered.ca:443?transport=tcp"} | creds,
            ]

        def build_turn_servers() -> dict:
            return {
                "type": "turn_servers",
                "ice_servers": build_ice_servers(),
            }

        def build_rtc_configuration() -> RTCConfiguration:
            ice_servers = []
            for server in build_ice_servers():
                ice_servers.append(
                    RTCIceServer(
                        urls=server["urls"],
                        username=server.get("username"),
                        credential=server.get("credential"),
                    )
                )
            return RTCConfiguration(iceServers=ice_servers)

        class GameFrameBuffer:
            def __init__(self, should_stop, target_fps: float = 60.0):
                self.should_stop = should_stop
                self.target_fps = target_fps
                self.latest_frame = None
                self.phase_frames = queue.Queue(maxsize=2)

            def set_frame(self, frame):
                self.latest_frame = frame

            def queue_frame(self, frame):
                while True:
                    try:
                        self.phase_frames.put_nowait(frame)
                        self.latest_frame = frame
                        return
                    except queue.Full:
                        try:
                            self.phase_frames.get_nowait()
                        except queue.Empty:
                            pass

            async def wait_for_phase_frames(self):
                while not self.phase_frames.empty() and not self.should_stop():
                    await asyncio.sleep(1 / (self.target_fps * 2))

        class GameVideoTrack(VideoStreamTrack):
            def __init__(self, frame_buffer):
                super().__init__()
                self.frame_buffer = frame_buffer
                self._timestamp = 0
                self._has_sent_frame = False
                self._last_frame_at = None

            async def recv(self) -> VideoFrame:
                while self.frame_buffer.latest_frame is None:
                    await asyncio.sleep(1 / self.frame_buffer.target_fps)
                frame = self.frame_buffer.latest_frame

                loop = asyncio.get_running_loop()
                timestamp_step = int((1 / self.frame_buffer.target_fps) * 90000)
                if self._last_frame_at is not None:
                    deadline = self._last_frame_at + (1 / self.frame_buffer.target_fps)
                    while (delay := deadline - loop.time()) > 0:
                        await asyncio.sleep(delay)
                self._last_frame_at = loop.time()
                if self._has_sent_frame:
                    self._timestamp += timestamp_step
                else:
                    self._has_sent_frame = True

                try:
                    frame = self.frame_buffer.phase_frames.get_nowait()
                except queue.Empty:
                    frame = self.frame_buffer.latest_frame
                output = VideoFrame.from_ndarray(frame, format="rgb24")
                output.pts = self._timestamp
                output.time_base = Fraction(1, 90000)
                return output

        class NumpyJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if hasattr(obj, "__dict__"):
                    return vars(obj)
                return super().default(obj)

        def make_json_safe(obj):
            return json.loads(json.dumps(obj, cls=NumpyJSONEncoder))

        def create_initial_game_state():
            return {
                "status": "initializing",
                "scores": [0, 0],
                "round_number": 1,
                "winner": "",
                "winner_side": "",
                "error": "",
                "accepts_input": False,
                "match_identity": None,
                "player1_selection": None,
                "player2_selection": None,
            }

        # manages game state and communication

        class GameSession:
            def __init__(self):
                # game state

                self.env = None
                self.game_running = False
                self.accepts_input = False
                self.start_requested = False
                self.match_identity = None
                self.game_settings = {
                    "player1": {
                        "character": "Ken",
                        "outfit": 1,
                        "superArt": 1,
                    },
                    "player2": {
                        "character": "Ryu",
                        "outfit": 1,
                        "superArt": 1,
                    },
                    "player1Participant": DEFAULT_PLAYER1_PARTICIPANT,
                    "player2Participant": DEFAULT_PLAYER2_PARTICIPANT,
                }
                self.game_state = create_initial_game_state()

                # per frame state

                self.observation = None
                self.info = None

                # game duration state

                self.player1_next_buttons = []
                self.player2_next_buttons = []
                self.selection_next_buttons = []
                self.next_buttons_limit = (
                    20  # simply for memory, roughly length of longest combo
                )
                self.player1_current_action = 0
                self.player2_current_action = 0
                self.selection_current_action = 0
                self.actions = {"agent_0": 0, "agent_1": 0}
                self.action_generation = 0

                self.player1_recent_move_names = deque(maxlen=RECENT_MOVE_LIMIT)
                self.player2_recent_move_names = deque(maxlen=RECENT_MOVE_LIMIT)

                # communication

                self.stop_event = asyncio.Event()
                self.frame_buffer = GameFrameBuffer(self.stop_event.is_set)
                self.outbound_message_queue = asyncio.Queue()
                self.connection_event = asyncio.Event()
                self.connection_lost_event = None
                self.runtime_task = None
                self.expiry_handle = None
                self.cleanup_tasks = set()
                self.env_operation_task = None

            def clear_recent_move_history(self):
                self.player1_recent_move_names.clear()
                self.player2_recent_move_names.clear()

            def request_stop(self):
                self.stop_event.set()
                self.connection_event.set()
                if self.connection_lost_event is not None:
                    self.connection_lost_event.set()
                if self.env is not None:
                    self.env.request_stop()

            async def attach_connection(self, connection_lost_event):
                if self.connection_lost_event is not None:
                    self.connection_lost_event.set()
                if self.expiry_handle is not None:
                    self.expiry_handle.cancel()
                    self.expiry_handle = None
                self.connection_lost_event = connection_lost_event
                self.connection_event.set()
                self.outbound_message_queue = asyncio.Queue()
                await self.send_game_state()

            def detach_connection(self, connection_lost_event):
                if self.connection_lost_event is not connection_lost_event:
                    return False
                self.connection_lost_event = None
                self.connection_event.clear()
                self.invalidate_actions()
                return True

            async def wait_for_connection(self):
                await self.connection_event.wait()
                return not self.stop_event.is_set()

            async def run_env_operation(self, func, /, *args, **kwargs):
                if self.env_operation_task is not None:
                    raise RuntimeError("Environment operation already in progress")
                task = asyncio.create_task(asyncio.to_thread(func, *args, **kwargs))
                self.env_operation_task = task
                try:
                    return await asyncio.shield(task)
                finally:
                    if task.done() and self.env_operation_task is task:
                        self.env_operation_task = None

            async def wait_for_env_operation(self):
                task = self.env_operation_task
                if task is None:
                    return
                try:
                    await asyncio.shield(task)
                except Exception:
                    pass
                finally:
                    if self.env_operation_task is task:
                        self.env_operation_task = None

            def enqueue_buttons(self, queue, buttons):
                available = self.next_buttons_limit - len(queue)
                if available > 0:
                    queue.extend(buttons[:available])

            def invalidate_actions(self):
                self.action_generation += 1
                self.player1_next_buttons.clear()
                self.player2_next_buttons.clear()
                self.selection_next_buttons.clear()
                self.player1_current_action = 0
                self.player2_current_action = 0
                self.selection_current_action = 0
                self.actions = {"agent_0": 0, "agent_1": 0}

            def sync_accepts_input_state(self):
                self.game_state["accepts_input"] = self.accepts_input

            def human_player_number(self):
                player1_participant, player2_participant = normalize_game_participants(
                    self.game_settings
                )
                if player1_participant == "human":
                    return 1
                if player2_participant == "human":
                    return 2
                return None

            def next_player_action(self, player: int, participant: str):
                next_buttons = (
                    self.player1_next_buttons
                    if player == 1
                    else self.player2_next_buttons
                )
                if next_buttons:
                    return next_buttons.pop(0)
                if participant == "human":
                    return (
                        self.player1_current_action
                        if player == 1
                        else self.player2_current_action
                    )
                return 0

            def next_selection_action(self):
                if self.selection_next_buttons:
                    return self.selection_next_buttons.pop(0)
                return self.selection_current_action

            async def send_game_state(self):
                self.sync_accepts_input_state()
                self.game_state["player1_participant"] = self.game_settings[
                    "player1Participant"
                ]
                self.game_state["player2_participant"] = self.game_settings[
                    "player2Participant"
                ]
                await self.outbound_message_queue.put(
                    {
                        "type": "game_state",
                        "data": make_json_safe(self.game_state),
                    }
                )

            async def handle_inbound_message(self, data):
                message_type = data.get("type", "unknown")

                if message_type == "start_game":
                    if (
                        self.start_requested
                        or self.game_running
                        or self.game_state["status"] != "pregame"
                    ):
                        return
                    received = data.get("data", {}) or {}
                    self.game_settings["player1Participant"] = received.get(
                        "player1Participant", DEFAULT_PLAYER1_PARTICIPANT
                    )
                    self.game_settings["player2Participant"] = received.get(
                        "player2Participant", DEFAULT_PLAYER2_PARTICIPANT
                    )
                    normalize_game_participants(self.game_settings)
                    self.invalidate_actions()
                    self.accepts_input = False
                    self.start_requested = True
                elif message_type == "player_action":
                    await self.handle_player_action(data["data"])

            async def handle_player_action(self, action_data):
                if not self.accepts_input:
                    return
                if self.observation is None:
                    return

                action = action_data.get("action")
                if not isinstance(action, int):
                    return
                if self.info and self.info.get("selecting"):
                    if action <= 8:
                        self.selection_current_action = action
                    else:
                        self.enqueue_buttons(self.selection_next_buttons, [action])
                    return

                player_number = self.human_player_number()
                if player_number is None:
                    return
                player_key = f"player{player_number}"
                observation_key = f"P{player_number}"
                next_buttons = (
                    self.player1_next_buttons
                    if player_number == 1
                    else self.player2_next_buttons
                )

                # super art

                if action == 18:
                    super_art_name = action_data.get("super_art")
                    if not super_art_name:
                        return

                    player_obs = self.observation[observation_key]
                    character = self.game_settings[player_key]["character"]
                    direction = "left" if player_obs["side"] == 0 else "right"

                    move = SPECIAL_MOVES.get(character, {}).get(super_art_name)
                    if move is not None:
                        self.enqueue_buttons(next_buttons, move[direction])

                # combo

                elif action == 19:
                    combo_name = action_data.get("combo")
                    if not combo_name:
                        return

                    player_obs = self.observation[observation_key]
                    character = self.game_settings[player_key]["character"]
                    direction = "left" if player_obs["side"] == 0 else "right"

                    move = COMBOS.get(character, {}).get(combo_name)
                    if move is not None:
                        self.enqueue_buttons(next_buttons, move[direction])

                # normal move

                else:
                    if action <= 8:  # directional, so don't queue
                        if player_number == 1:
                            self.player1_current_action = action
                        else:
                            self.player2_current_action = action
                    else:  # attack moves (9-17), so queue
                        self.enqueue_buttons(next_buttons, [action])

            async def cleanup_environment(self):
                env = self.env
                if env is None:
                    return
                env.request_stop()
                await self.wait_for_env_operation()
                self.env = None
                try:
                    await asyncio.to_thread(env.close)
                except Exception as exc:
                    print(f"Warning: could not close environment: {exc!r}")

            async def prepare_for_next_game(self):
                self.game_running = False
                self.accepts_input = False
                self.start_requested = False
                self.match_identity = None
                self.game_state = create_initial_game_state()
                self.observation = None
                self.info = None
                self.clear_recent_move_history()
                self.invalidate_actions()

            async def hold_finished(self):
                self.game_running = False
                self.accepts_input = False
                self.start_requested = False
                self.observation = None
                self.info = None
                self.clear_recent_move_history()
                self.invalidate_actions()

            async def fail_game(self, message: str):
                self.game_running = False
                self.accepts_input = False
                self.start_requested = False
                self.game_state["status"] = "error"
                self.game_state["error"] = message
                await self.send_game_state()

            def sync_round_number(self):
                observation = self.observation
                if observation is None:
                    return
                self.game_state["round_number"] = (
                    int(observation["P1"]["wins"][0])
                    + int(observation["P2"]["wins"][0])
                    + 1
                )

            def apply_match_identity(self, identity: dict):
                self.match_identity = identity
                self.game_settings["player1"] = dict(identity["player1"])
                self.game_settings["player2"] = dict(identity["player2"])
                self.game_state["players"] = identity
                self.game_state["match_identity"] = identity
                self.game_state["player1_selection"] = identity["player1"]
                self.game_state["player2_selection"] = identity["player2"]

            async def cleanup(self):
                await self.cleanup_environment()
                if self.cleanup_tasks:
                    await asyncio.wait(
                        tuple(self.cleanup_tasks),
                        timeout=ENV_INIT_TIMEOUT_SECONDS,
                    )

        # routes

        @web_app.websocket("/ws/{game_id}")
        async def websocket_endpoint(websocket: WebSocket, game_id: str):
            if (
                not game_id
                or len(game_id) > 64
                or not all(
                    character.isalnum() or character in "-_" for character in game_id
                )
            ):
                await websocket.close(code=1008)
                return

            await websocket.accept()

            session = self.game_sessions.get(game_id)
            creating_session = False
            if session is None or session.stop_event.is_set():
                if session is None and len(self.game_sessions) >= MAX_GAME_SESSIONS:
                    await websocket.close(code=1013)
                    return
                session = GameSession()
                creating_session = True

            frame_encoder = FrameEncoder()
            video_track = GameVideoTrack(session.frame_buffer)
            control_channel = None
            control_channel_ready = asyncio.Event()
            control_message_queue = asyncio.Queue(maxsize=CONTROL_MESSAGE_QUEUE_LIMIT)
            connection_lost_event = asyncio.Event()
            pc = RTCPeerConnection(configuration=build_rtc_configuration())
            await session.attach_connection(connection_lost_event)

            async def prefetch_required_servers(
                player1_participant: str, player2_participant: str
            ) -> None:
                tasks = [
                    self.create_participant_server(participant)
                    for participant in {player1_participant, player2_participant}
                    if is_model_participant(participant)
                ]
                if tasks:
                    await asyncio.gather(*tasks)

            @pc.on("connectionstatechange")
            async def on_connectionstatechange():
                state = pc.connectionState
                if state in {"closed", "failed"} and not session.stop_event.is_set():
                    print(f"Detaching connection: peer connection {state}")
                    connection_lost_event.set()

            @pc.on("icecandidate")
            async def on_icecandidate(candidate):
                try:
                    if candidate is None:
                        return
                    if websocket.client_state == WebSocketState.DISCONNECTED:
                        return
                    await websocket.send_json(
                        {
                            "type": "ice_candidate",
                            "candidate": {
                                "candidate_sdp": candidate.to_sdp(),
                                "sdpMid": candidate.sdpMid,
                                "sdpMLineIndex": candidate.sdpMLineIndex,
                            },
                        }
                    )
                except Exception:
                    print(f"Error sending ICE candidate: {traceback.format_exc()}")

            @pc.on("datachannel")
            def on_datachannel(channel):
                nonlocal control_channel

                if channel.label != "game_control":
                    return
                control_channel = channel

                @channel.on("open")
                def on_channel_open():
                    control_channel_ready.set()

                if channel.readyState == "open":
                    control_channel_ready.set()

                @channel.on("close")
                def on_channel_close():
                    control_channel_ready.clear()
                    connection_lost_event.set()

                @channel.on("message")
                def on_channel_message(message):
                    try:
                        control_message_queue.put_nowait(message)
                    except asyncio.QueueFull:
                        print("Closing connection with a full control-message queue")
                        connection_lost_event.set()

            async def wait_first(*awaitables, timeout=None):
                tasks = [asyncio.create_task(awaitable) for awaitable in awaitables]
                try:
                    done, _ = await asyncio.wait(
                        tasks,
                        timeout=timeout,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    return done, tasks
                finally:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)

            async def process_control_messages():
                while (
                    not session.stop_event.is_set()
                    and not connection_lost_event.is_set()
                ):
                    done, (get_message_task, stop_task, lost_task) = await wait_first(
                        control_message_queue.get(),
                        session.stop_event.wait(),
                        connection_lost_event.wait(),
                    )
                    if stop_task in done or lost_task in done:
                        break
                    message = get_message_task.result()
                    try:
                        if isinstance(message, bytes):
                            payload = json.loads(message.decode("utf-8"))
                        else:
                            payload = json.loads(message)
                        await session.handle_inbound_message(payload)
                    except Exception:
                        print(
                            f"Error processing datachannel msg: {traceback.format_exc()}"
                        )

            async def process_signaling_messages():
                try:
                    while (
                        not session.stop_event.is_set()
                        and not connection_lost_event.is_set()
                    ):
                        if websocket.client_state == WebSocketState.DISCONNECTED:
                            break
                        done, (receive_task, stop_task, lost_task) = await wait_first(
                            websocket.receive_json(),
                            session.stop_event.wait(),
                            connection_lost_event.wait(),
                        )
                        if stop_task in done or lost_task in done:
                            break
                        data = receive_task.result()
                        message_type = data.get("type", "unknown")

                        if message_type == "offer":
                            offer_sdp = data.get("sdp", "")
                            if "m=video" in offer_sdp and not any(
                                sender.track is video_track
                                for sender in pc.getSenders()
                            ):
                                pc.addTrack(video_track)

                            await pc.setRemoteDescription(
                                RTCSessionDescription(
                                    sdp=data["sdp"],
                                    type=data["type"],
                                )
                            )
                            answer = await pc.createAnswer()
                            await pc.setLocalDescription(answer)
                            await websocket.send_json(
                                {
                                    "type": "answer",
                                    "sdp": pc.localDescription.sdp,
                                }
                            )
                            continue

                        if message_type == "ice_candidate":
                            candidate = data.get("candidate")
                            if candidate and candidate.get("candidate_sdp"):
                                ice_candidate = candidate_from_sdp(
                                    candidate["candidate_sdp"]
                                )
                                ice_candidate.sdpMid = candidate.get("sdpMid")
                                ice_candidate.sdpMLineIndex = candidate.get(
                                    "sdpMLineIndex"
                                )
                                await pc.addIceCandidate(ice_candidate)
                            continue

                        if message_type == "get_turn_servers":
                            await websocket.send_json(build_turn_servers())
                            continue
                except WebSocketDisconnect:
                    if pc.connectionState == "connected":
                        await connection_lost_event.wait()
                    else:
                        print(
                            "Detaching connection: signaling websocket disconnected "
                            f"while peer connection {pc.connectionState}"
                        )
                        connection_lost_event.set()
                except Exception:
                    print(f"Error in signaling processor: {traceback.format_exc()}")
                    connection_lost_event.set()

            async def process_outbound_messages():
                try:
                    while (
                        not session.stop_event.is_set()
                        and not connection_lost_event.is_set()
                    ):
                        (
                            done,
                            (get_message_task, stop_task, lost_task),
                        ) = await wait_first(
                            session.outbound_message_queue.get(),
                            session.stop_event.wait(),
                            connection_lost_event.wait(),
                        )
                        if stop_task in done or lost_task in done:
                            break
                        message = get_message_task.result()
                        while not session.stop_event.is_set():
                            if control_channel and control_channel.readyState == "open":
                                control_channel.send(json.dumps(message))
                                break

                            (
                                done,
                                (_channel_ready_task, stop_task, lost_task),
                            ) = await wait_first(
                                control_channel_ready.wait(),
                                session.stop_event.wait(),
                                connection_lost_event.wait(),
                            )
                            if stop_task in done or lost_task in done:
                                break
                except Exception:
                    print(f"Error in outgoing processor: {traceback.format_exc()}")
                    connection_lost_event.set()

            async def keepalive():
                try:
                    while (
                        not session.stop_event.is_set()
                        and not connection_lost_event.is_set()
                    ):
                        if websocket.client_state != WebSocketState.DISCONNECTED:
                            await websocket.send_json(
                                {
                                    "type": "heartbeat",
                                }
                            )
                        await session.outbound_message_queue.put(
                            {
                                "type": "heartbeat",
                                "data": {},
                            }
                        )
                        done, _ = await wait_first(
                            session.stop_event.wait(),
                            connection_lost_event.wait(),
                            timeout=15,
                        )
                        if done:
                            break
                except Exception:
                    print(f"Error in keepalive: {traceback.format_exc()}")
                    connection_lost_event.set()

            async def fail_current_game(message: str):
                try:
                    await session.fail_game(message)
                finally:
                    await session.cleanup_environment()
                    session.request_stop()

            def format_runtime_error(exc: BaseException) -> str:
                if isinstance(exc, asyncio.TimeoutError):
                    return "A Modal model container timed out."
                message = str(exc)
                if message:
                    return message
                return type(exc).__name__

            event_loop = asyncio.get_running_loop()
            phase_generation = 0
            phase_edge_armed = False

            async def show_models_loading():
                if session.game_state["status"] == "models_loading":
                    return
                session.accepts_input = False
                session.invalidate_actions()
                session.game_state["status"] = "models_loading"
                await session.send_game_state()

            async def wait_for_models(model_ready_task, *, show_loading: bool) -> bool:
                if show_loading and not model_ready_task.done():
                    await session.frame_buffer.wait_for_phase_frames()
                    await show_models_loading()
                try:
                    await model_ready_task
                except Exception as exc:
                    print(f"Model prefetch failed: {exc}")
                    await fail_current_game(format_runtime_error(exc))
                    return False
                if session.game_state["status"] == "models_loading":
                    session.game_state["status"] = "transitioning"
                    await session.send_game_state()
                return True

            def begin_non_fight_phase(generation: int):
                if generation != phase_generation:
                    return
                if not session.game_running:
                    return
                if session.game_state["status"] in {
                    "pregame",
                    "selecting",
                    "models_loading",
                    "finished",
                }:
                    return
                session.accepts_input = False
                session.invalidate_actions()
                session.game_state["status"] = "transitioning"
                asyncio.create_task(session.send_game_state())

            def stream_phase_frame(frame):
                nonlocal phase_edge_armed
                if phase_edge_armed:
                    phase_edge_armed = False
                    generation = phase_generation
                    event_loop.call_soon_threadsafe(begin_non_fight_phase, generation)
                session.frame_buffer.queue_frame(np.ascontiguousarray(frame))

            def send_presentation(name: str, **fields):
                event_loop.call_soon_threadsafe(
                    session.outbound_message_queue.put_nowait,
                    {
                        "type": "presentation",
                        "data": {"name": name, **fields},
                    },
                )

            def human_floor_wait_ms(n_buttons: int, generation_ms: float) -> float:
                if n_buttons < 1:
                    return 0.0
                human_ms = 100.0 + 30.0 * (n_buttons - 1)
                llm_ms = generation_ms + (n_buttons - 1) * (1000.0 / 60.0)
                return max(0.0, human_ms - llm_ms)

            def snapshot_robot_observation():
                obs = session.observation
                if (
                    obs is None
                    or "timer" not in obs
                    or obs["timer"] is None
                    or "frame" not in obs
                ):
                    return None
                player1 = player_state(
                    obs,
                    session.game_settings["player1"],
                    "P1",
                )
                player2 = player_state(
                    obs,
                    session.game_settings["player2"],
                    "P2",
                )
                return player1, player2, frame_encoder.data_url(obs["frame"])

            async def generate_robot_move(player_number, participant, snapshot):
                player1, player2, frame_url = snapshot
                if player_number == 1:
                    args = (
                        player1,
                        player2,
                    )
                    recent_moves = session.player1_recent_move_names
                else:
                    args = (
                        player2,
                        player1,
                    )
                    recent_moves = session.player2_recent_move_names

                started_at = time.perf_counter()
                server = await self.create_participant_server(participant)
                moves, move_name = await asyncio.wait_for(
                    generate_move(
                        server.chat.remote.aio, *args, frame_url, recent_moves
                    ),
                    timeout=1 * MINUTES,
                )
                generation_ms = (time.perf_counter() - started_at) * 1000.0
                return player_number, moves, move_name, generation_ms

            async def run_robot_background():
                try:
                    while not session.stop_event.is_set():
                        if not await session.wait_for_connection():
                            break
                        await asyncio.sleep(0.001)

                        if (
                            not session.game_running
                            or session.observation is None
                            or session.game_state["status"] != "running"
                            or session.match_identity is None
                        ):
                            continue

                        if (
                            "timer" not in session.observation
                            or session.observation["timer"] is None
                        ):  # in case env was just reset
                            continue

                        action_generation = session.action_generation
                        player1_participant, player2_participant = (
                            normalize_game_participants(session.game_settings)
                        )
                        player1_is_model = is_model_participant(player1_participant)
                        player2_is_model = is_model_participant(player2_participant)
                        if not (player1_is_model or player2_is_model):
                            continue

                        snapshot = snapshot_robot_observation()
                        if snapshot is None:
                            continue
                        requests = []
                        if player1_is_model and not session.player1_next_buttons:
                            requests.append(
                                generate_robot_move(
                                    1,
                                    player1_participant,
                                    snapshot,
                                )
                            )
                        if player2_is_model and not session.player2_next_buttons:
                            requests.append(
                                generate_robot_move(
                                    2,
                                    player2_participant,
                                    snapshot,
                                )
                            )
                        if not requests:
                            continue

                        results = await asyncio.gather(*requests)
                        if action_generation != session.action_generation:
                            continue
                        wait_ms = max(
                            human_floor_wait_ms(len(moves), generation_ms)
                            for _, moves, _, generation_ms in results
                        )
                        if wait_ms > 0:
                            await asyncio.sleep(wait_ms / 1000.0)
                            if action_generation != session.action_generation:
                                continue

                        for player_number, moves, move_name, _ in results:
                            if player_number == 1:
                                next_buttons = session.player1_next_buttons
                                recent_move_names = session.player1_recent_move_names
                            else:
                                next_buttons = session.player2_next_buttons
                                recent_move_names = session.player2_recent_move_names
                            session.enqueue_buttons(next_buttons, moves)
                            recent_move_names.append(move_name)

                except WebSocketDisconnect:
                    session.request_stop()
                except Exception as exc:
                    print(f"Error in robot background: {traceback.format_exc()}")
                    await fail_current_game(format_runtime_error(exc))
                    session.request_stop()

            async def create_interactive_env():
                env_config = EnvironmentConfig(
                    characters=("Ken", "Ryu"),
                    outfits=(1, 1),
                    super_arts=(1, 1),
                    step_ratio=1,
                    render_mode="rgb_array",
                    vs_cpu=False,
                    cpu_difficulty=MAX_CPU_DIFFICULTY,
                    interactive_select=True,
                )
                environment_task = asyncio.create_task(
                    asyncio.to_thread(create_environment, env_config)
                )

                async def close_abandoned_environment():
                    try:
                        abandoned = await environment_task
                        await asyncio.to_thread(abandoned.close)
                    except Exception as exc:
                        print(
                            f"Warning: could not close abandoned environment: {exc!r}"
                        )

                def schedule_abandoned_cleanup():
                    cleanup_task = asyncio.create_task(close_abandoned_environment())
                    session.cleanup_tasks.add(cleanup_task)
                    cleanup_task.add_done_callback(session.cleanup_tasks.discard)

                try:
                    env = await asyncio.wait_for(
                        asyncio.shield(environment_task),
                        timeout=ENV_INIT_TIMEOUT_SECONDS,
                    )
                except asyncio.CancelledError:
                    schedule_abandoned_cleanup()
                    raise
                except Exception as e:
                    if isinstance(e, asyncio.TimeoutError):
                        schedule_abandoned_cleanup()
                    raise
                return env

            ENV_INIT_ATTEMPTS = 3
            ENV_INIT_BACKOFF_S = (0.5, 1.0, 2.0)

            async def bring_up_interactive_env(pregame_frame_sink):
                last_error = None
                for attempt in range(1, ENV_INIT_ATTEMPTS + 1):
                    env = None
                    try:
                        env = await create_interactive_env()
                        session.env = env
                        raw = await session.run_env_operation(
                            env.pregame_step,
                            pregame_frame_sink,
                        )
                        return env, raw
                    except Exception as e:
                        last_error = e
                        print(
                            "Error bringing up local environment "
                            f"(attempt {attempt}/{ENV_INIT_ATTEMPTS}): {e}"
                        )
                        if env is not None:
                            await session.cleanup_environment()
                        if attempt < ENV_INIT_ATTEMPTS:
                            await asyncio.sleep(ENV_INIT_BACKOFF_S[attempt - 1])
                raise last_error

            async def run_game_loop():
                nonlocal phase_edge_armed, phase_generation
                frame_interval = 1.0 / session.frame_buffer.target_fps

                async def pace_frame(last_frame_at):
                    if not await session.wait_for_connection():
                        return None
                    if last_frame_at is not None:
                        deadline = last_frame_at + frame_interval
                        while (
                            delay := deadline - asyncio.get_running_loop().time()
                        ) > 0:
                            await asyncio.sleep(delay)
                    return asyncio.get_running_loop().time()

                try:
                    while not session.stop_event.is_set():
                        if not await session.wait_for_connection():
                            break
                        await session.cleanup_environment()
                        stream_pregame_frames = (
                            session.frame_buffer.latest_frame is None
                        )
                        pregame_frame_sink = (
                            stream_phase_frame if stream_pregame_frames else None
                        )
                        try:
                            session.env, raw = await bring_up_interactive_env(
                                pregame_frame_sink
                            )
                        except Exception as e:
                            print(f"Error creating local environment: {e}")
                            await fail_current_game(format_runtime_error(e))
                            return

                        session.match_identity = None
                        session.accepts_input = False
                        session.game_running = False
                        session.start_requested = False

                        initial_frame = raw.get("frame")
                        if stream_pregame_frames and initial_frame is not None:
                            session.frame_buffer.set_frame(
                                np.ascontiguousarray(initial_frame)
                            )

                        session.game_state["status"] = "pregame"
                        session.game_state["winner"] = ""
                        session.game_state["winner_side"] = ""
                        session.game_state["player1_selection"] = None
                        session.game_state["player2_selection"] = None
                        session.game_state["error"] = ""
                        await session.send_game_state()

                        last_frame_at = None
                        while (
                            not session.start_requested
                            and not session.stop_event.is_set()
                        ):
                            last_frame_at = await pace_frame(last_frame_at)
                            if session.stop_event.is_set():
                                break
                            try:
                                raw = await session.run_env_operation(
                                    session.env.pregame_step,
                                    pregame_frame_sink,
                                )
                            except Exception as e:
                                if session.stop_event.is_set():
                                    break
                                print(f"Error during pregame step: {e}")
                                await fail_current_game(format_runtime_error(e))
                                session.request_stop()
                                break
                            session.observation = {
                                "frame": raw.get("frame"),
                            }
                            frame = session.observation.get("frame")
                            if stream_pregame_frames and frame is not None:
                                session.frame_buffer.set_frame(
                                    np.ascontiguousarray(frame)
                                )

                        if session.stop_event.is_set():
                            break

                        player1_participant, player2_participant = (
                            normalize_game_participants(session.game_settings)
                        )
                        human_player_number = session.human_player_number()
                        vs_cpu = player2_participant == "cpu"
                        model_ready_task = asyncio.create_task(
                            prefetch_required_servers(
                                player1_participant,
                                player2_participant,
                            )
                        )
                        models_loading_delay_frames = max(
                            1, round(session.frame_buffer.target_fps)
                        )
                        models_loading_frames_remaining = None
                        models_ready_checked = False

                        session.game_running = True
                        session.accepts_input = False
                        session.game_state["status"] = "starting"
                        await session.send_game_state()

                        try:
                            (
                                session.observation,
                                session.info,
                            ) = await session.run_env_operation(
                                session.env.start_interactive_game,
                                vs_cpu=vs_cpu,
                                cpu_difficulty=MAX_CPU_DIFFICULTY,
                                frame_sink=stream_phase_frame,
                                presentation_sink=send_presentation,
                            )
                        except Exception as e:
                            print(f"Error starting interactive game: {e}")
                            await fail_current_game(format_runtime_error(e))
                            return

                        await session.frame_buffer.wait_for_phase_frames()
                        frame = session.observation.get("frame")
                        if frame is not None:
                            session.frame_buffer.set_frame(np.ascontiguousarray(frame))

                        session.accepts_input = True
                        session.game_state["status"] = "selecting"
                        await session.send_game_state()

                        last_frame_at = None
                        while session.game_running and not session.stop_event.is_set():
                            last_frame_at = await pace_frame(last_frame_at)
                            if session.stop_event.is_set():
                                break

                            selecting = bool(
                                session.info and session.info.get("selecting")
                            )
                            if selecting:
                                session.actions = {
                                    "selection": session.next_selection_action(),
                                }
                            else:
                                session.actions = {
                                    "agent_0": session.next_player_action(
                                        1, player1_participant
                                    ),
                                    "agent_1": session.next_player_action(
                                        2, player2_participant
                                    ),
                                }

                            try:
                                phase_edge_armed = True
                                (
                                    session.observation,
                                    reward,
                                    terminated,
                                    truncated,
                                    session.info,
                                ) = await session.run_env_operation(
                                    session.env.step,
                                    session.actions,
                                    stream_phase_frame,
                                )
                            except Exception as e:
                                if session.stop_event.is_set():
                                    break
                                print(f"Error during env.step: {e}")
                                await fail_current_game(format_runtime_error(e))
                                break
                            finally:
                                phase_edge_armed = False
                                phase_generation += 1

                            frame = session.observation.get("frame")
                            if frame is not None:
                                session.frame_buffer.set_frame(
                                    np.ascontiguousarray(frame)
                                )

                            selection_changed = False
                            if selecting:
                                for player_number in (1, 2):
                                    selection_key = f"player{player_number}_selection"
                                    selected_key = f"player{player_number}_selected"
                                    if (
                                        session.info.get(selected_key)
                                        and session.game_state[selection_key] is None
                                    ):
                                        session.game_state[selection_key] = (
                                            session.env.read_player_identity(
                                                f"P{player_number}"
                                            )
                                        )
                                        selection_changed = True
                            both_selected_while_selecting = bool(
                                selecting
                                and session.info.get("player1_selected")
                                and session.info.get("player2_selected")
                            )
                            if both_selected_while_selecting:
                                if (
                                    not models_ready_checked
                                    and models_loading_frames_remaining is None
                                ):
                                    models_loading_frames_remaining = (
                                        models_loading_delay_frames
                                    )
                                if session.accepts_input:
                                    session.accepts_input = False
                                    session.invalidate_actions()
                                    selection_changed = True
                            if selection_changed:
                                await session.send_game_state()

                            if (
                                selecting
                                and models_loading_frames_remaining is not None
                            ):
                                models_loading_frames_remaining -= 1
                                if models_loading_frames_remaining <= 0:
                                    models_loading_frames_remaining = None
                                    if not model_ready_task.done() and not (
                                        await wait_for_models(
                                            model_ready_task,
                                            show_loading=True,
                                        )
                                    ):
                                        return
                                    models_ready_checked = True
                                    if session.game_state["status"] == "models_loading":
                                        session.game_state["status"] = "selecting"
                                        await session.send_game_state()

                            if selecting and not session.info.get("selecting"):
                                identity = session.env.read_match_identity()
                                if not await wait_for_models(
                                    model_ready_task,
                                    show_loading=not model_ready_task.done(),
                                ):
                                    return
                                models_ready_checked = True
                                session.apply_match_identity(identity)
                                session.accepts_input = human_player_number is not None
                                session.sync_round_number()
                                session.game_state["status"] = "running"
                                await session.send_game_state()

                            if session.info.get("game_done", False):
                                if terminated or truncated:
                                    await session.frame_buffer.wait_for_phase_frames()
                                    session.accepts_input = False
                                    session.invalidate_actions()
                                    p1_wins = session.observation["P1"]["wins"][0]
                                    p2_wins = session.observation["P2"]["wins"][0]

                                    if p1_wins > p2_wins:
                                        session.game_state["scores"][0] += 1
                                        winner_side = "P1"
                                        winner = PARTICIPANT_LABELS.get(
                                            player1_participant,
                                            player1_participant,
                                        )
                                        if player1_participant == player2_participant:
                                            winner = f"{winner} (P1)"
                                    elif p2_wins > p1_wins:
                                        session.game_state["scores"][1] += 1
                                        winner_side = "P2"
                                        winner = PARTICIPANT_LABELS.get(
                                            player2_participant,
                                            player2_participant,
                                        )
                                        if player2_participant == player1_participant:
                                            winner = f"{winner} (P2)"
                                    else:
                                        winner_side = "draw"
                                        winner = "Draw"

                                    session.game_state["status"] = "finished"
                                    session.game_state["winner"] = winner
                                    session.game_state["winner_side"] = winner_side
                                    await session.send_game_state()
                                    await session.hold_finished()
                                    break
                            elif session.info.get("round_done", False):
                                session.invalidate_actions()
                                session.clear_recent_move_history()
                                if session.info.get("stage_done", False):
                                    identity = session.env.read_match_identity()
                                    session.apply_match_identity(identity)
                                session.accepts_input = human_player_number is not None
                                session.sync_round_number()
                                session.game_state["status"] = "running"
                                last_frame_at = None
                                await session.send_game_state()

                        if session.stop_event.is_set():
                            break
                        await session.prepare_for_next_game()

                except Exception:
                    print(f"Error in game loop: {traceback.format_exc()}")
                    session.request_stop()

            if session.runtime_task is None:

                async def run_runtime():
                    tasks = [
                        asyncio.create_task(run_robot_background()),
                        asyncio.create_task(run_game_loop()),
                    ]
                    try:
                        done, _ = await asyncio.wait(
                            tasks,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for task in done:
                            task.result()
                    except Exception as exc:
                        print(f"Game runtime error: {exc}")
                    finally:
                        session.request_stop()
                        session.game_running = False
                        if session.expiry_handle is not None:
                            session.expiry_handle.cancel()
                            session.expiry_handle = None
                        _, pending_tasks = await asyncio.wait(
                            tasks,
                            timeout=SESSION_TASK_SHUTDOWN_TIMEOUT_SECONDS,
                        )
                        for task in pending_tasks:
                            task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                        await session.cleanup()
                        if self.game_sessions.get(game_id) is session:
                            self.game_sessions.pop(game_id, None)

                if creating_session:
                    session.runtime_task = asyncio.create_task(run_runtime())
                    self.game_sessions[game_id] = session

            tasks = [
                asyncio.create_task(process_signaling_messages()),
                asyncio.create_task(process_control_messages()),
                asyncio.create_task(process_outbound_messages()),
                asyncio.create_task(keepalive()),
            ]
            stop_waiter = asyncio.create_task(session.stop_event.wait())
            lost_waiter = asyncio.create_task(connection_lost_event.wait())
            waiters = [stop_waiter, lost_waiter]

            try:
                done, _ = await asyncio.wait(
                    {*tasks, *waiters},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    if task in tasks:
                        task.result()
            except WebSocketDisconnect:
                connection_lost_event.set()
            except Exception as exc:
                print(f"WebSocket connection error: {exc}")
                connection_lost_event.set()
            finally:
                for task in (*tasks, *waiters):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, *waiters, return_exceptions=True)
                try:
                    await pc.close()
                finally:
                    if session.detach_connection(connection_lost_event):
                        if not session.stop_event.is_set():
                            session.expiry_handle = (
                                asyncio.get_running_loop().call_later(
                                    GAMEPLAY_SESSION_IDLE_TIMEOUT_SECONDS,
                                    session.request_stop,
                                )
                            )

        @web_app.websocket("/ws")
        async def websocket_missing_game_id(websocket: WebSocket):
            await websocket.close(code=1008)

        @web_app.get("/api/extra-moves")
        async def get_extra_moves():
            return JSONResponse(
                make_json_safe({"combos": COMBOS, "special_moves": SPECIAL_MOVES}),
                headers={"Cache-Control": "public, max-age=300"},
            )

        no_cache_header = "no-store, max-age=0"
        no_cache_suffixes = (".html", ".js", ".css")
        frontend_root = Path(remote_frontend_dir).resolve()

        @web_app.get("/favicon.ico")
        async def favicon():
            return FileResponse(
                f"{remote_logos_dir}/mobile.webp",
                media_type="image/webp",
            )

        web_app.mount("/icons", StaticFiles(directory=remote_icons_dir), name="icons")
        web_app.mount("/logos", StaticFiles(directory=remote_logos_dir), name="logos")
        web_app.mount(
            "/sounds", StaticFiles(directory=remote_sounds_dir), name="sounds"
        )

        def frontend_file_response(frontend_path: str):
            path = (frontend_root / frontend_path).resolve()
            if not path.is_relative_to(frontend_root):
                return Response(status_code=404)
            if not path.is_file():
                return Response(status_code=404)
            cache_header = (
                no_cache_header
                if path.suffix.lower() in no_cache_suffixes
                else "public, max-age=31536000, immutable"
            )
            return FileResponse(path, headers={"Cache-Control": cache_header})

        @web_app.get("/")
        async def index():
            return frontend_file_response("index.html")

        @web_app.get("/{frontend_path:path}")
        async def frontend_file(frontend_path: str):
            return frontend_file_response(frontend_path)

        return web_app


@app.function(
    image=launcher_image,
    region=CONTAINER_REGION,
    min_containers=1,
)
@modal.concurrent(max_inputs=96, target_inputs=64)
@modal.asgi_app(label="sf3", custom_domains=["sf3.modal.dev"])
def launcher():
    from urllib.parse import urlencode

    from fastapi import FastAPI, Request
    from fastapi.responses import RedirectResponse, Response

    web_app = FastAPI()

    @web_app.get("/")
    async def start_session(request: Request):
        if (
            request.headers.get("sec-purpose") is not None
            or request.headers.get("sec-fetch-mode", "navigate") != "navigate"
        ):
            return Response(status_code=204)
        session = await Web.sessions.start.aio(
            idle_timeout=GAMEPLAY_SESSION_IDLE_TIMEOUT_SECONDS
        )
        gameplay_url = (await Web.get_url.aio()).rstrip("/")
        query = urlencode({"modal_session_token": session.token})
        return RedirectResponse(
            f"{gameplay_url}/?{query}",
            status_code=303,
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    return web_app
