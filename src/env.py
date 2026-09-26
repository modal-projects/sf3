from __future__ import annotations

import hashlib
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from src.utils import STUN_BAR_MAX, SUPER_BAR_MAX, TIMER_MAX

_ROM_FILENAME = "sfiii3n.zip"
_ROM_SHA256 = "7239b5eb005488db22ace477501c574e9420c0ab70aeeb0795dfeb474284d416"
_CPU_CHARACTER_MENU = 2
_CPU_OPPONENT_MENUS = {3, 9}
_CPU_OPPONENT_CONFIRM_DELAY = 30
_CPU_OPPONENT_CONFIRM_INTERVAL = 30
_CAPCOM_PRESENTATION_OFFSET_FRAMES = 345
_INSERT_COIN_PRESENTATION_FRAMES = 600
_SELECTION_PRESENTATION_FRAMES = 300
_TERMINAL_MIN_SECONDS = 20
_TOURNAMENT_WIN_MIN_SECONDS = 60
_TERMINAL_MAX_SECONDS = 120
_TERMINAL_IDLE_SECONDS = 5
FrameSink = Callable[[Any], None]
PresentationSink = Callable[..., None]

# Raw emulator ids used by MAME / sfiii-gym.
# Intentionally different than CHARACTER_TO_ID in utils.py
# since the emulator uses a different numbering scheme.
CHARACTER_NAME_TO_LOCAL_ID = {
    "Gill": 0,
    "Alex": 1,
    "Ryu": 2,
    "Yun": 3,
    "Dudley": 4,
    "Necro": 5,
    "Hugo": 6,
    "Ibuki": 7,
    "Elena": 8,
    "Oro": 9,
    "Yang": 10,
    "Ken": 11,
    "Sean": 12,
    "Urien": 13,
    "Gouki": 14,
    "Chun-Li": 16,
    "Makoto": 17,
    "Q": 18,
    "Twelve": 19,
    "Remy": 20,
}
LOCAL_ID_TO_CHARACTER_NAME = {
    local_id: name for name, local_id in CHARACTER_NAME_TO_LOCAL_ID.items()
}


def _scalar(value: Any) -> int:
    import numpy as np

    if isinstance(value, np.ndarray):
        return int(value.reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        return int(value[0])
    return int(value)


def _array1(value: int, *, dtype: Any = None) -> Any:
    import numpy as np

    return np.array([value], dtype=dtype or np.int16)


def _normalize_player(
    raw: Mapping[str, Any],
    *,
    prefix: str,
    wins: int,
) -> dict[str, Any]:
    side = _scalar(raw[f"side{prefix}"])
    stun_timer = _scalar(raw[f"stunTimer{prefix}"])
    stun_bar = _scalar(raw[f"stunBar{prefix}"])
    super_bar = _scalar(raw[f"superGauge{prefix}"])
    super_count = _scalar(raw[f"superCount{prefix}"])
    health = _scalar(raw[f"health{prefix}"])

    return {
        "wins": _array1(wins),
        "side": side,
        "stunned": stun_timer > 0,
        "stun_bar": _array1(max(0, min(STUN_BAR_MAX, stun_bar))),
        "health": _array1(health),
        "super_count": _array1(max(0, super_count)),
        "super_bar": _array1(max(0, min(SUPER_BAR_MAX, super_bar))),
    }


def _normalize_local_observation(raw: Mapping[str, Any]) -> dict[str, Any]:
    timer = max(0, min(TIMER_MAX, _scalar(raw["timer"])))
    wins_p1 = _scalar(raw["winsP1"])
    wins_p2 = _scalar(raw["winsP2"])

    return {
        "frame": raw["frame"],
        "timer": _array1(timer),
        "P1": _normalize_player(raw, prefix="P1", wins=wins_p1),
        "P2": _normalize_player(raw, prefix="P2", wins=wins_p2),
    }


def arcade_level(difficulty: int) -> int:
    """Map public difficulties 1..8 to arcade menu levels 0..7."""
    return max(0, min(7, int(difficulty) - 1))


def _menu_steps(
    frame_ratio: int,
    entries: Sequence[tuple[int, tuple[Any, ...]]],
) -> list[dict[str, Any]]:
    return [
        {"wait": int(wait / frame_ratio), "actions": list(actions)}
        for wait, actions in entries
    ]


def _boot_steps(frame_ratio: int, *, vs_cpu: bool = False):
    # MAME is container-only; local imports must not load its native wheel.
    from MAMEToolkit.sf_environment.Actions import Actions

    coin = (Actions.COIN_P1,) if vs_cpu else (Actions.COIN_P1, Actions.COIN_P2)
    start = (Actions.P1_START,) if vs_cpu else (Actions.P1_START, Actions.P2_START)
    steps = _menu_steps(
        frame_ratio,
        [
            (0, (Actions.SERVICE,)),
            (30, (Actions.P1_UP,)),
            (30, (Actions.P1_JPUNCH,)),
            (300 + _INSERT_COIN_PRESENTATION_FRAMES, coin),
            (10, coin),
            (60, start),
        ],
    )
    intro_step = steps[3]
    intro_step["presentations"] = {
        int(_CAPCOM_PRESENTATION_OFFSET_FRAMES / frame_ratio): "capcom",
    }
    intro_step["action_presentation"] = "coin"
    return steps


def _cpu_difficulty_steps(frame_ratio: int, difficulty: int):
    from MAMEToolkit.sf_environment.Actions import Actions

    up = Actions.P1_UP
    down = Actions.P1_DOWN
    jab = Actions.P1_JPUNCH
    entries = [
        (0, (Actions.SERVICE,)),
        *([(10, (up,))] * 4),
        (10, (jab,)),
        (10, (down,)),
        (10, (down,)),
        (10, (jab, Actions.P1_FPUNCH)),
        (10, (up,)),
        (10, (jab,)),
    ]
    level = arcade_level(difficulty)
    direction = Actions.P1_LEFT if level < 3 else Actions.P1_RIGHT
    entries.extend((10, (direction,)) for _ in range(abs(level - 3)))
    tail = [down] * 6 + [jab, down, jab, down, down, jab, down, down, down, jab]
    entries.extend((10, (action,)) for action in tail)
    return _menu_steps(frame_ratio, entries)


def _next_stage_steps(frame_ratio: int):
    # MAME is container-only; local imports must not load its native wheel.
    from MAMEToolkit.sf_environment.Actions import Actions

    jab = Actions.P1_JPUNCH
    entries = [(60, (jab,))]
    entries.extend((0, (jab,)) for _ in range(int(180 / frame_ratio)))
    entries.append((60, (jab,)))
    return _menu_steps(frame_ratio, entries)


def _action_values(player: int, action_id: int):
    from MAMEToolkit.sf_environment.Actions import Actions

    prefix = f"P{player}"
    action_names = {
        0: (),
        1: ("LEFT",),
        2: ("LEFT", "UP"),
        3: ("UP",),
        4: ("UP", "RIGHT"),
        5: ("RIGHT",),
        6: ("RIGHT", "DOWN"),
        7: ("DOWN",),
        8: ("DOWN", "LEFT"),
        9: ("JPUNCH",),
        10: ("SPUNCH",),
        11: ("FPUNCH",),
        12: ("SKICK",),
        13: ("FKICK",),
        14: ("RKICK",),
        15: ("JPUNCH", "SKICK"),
        16: ("SPUNCH", "FKICK"),
        17: ("FPUNCH", "RKICK"),
    }.get(int(action_id), ())
    return [getattr(Actions, f"{prefix}_{name}").value for name in action_names]


@dataclass(frozen=True)
class EnvironmentConfig:
    characters: tuple[str, str]
    outfits: tuple[int, int]
    super_arts: tuple[int, int]
    step_ratio: int
    render_mode: str = "rgb_array"
    roms_path: str = "/root"
    env_id: str | None = None
    vs_cpu: bool = False
    cpu_difficulty: int = 8
    interactive_select: bool = False

    def resolved_env_id(self) -> str:
        return self.env_id or f"sf3-{uuid.uuid4().hex[:8]}"


@dataclass
class _CpuMenuAdvanceState:
    pending_p1_states: set[int]
    opponent_menu_frames: int = 0


class _FramePacer:
    def __init__(self, target_fps: float = 60.0):
        self.frame_interval = 1.0 / target_fps
        self.last_frame_at: float | None = None

    def wait(self) -> None:
        if self.last_frame_at is not None:
            deadline = self.last_frame_at + self.frame_interval
            while (delay := deadline - time.perf_counter()) > 0:
                time.sleep(delay)
        self.last_frame_at = time.perf_counter()


class GameEnvironment(Protocol):
    def pregame_step(
        self,
        frame_sink: FrameSink | None = None,
    ) -> dict[str, Any]: ...

    def start_interactive_game(
        self,
        *,
        vs_cpu: bool,
        cpu_difficulty: int,
        auto_select: bool = False,
        frame_sink: FrameSink | None = None,
        presentation_sink: PresentationSink | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]: ...

    def reset(
        self,
        frame_sink: FrameSink | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]: ...

    def step(
        self,
        actions: dict[str, int],
        frame_sink: FrameSink | None = None,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]: ...

    def read_match_identity(self) -> dict[str, dict[str, Any]]: ...

    def read_player_identity(self, player: str) -> dict[str, Any]: ...

    def request_stop(self) -> None: ...

    def close(self) -> None: ...


class LocalSfiiiAdapter:
    """Local emulator adapter for automated or interactive matches."""

    def __init__(self, config: EnvironmentConfig):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp")

        rom_path = Path(config.roms_path) / _ROM_FILENAME
        if not rom_path.is_file():
            raise FileNotFoundError(f"Local SFIII runtime expected ROM at {rom_path}")

        actual_sha256 = hashlib.sha256(rom_path.read_bytes()).hexdigest()
        if actual_sha256 != _ROM_SHA256:
            raise ValueError(
                f"ROM '{_ROM_FILENAME}' failed SHA256 checksum verification. "
                f"Expected {_ROM_SHA256}, got {actual_sha256}"
            )

        from MAMEToolkit import emulator
        from MAMEToolkit.emulator import Address, Emulator

        Path(emulator.__file__).with_name("mame").joinpath("pipes").mkdir(exist_ok=True)
        self.config = config
        self.memory_addresses = {
            "fighting": Address("0x02011389", "u8"),
            "winsP1": Address("0x02011383", "u8"),
            "winsP2": Address("0x02011385", "u8"),
            "timer": Address("0x02011377", "u8"),
            "healthP1": Address("0x02068D0A", "s16"),
            "healthP2": Address("0x020691A2", "s16"),
            "sideP1": Address("0x02068C76", "u8"),
            "sideP2": Address("0x0206910E", "u8"),
            "superGaugeP1": Address("0x020695B5", "u8"),
            "superCountP1": Address("0x020695BF", "u8"),
            "superGaugeP2": Address("0x020695E1", "u8"),
            "superCountP2": Address("0x020695EB", "u8"),
            "stunTimerP1": Address("0x020695F9", "u8"),
            "stunBarP1": Address("0x020695FD", "u32"),
            "stunTimerP2": Address("0x0206960D", "u8"),
            "stunBarP2": Address("0x02069611", "u32"),
            "characterP1": Address("0x02011387", "u8"),
            "characterP2": Address("0x02011388", "u8"),
            "menuState": Address("0x0201546B", "u8"),
            "characterSelectStateP1": Address("0x0201553D", "u8"),
            "characterSelectStateP2": Address("0x02015545", "u8"),
            "characterSelectSaP1": Address("0x020154D3", "u8"),
            "characterSelectSaP2": Address("0x020154D5", "u8"),
            "characterSelectColorP1": Address("0x02015683", "u8"),
            "characterSelectColorP2": Address("0x02015684", "u8"),
        }

        self.emu = Emulator.__new__(Emulator)
        try:
            Emulator.__init__(
                self.emu,
                config.resolved_env_id(),
                config.roms_path,
                "sfiii3n",
                self.memory_addresses,
                frame_ratio=config.step_ratio,
                render=config.render_mode == "human",
                throttle=False,
                frame_skip=0,
                sound=False,
                debug=False,
                binary_path=None,
            )
        except BaseException:
            console = getattr(self.emu, "console", None)
            if console is not None:
                console.process.kill()
            raise
        self._data: dict[str, Any] = {}
        self.expected_health = {"P1": 0, "P2": 0}
        self.expected_wins = {"P1": 0, "P2": 0}
        self._closed = False
        self._reset_called = False
        self._stop_requested = threading.Event()
        self._matchup_character_lock_installed = False
        self._cpu_opponent_menu_seen = False
        self._selecting = False
        # Per-game: one auto-selected game must not stop a later game on the
        # same adapter from parking in the character-select UI.
        self._interactive_select = config.interactive_select
        self._vs_cpu = config.vs_cpu
        self._cpu_difficulty = config.cpu_difficulty
        self._match_identity: dict[str, dict[str, Any]] | None = None
        self._presentation_sink: PresentationSink | None = None
        self._versus_presented = False
        self._frame_pacer = _FramePacer()

    def pregame_step(
        self,
        frame_sink: FrameSink | None = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Cannot step a closed LocalSfiiiAdapter")
        self._data = self._phase_step([], frame_sink)
        return self._data

    def start_interactive_game(
        self,
        *,
        vs_cpu: bool,
        cpu_difficulty: int,
        auto_select: bool = False,
        frame_sink: FrameSink | None = None,
        presentation_sink: PresentationSink | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._closed:
            raise RuntimeError("Cannot start a closed LocalSfiiiAdapter")
        if not self.config.interactive_select:
            raise RuntimeError(
                "Interactive game requested from an automated environment"
            )
        # auto_select reuses the automated selection and character-lock path so
        # the configured matchup is seated without any UI cursor input.
        self._interactive_select = not auto_select
        self._vs_cpu = vs_cpu
        self._cpu_difficulty = max(1, min(8, int(cpu_difficulty)))
        self._match_identity = None
        self._cpu_opponent_menu_seen = False
        self._selecting = False
        self._presentation_sink = presentation_sink
        self._new_game(frame_sink, presentation_sink)
        return _normalize_local_observation(self._data), {
            "selecting": self._selecting,
            "player1_selected": False,
            "player2_selected": False,
            "game_done": False,
            "round_done": False,
            "stage_done": False,
            "winner": None,
        }

    def reset(
        self,
        frame_sink: FrameSink | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._closed:
            raise RuntimeError("Cannot reset a closed LocalSfiiiAdapter")
        if self._reset_called:
            raise RuntimeError(
                "LocalSfiiiAdapter supports one reset; create a new environment "
                "for each game"
            )
        self._reset_called = True
        self._interactive_select = self.config.interactive_select
        self._vs_cpu = self.config.vs_cpu
        self._cpu_difficulty = self.config.cpu_difficulty
        self._match_identity = None
        self._cpu_opponent_menu_seen = False
        self._new_game(frame_sink)
        return _normalize_local_observation(self._data), {
            "selecting": self._selecting,
            "player1_selected": False,
            "player2_selected": False,
            "game_done": False,
            "round_done": False,
            "stage_done": False,
            "winner": None,
        }

    def step(
        self,
        actions: dict[str, int],
        frame_sink: FrameSink | None = None,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._selecting:
            return self._step_select(actions, frame_sink)

        pressed = _action_values(1, actions.get("agent_0", 0))
        if not self._vs_cpu:
            pressed = pressed + _action_values(2, actions.get("agent_1", 0))
        raw = self._sub_step(pressed)
        reward = float(raw["reward"])
        round_done = False
        stage_done = False
        game_done = False
        winner = None

        if int(raw["fighting"]) == 0:
            # Equal health at the fighting edge is time-over or double KO.
            if int(raw["healthP1"]) == int(raw["healthP2"]):
                self._emit_presentation("judgement")
            raw = self._run_till_victor(raw, frame_sink)
            reward = float(raw["reward"])
            round_done = True
            p1_won_match = int(raw["winsP1"]) >= 2
            p2_won_match = int(raw["winsP2"]) >= 2
            tournament_mode = self._vs_cpu
            tournament_won = (
                tournament_mode
                and p1_won_match
                and int(raw["characterP2"]) == CHARACTER_NAME_TO_LOCAL_ID["Gill"]
            )
            stage_done = tournament_mode and p1_won_match and not tournament_won
            game_done = (
                p2_won_match or tournament_won
                if tournament_mode
                else p1_won_match or p2_won_match
            )
            if game_done:
                if tournament_mode:
                    winner = "P1" if tournament_won else "P2"
                else:
                    # double KO on the deciding round awards the round to both
                    # fighters at once, so rank on wins -> remaining health
                    p1_score = (int(raw["winsP1"]), int(raw["healthP1"]))
                    p2_score = (int(raw["winsP2"]), int(raw["healthP2"]))
                    if p1_score > p2_score:
                        winner = "P1"
                    elif p2_score > p1_score:
                        winner = "P2"
                    else:
                        winner = "draw"
            if stage_done:
                self._data = self._wait_for_post_ko_black_frame(raw, frame_sink)
                self._emit_presentation("winner")
                self._run_steps(
                    _next_stage_steps(self.config.step_ratio),
                    frame_sink,
                )
                if int(self._data["characterP2"]) == CHARACTER_NAME_TO_LOCAL_ID["Gill"]:
                    self._emit_presentation("gill_intro")
                self._data = self._wait_for_fight_start(frame_sink)
                self._data["reward"] = reward
            elif not game_done:
                self._data = self._wait_for_fight_start(frame_sink)
                self._data["reward"] = reward
            else:
                self._data = self._stream_terminal_sequence(
                    raw,
                    frame_sink,
                    tournament_won=tournament_won,
                    winner=winner,
                )
                self._data["reward"] = reward
        else:
            self._data = raw

        observation = _normalize_local_observation(self._data)
        info = {
            "game_done": game_done,
            "round_done": round_done,
            "stage_done": stage_done,
            "selecting": False,
            "winner": winner,
        }
        return observation, reward, game_done, False, info

    def read_match_identity(self) -> dict[str, dict[str, Any]]:
        if self._match_identity is not None:
            return {
                player: dict(settings)
                for player, settings in self._match_identity.items()
            }

        return {
            "player1": self.read_player_identity("P1"),
            "player2": self.read_player_identity("P2"),
        }

    def read_player_identity(self, player: str) -> dict[str, Any]:
        if player not in {"P1", "P2"}:
            raise ValueError(f"Unknown player seat {player}")
        local_id = int(self._data[f"character{player}"])
        character = LOCAL_ID_TO_CHARACTER_NAME.get(local_id)
        if character is None:
            raise RuntimeError(f"Unknown MAME character id {local_id} for {player}")
        return {
            "character": character,
            "outfit": max(
                1,
                min(
                    6,
                    int(self._data[f"characterSelectColor{player}"]) + 1,
                ),
            ),
            "superArt": max(
                1,
                min(
                    3,
                    int(self._data[f"characterSelectSa{player}"]) + 1,
                ),
            ),
        }

    def request_stop(self) -> None:
        self._stop_requested.set()

    def close(self) -> None:
        if self._closed:
            return
        self.request_stop()
        self._closed = True
        self.emu.close()

    def _raise_if_stopping(self) -> None:
        if self._stop_requested.is_set():
            raise RuntimeError("Environment stop requested")

    def _emit_frame(
        self,
        data: Mapping[str, Any],
        frame_sink: FrameSink | None,
    ) -> None:
        if frame_sink is not None and data.get("frame") is not None:
            self._frame_pacer.wait()
            frame_sink(data["frame"])

    def _emit_presentation(self, name: str, **fields: Any) -> None:
        if self._presentation_sink is not None:
            self._presentation_sink(name, **fields)

    def _emit_fight_presentation(self, data: Mapping[str, Any]) -> None:
        kwargs: dict[str, Any] = {
            "round_number": int(data["winsP1"]) + int(data["winsP2"]) + 1,
        }
        if self._match_identity is not None:
            kwargs["player1"] = self._match_identity["player1"]
            kwargs["player2"] = self._match_identity["player2"]
        self._emit_presentation("fight", **kwargs)

    def _phase_step(
        self,
        actions: list[Any],
        frame_sink: FrameSink | None,
    ) -> dict[str, Any]:
        self._raise_if_stopping()
        data = self.emu.step(actions)
        self._emit_frame(data, frame_sink)
        return data

    def _run_steps(
        self,
        steps: list[dict[str, Any]],
        frame_sink: FrameSink | None,
        presentation_sink: PresentationSink | None = None,
    ) -> None:
        for step in steps:
            presentations = step.get("presentations", {})
            for frame_number in range(1, step["wait"] + 1):
                self._data = self._phase_step([], frame_sink)
                presentation = presentations.get(frame_number)
                if presentation_sink is not None and presentation is not None:
                    presentation_sink(presentation)
            actions = [action.value for action in step["actions"]]
            self._data = self._phase_step(actions, frame_sink)
            action_presentation = step.get("action_presentation")
            if presentation_sink is not None and action_presentation is not None:
                presentation_sink(action_presentation)

    def _matchup_character_assignments(self) -> tuple[tuple[str, int], ...]:
        character_count = 1 if self._vs_cpu else len(self.config.characters)
        return tuple(
            (
                f"characterP{player}",
                CHARACTER_NAME_TO_LOCAL_ID[character],
            )
            for player, character in enumerate(
                self.config.characters[:character_count], start=1
            )
        )

    def _matchup_lock_command(self) -> str:
        statements = [
            (
                "local p1State = mem:read_u8("
                f"{self.memory_addresses['characterSelectStateP1'].address})"
            ),
            "if p1State >= 2 then configuredMatchupSeenSelect = true end",
            "if not configuredMatchupSeenSelect then return end",
            (
                "local fighting = mem:read_u8("
                f"{self.memory_addresses['fighting'].address})"
            ),
            (
                "if fighting == 0 then configuredMatchupFightFrames = 0 "
                "else configuredMatchupFightFrames = "
                "configuredMatchupFightFrames + 1 end"
            ),
            (
                "if fighting ~= 0 and configuredMatchupFightFrames > "
                f"{self.config.step_ratio * 2} then return end"
            ),
        ]
        for address_name, character in self._matchup_character_assignments():
            player = address_name[-2:]
            state = f"state{player}"
            statements.extend(
                [
                    (
                        f"local {state} = mem:read_u8("
                        f"{self.memory_addresses[f'characterSelectState{player}'].address})"
                    ),
                    (
                        f"mem:write_u8({self.memory_addresses[address_name].address}, "
                        f"{character})"
                    ),
                    (
                        f"if {state} >= 2 then mem:write_u8("
                        f"{self.memory_addresses[f'characterSelectColor{player}'].address}, "
                        f"{max(0, self.config.outfits[int(player[1]) - 1] - 1)}) end"
                    ),
                    (
                        f"if {state} >= 3 then mem:write_u8("
                        f"{self.memory_addresses[f'characterSelectSa{player}'].address}, "
                        f"{max(0, self.config.super_arts[int(player[1]) - 1] - 1)}) end"
                    ),
                ]
            )
        body = "; ".join(statements)
        return (
            "configuredMatchupSeenSelect = false; "
            "configuredMatchupFightFrames = 0; "
            f"function lockConfiguredMatchup() {body} end; "
            'emu.register_frame(lockConfiguredMatchup, "configured-matchup")'
        )

    def _register_matchup_character_lock(self) -> None:
        if self._matchup_character_lock_installed or self._interactive_select:
            return
        command = self._matchup_lock_command()
        # console.writeln() drains MAME's shared stdout queue for 0.5s and raises on
        # any output, which would steal the frame data Emulator.step() reads back.
        self.emu.console.process.stdin.write(command.encode("utf-8") + b"\n")
        self.emu.console.process.stdin.flush()
        self._matchup_character_lock_installed = True

    def _matchup_characters_locked(self, data: Mapping[str, Any]) -> bool:
        return all(
            int(data[address_name]) == character
            for address_name, character in self._matchup_character_assignments()
        )

    def _advance_cpu_menus(
        self,
        state: _CpuMenuAdvanceState,
        *,
        p1_jab: int,
    ) -> list[int]:
        pressed: list[int] = []
        menu_state = int(self._data["menuState"])
        p1_state = int(self._data["characterSelectStateP1"])
        if menu_state == _CPU_CHARACTER_MENU and p1_state in state.pending_p1_states:
            pressed.append(p1_jab)
            state.pending_p1_states.remove(p1_state)
        if menu_state in _CPU_OPPONENT_MENUS:
            state.opponent_menu_frames += 1
        else:
            state.opponent_menu_frames = 0
        cpu_confirm_elapsed = state.opponent_menu_frames - _CPU_OPPONENT_CONFIRM_DELAY
        if (
            cpu_confirm_elapsed >= 0
            and cpu_confirm_elapsed % _CPU_OPPONENT_CONFIRM_INTERVAL == 0
        ):
            pressed.append(p1_jab)
        return pressed

    def _step_select(
        self,
        actions: dict[str, int],
        frame_sink: FrameSink | None,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        selection_action = actions.get("selection", 0)
        if not isinstance(selection_action, int):
            selection_action = 0
        player = 1
        if not self._vs_cpu:
            player1_state = int(self._data["characterSelectStateP1"])
            player2_state = int(self._data["characterSelectStateP2"])
            if player1_state >= 4 and player2_state < 4:
                player = 2
            elif player1_state >= 5:
                player = 2
        pressed = _action_values(player, selection_action)
        self._data = self._phase_step(pressed, frame_sink)
        player1_selected = int(self._data["characterSelectStateP1"]) >= 5
        if self._vs_cpu:
            menu_state = int(self._data["menuState"])
            if menu_state in _CPU_OPPONENT_MENUS:
                self._cpu_opponent_menu_seen = True
            player2_selected = (
                self._cpu_opponent_menu_seen and menu_state not in _CPU_OPPONENT_MENUS
            )
        else:
            player2_selected = int(self._data["characterSelectStateP2"]) >= 5
        if player1_selected and player2_selected and not self._versus_presented:
            self._versus_presented = True
            self._emit_presentation("versus")
        if int(self._data["fighting"]) != 0:
            player1_selected = True
            player2_selected = True
            self._selecting = False
            self._match_identity = self.read_match_identity()
            self.expected_health = {
                "P1": int(self._data["healthP1"]),
                "P2": int(self._data["healthP2"]),
            }
            self.expected_wins = {
                "P1": int(self._data["winsP1"]),
                "P2": int(self._data["winsP2"]),
            }
            self._data = self._sub_step([])
            self._emit_frame(self._data, frame_sink)
            self._emit_fight_presentation(self._data)
        observation = _normalize_local_observation(self._data)
        info = {
            "game_done": False,
            "round_done": False,
            "stage_done": False,
            "selecting": self._selecting,
            "player1_selected": player1_selected,
            "player2_selected": player2_selected,
            "winner": None,
        }
        return observation, 0.0, False, False, info

    def _wait_for_boot_ready(self, frame_sink: FrameSink | None) -> None:
        stable_frames = 0
        for _ in range(240):
            self._data = self._phase_step([], frame_sink)
            if (
                int(self._data["characterSelectStateP1"]) == 0
                and int(self._data["characterSelectStateP2"]) == 0
            ):
                stable_frames += 1
                if stable_frames >= 5:
                    return
            else:
                stable_frames = 0
        # These bytes are not a reliable readiness signal before service inputs.
        # This bounded wait is only a warm-up; later transitions enforce readiness.
        return

    def _wait_for_character_select(self, frame_sink: FrameSink | None) -> None:
        from MAMEToolkit.sf_environment.Actions import Actions

        players = ("P1",) if self._vs_cpu else ("P1", "P2")
        start_actions = [Actions.P1_START.value]
        if not self._vs_cpu:
            start_actions.append(Actions.P2_START.value)
        for _ in range(240):
            self._data = self._phase_step([], frame_sink)
            states = [
                int(self._data[f"characterSelectState{player}"]) for player in players
            ]
            if all(state == 2 for state in states):
                return
            if any(state == 0 for state in states):
                self._data = self._phase_step(start_actions, frame_sink)
        mode = "1-player" if self._vs_cpu else "2-player"
        raise TimeoutError(f"Timed out waiting for {mode} character select")

    def _select_characters(self, frame_sink: FrameSink | None) -> None:
        from MAMEToolkit.sf_environment.Actions import Actions

        players = [
            (
                f"P{index + 1}",
                (Actions.P1_JPUNCH if index == 0 else Actions.P2_JPUNCH).value,
            )
            for index, _ in enumerate(self.config.characters)
            if index == 0 or not self._vs_cpu
        ]
        character_locked = {player: False for player, _ in players}
        sa_locked = character_locked.copy()
        character_visible_frames = {player: 0 for player, _ in players}
        sa_visible_frames = character_visible_frames.copy()
        presentation_steps = max(
            1,
            int(_SELECTION_PRESENTATION_FRAMES / self.config.step_ratio),
        )
        selection_timeout_steps = presentation_steps * 2 + 240

        for _ in range(selection_timeout_steps):
            self._data = self._phase_step([], frame_sink)
            pressed = []
            states = {}
            for player, jab in players:
                state = int(self._data[f"characterSelectState{player}"])
                states[player] = state
                if state == 2 and not character_locked[player]:
                    character_visible_frames[player] += 1
                    if character_visible_frames[player] >= presentation_steps:
                        pressed.append(jab)
                        character_locked[player] = True
                if state == 4 and not sa_locked[player]:
                    sa_visible_frames[player] += 1
                    if sa_visible_frames[player] >= presentation_steps:
                        pressed.append(jab)
                        sa_locked[player] = True

            if pressed:
                self._data = self._phase_step(pressed, frame_sink)
                states = {
                    player: int(self._data[f"characterSelectState{player}"])
                    for player, _ in players
                }
            if all(state == 5 for state in states.values()):
                return

        raise TimeoutError("Timed out locking characters/super arts")

    def _wait_for_fight_start(
        self,
        frame_sink: FrameSink | None,
    ) -> dict[str, Any]:
        interactive = self._interactive_select
        tournament_mode = self._vs_cpu
        p1_char = CHARACTER_NAME_TO_LOCAL_ID[self.config.characters[0]]
        p2_char = CHARACTER_NAME_TO_LOCAL_ID[self.config.characters[1]]
        cpu_menus = None
        p1_jab = 0
        if self._vs_cpu:
            from MAMEToolkit.sf_environment.Actions import Actions

            pending = set() if interactive else {2, 4}
            cpu_menus = _CpuMenuAdvanceState(pending_p1_states=pending)
            p1_jab = Actions.P1_JPUNCH.value
        locked_fighting_frames = 0
        for _ in range(900):
            pressed = (
                self._advance_cpu_menus(
                    cpu_menus,
                    p1_jab=p1_jab,
                )
                if cpu_menus is not None
                else []
            )
            self._data = self._phase_step(pressed, frame_sink)
            if int(self._data["fighting"]) == 0:
                locked_fighting_frames = 0
                continue
            if interactive:
                break
            if self._matchup_characters_locked(self._data):
                locked_fighting_frames += 1
                if locked_fighting_frames >= 2:
                    break
            else:
                locked_fighting_frames = 0
        else:
            raise TimeoutError(
                "Timed out waiting for fight start "
                f"(p1_state={int(self._data['characterSelectStateP1'])}, "
                f"p2_state={int(self._data['characterSelectStateP2'])}, "
                f"menu={int(self._data['menuState'])}, "
                f"fighting={int(self._data['fighting'])})"
            )

        self.expected_health = {
            "P1": int(self._data["healthP1"]),
            "P2": int(self._data["healthP2"]),
        }
        self.expected_wins = {
            "P1": int(self._data["winsP1"]),
            "P2": int(self._data["winsP2"]),
        }
        data = self._sub_step([])
        self._emit_frame(data, frame_sink)
        if not interactive and not self._matchup_characters_locked(data):
            actual_p1 = int(data["characterP1"])
            actual_p2 = int(data["characterP2"])
            expected_p2 = "rotating" if tournament_mode else str(p2_char)
            raise RuntimeError(
                "fight started with unexpected matchup "
                f"(p1={actual_p1}, p2={actual_p2}; "
                f"expected p1={p1_char}, p2={expected_p2})"
            )
        self._match_identity = {
            "player1": self.read_player_identity("P1"),
            "player2": self.read_player_identity("P2"),
        }
        self._emit_fight_presentation(data)
        return data

    def _new_game(
        self,
        frame_sink: FrameSink | None,
        presentation_sink: PresentationSink | None = None,
    ) -> None:
        self._versus_presented = False
        self._wait_for_boot_ready(frame_sink)
        if self._vs_cpu:
            self._run_steps(
                _cpu_difficulty_steps(self.config.step_ratio, self._cpu_difficulty),
                None,
            )
            self._run_steps(
                _boot_steps(self.config.step_ratio, vs_cpu=True),
                frame_sink,
                presentation_sink,
            )
        else:
            self._run_steps(
                _boot_steps(self.config.step_ratio),
                frame_sink,
                presentation_sink,
            )
        if not self._interactive_select:
            self._register_matchup_character_lock()
        self._wait_for_character_select(frame_sink)
        self.expected_health = {"P1": 0, "P2": 0}
        self.expected_wins = {"P1": 0, "P2": 0}
        if self._interactive_select:
            self._selecting = True
            return
        self._select_characters(frame_sink)
        self._data = self._wait_for_fight_start(frame_sink)

    def _sub_step(self, actions: list[Any]) -> dict[str, Any]:
        self._raise_if_stopping()
        data = self.emu.step(actions)

        p1_diff = self.expected_health["P1"] - int(data["healthP1"])
        p2_diff = self.expected_health["P2"] - int(data["healthP2"])
        self.expected_health = {
            "P1": int(data["healthP1"]),
            "P2": int(data["healthP2"]),
        }

        data["stunBarP1"] = int(data["stunBarP1"]) >> 24
        data["stunBarP2"] = int(data["stunBarP2"]) >> 24
        data["reward"] = float(p2_diff - p1_diff)
        return data

    def _run_till_victor(
        self,
        data: dict[str, Any],
        frame_sink: FrameSink | None,
    ) -> dict[str, Any]:
        total_reward = float(data["reward"])
        self._emit_frame(data, frame_sink)
        while self.expected_wins["P1"] == int(data["winsP1"]) and self.expected_wins[
            "P2"
        ] == int(data["winsP2"]):
            data = self._sub_step([])
            self._emit_frame(data, frame_sink)
            total_reward += float(data["reward"])
        self.expected_wins = {
            "P1": int(data["winsP1"]),
            "P2": int(data["winsP2"]),
        }
        data["reward"] = total_reward
        return data

    def _wait_for_post_ko_black_frame(
        self,
        data: dict[str, Any],
        frame_sink: FrameSink | None,
    ) -> dict[str, Any]:
        for _ in range(900):
            if not data["frame"].any():
                return data
            data = self._sub_step([])
            self._emit_frame(data, frame_sink)
        raise TimeoutError("Timed out waiting for the post-KO black frame")

    def _stream_terminal_sequence(
        self,
        data: dict[str, Any],
        frame_sink: FrameSink | None,
        *,
        tournament_won: bool,
        winner: str | None,
    ) -> dict[str, Any]:
        frames_per_second = 60 / self.config.step_ratio
        minimum_frames = round(
            (_TOURNAMENT_WIN_MIN_SECONDS if tournament_won else _TERMINAL_MIN_SECONDS)
            * frames_per_second
        )
        maximum_frames = round(_TERMINAL_MAX_SECONDS * frames_per_second)
        idle_frames_needed = round(_TERMINAL_IDLE_SECONDS * frames_per_second)
        black_frames_needed = max(1, round(0.25 * frames_per_second))
        idle_frames = 0
        black_frames = 0

        if winner == "P1":
            self._emit_presentation("winner")
        elif winner == "P2":
            self._emit_presentation("game_over")
        self._emit_presentation("continue")
        for frame_number in range(maximum_frames):
            data = self._sub_step([])
            self._emit_frame(data, frame_sink)
            if frame_number < minimum_frames:
                continue

            if data["frame"].any():
                black_frames = 0
            else:
                black_frames += 1

            idle = (
                int(data["fighting"]) == 0
                and int(data["characterSelectStateP1"]) == 0
                and int(data["characterSelectStateP2"]) == 0
            )
            idle_frames = idle_frames + 1 if idle else 0
            if black_frames >= black_frames_needed or idle_frames >= idle_frames_needed:
                return data

        return data


def create_environment(config: EnvironmentConfig) -> GameEnvironment:
    return LocalSfiiiAdapter(config)
