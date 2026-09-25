import base64
import os
import random
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any, Literal, Sequence, TypedDict

import modal

SEED = 42
random.seed(SEED)


# modal

ROUTING_REGION = "us-east"
CONTAINER_REGION = "us-east"
MINUTES = 60
GB = 1024


# web app


class ParticipantSpec(TypedDict):
    label: str
    kind: Literal["human", "cpu", "model"]
    seats: tuple[str, ...]


PARTICIPANT_SPECS: dict[str, ParticipantSpec] = {
    "human": {
        "label": "YOU",
        "kind": "human",
        "seats": ("P1",),
    },
    "cpu": {
        "label": "CPU",
        "kind": "cpu",
        "seats": ("P2",),
    },
    "qwen3_vl_8b": {
        "label": "QWEN3-VL-8B",
        "kind": "model",
        "seats": ("P1", "P2"),
    },
    "qwen35_9b": {
        "label": "QWEN3.5-9B",
        "kind": "model",
        "seats": ("P1", "P2"),
    },
    "gemma4_31b": {
        "label": "GEMMA4-31B",
        "kind": "model",
        "seats": ("P1", "P2"),
    },
    "ministral3_14b": {
        "label": "MINISTRAL3-14B",
        "kind": "model",
        "seats": ("P1", "P2"),
    },
}
PARTICIPANT_LABELS = {
    participant: spec["label"] for participant, spec in PARTICIPANT_SPECS.items()
}
DEFAULT_PLAYER1_PARTICIPANT = "human"
DEFAULT_PLAYER2_PARTICIPANT = "qwen35_9b"
MAX_CPU_DIFFICULTY = 8


def is_model_participant(participant: str) -> bool:
    spec = PARTICIPANT_SPECS.get(participant)
    return spec is not None and spec["kind"] == "model"


def normalize_participant(participant: str, *, seat: str) -> str:
    default = (
        DEFAULT_PLAYER1_PARTICIPANT if seat == "P1" else DEFAULT_PLAYER2_PARTICIPANT
    )
    spec = PARTICIPANT_SPECS.get(participant)
    if spec is None or seat not in spec["seats"]:
        return default
    return participant


def normalize_game_participants(game_settings: dict) -> tuple[str, str]:
    player1_participant = normalize_participant(
        game_settings.get("player1Participant", DEFAULT_PLAYER1_PARTICIPANT),
        seat="P1",
    )
    player2_participant = normalize_participant(
        game_settings.get("player2Participant", DEFAULT_PLAYER2_PARTICIPANT),
        seat="P2",
    )
    game_settings["player1Participant"] = player1_participant
    game_settings["player2Participant"] = player2_participant
    return player1_participant, player2_participant


def create_gameplay_image(
    *,
    extra_python_packages: Sequence[str] = (),
    base_image: modal.Image | None = None,
    copy: bool = False,
    add_python_source: bool = False,
) -> modal.Image:
    apt_packages = ["ffmpeg", "libturbojpeg-dev"]
    python_packages = [
        "MAMEToolkit==1.1.0",
        "PyTurboJPEG==1.8.2",
        *extra_python_packages,
    ]
    image = base_image or modal.Image.debian_slim(python_version="3.12")
    image = (
        image
        .apt_install(*apt_packages)
        .env({
            "SDL_VIDEODRIVER": "dummy",
            "SDL_AUDIODRIVER": "dummy",
            "SF3_WARM_MODELS": os.environ.get("SF3_WARM_MODELS", "0"),
            "XDG_RUNTIME_DIR": "/tmp",
        })
        .uv_pip_install(*python_packages)
        .add_local_file(
            Path(__file__).parent.parent / "assets" / "engine" / "sfiii3n.zip",
            "/root/sfiii3n.zip",
            copy=copy,
        )
    )
    if add_python_source:
        image = image.add_local_python_source("src", copy=copy)
    return image


def _exec_subprocess(cmd: list[str]) -> None:
    process = subprocess.Popen(cmd)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


# game constants

STUN_BAR_MAX = 72
SUPER_BAR_MAX = 128
HEALTH_MAX = 160
TIMER_MAX = 100


# characters

CHARACTER_MAPPING = {
    0: "Gouki",
    1: "Alex",
    2: "Chun-Li",
    3: "Dudley",
    4: "Elena",
    5: "Hugo",
    6: "Ibuki",
    7: "Ken",
    8: "Makoto",
    9: "Necro",
    10: "Oro",
    11: "Q",
    12: "Remy",
    13: "Ryu",
    14: "Sean",
    15: "Twelve",
    16: "Urien",
    17: "Yang",
    18: "Yun",
}
CHARACTER_TO_ID = {v: k for k, v in CHARACTER_MAPPING.items()}


# moves

MOVES = {
    "No-Move": 0,
    "Left": 1,
    "Left+Up": 2,
    "Up": 3,
    "Right+Up": 4,
    "Right": 5,
    "Right+Down": 6,
    "Down": 7,
    "Left+Down": 8,
    "Low Punch": 9,
    "Medium Punch": 10,
    "High Punch": 11,
    "Low Kick": 12,
    "Medium Kick": 13,
    "High Kick": 14,
    "Low Punch+Low Kick": 15,
    "Medium Punch+Medium Kick": 16,
    "High Punch+High Kick": 17,
}

INDEX_TO_MOVE = {v: k for k, v in MOVES.items()}


def mirror_moves(moves):
    mirrored = []
    for move in moves:
        if move == MOVES["Left"]:
            mirrored.append(MOVES["Right"])
        elif move == MOVES["Right"]:
            mirrored.append(MOVES["Left"])
        elif move == MOVES["Left+Up"]:
            mirrored.append(MOVES["Right+Up"])
        elif move == MOVES["Right+Up"]:
            mirrored.append(MOVES["Left+Up"])
        elif move == MOVES["Left+Down"]:
            mirrored.append(MOVES["Right+Down"])
        elif move == MOVES["Right+Down"]:
            mirrored.append(MOVES["Left+Down"])
        else:
            mirrored.append(move)
    return mirrored


def create_move_dict(moves_list):
    return {"left": tuple(moves_list), "right": tuple(mirror_moves(moves_list))}


COMBOS = {
    "Alex": {
        "Power Bomb": create_move_dict([
            MOVES["Right"],
            MOVES["Right+Down"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "Spiral DDT": create_move_dict([
            MOVES["Right"],
            MOVES["Right+Down"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Low Kick"],
        ]),
        "Flash Chop": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "Air Knee Smash": create_move_dict([
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Low Kick"],
        ]),
        "Air Stampede": create_move_dict([
            MOVES["Down"],
            MOVES["Down"],
            MOVES["Up"],
            MOVES["Low Kick"],
        ]),
        "Slash Elbow": create_move_dict([
            MOVES["Right"],
            MOVES["Right"],
            MOVES["Left"],
            MOVES["Low Kick"],
        ]),
    },
    "Chun-Li": {
        "Kikoken": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Left+Up"],
            MOVES["Up"],
            MOVES["Medium Punch"],
        ]),
        "Hazanshu": create_move_dict([
            MOVES["Right"],
            MOVES["Right+Down"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Low Kick"],
        ]),
        "Spinning Bird Kick": create_move_dict([
            MOVES["Down"],
            MOVES["Down"],
            MOVES["Up"],
            MOVES["Low Kick"],
        ]),
        "Hyakuretsu Kyaku": create_move_dict([
            MOVES["Low Kick"],
            MOVES["Low Kick"],
            MOVES["Low Kick"],
            MOVES["Low Kick"],
            MOVES["Low Kick"],
        ]),
    },
    "Dudley": {
        "Ducking": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Left+Up"],
            MOVES["Up"],
            MOVES["Low Kick"],
        ]),
        "Jet Upper": create_move_dict([
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Medium Punch"],
        ]),
        "Machine Gun Blow": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Left+Up"],
            MOVES["Up"],
            MOVES["Medium Punch"],
        ]),
        "Cross Counter": create_move_dict([
            MOVES["Right"],
            MOVES["Right+Down"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "Short Swing Blow": create_move_dict([
            MOVES["Right"],
            MOVES["Right+Down"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Low Kick"],
        ]),
    },
    "Elena": {
        "Rhino Horn": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Left+Up"],
            MOVES["Up"],
            MOVES["Medium Kick"],
        ]),
        "Mallet Smash": create_move_dict([
            MOVES["Right"],
            MOVES["Right+Down"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "Spin Scythe": create_move_dict([
            MOVES["Down"],
            MOVES["Right+Down"],
            MOVES["Right"],
            MOVES["Low Kick"],
        ]),
        "Scratch Wheel": create_move_dict([
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Medium Kick"],
        ]),
        "Lynx Tail": create_move_dict([
            MOVES["Right"],
            MOVES["Down"],
            MOVES["Right+Down"],
            MOVES["Low Kick"],
        ]),
    },
    "Gouki": {
        "Hadouken": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "Shakunetsu-Hadouken": create_move_dict([
            MOVES["Right"],
            MOVES["Right+Down"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "Go Shoryuken": create_move_dict([
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["High Punch"],
        ]),
        "Tatsumaki Zankuukyaku": create_move_dict([
            MOVES["Down"],
            MOVES["Right+Down"],
            MOVES["Right"],
            MOVES["Low Kick"],
        ]),
        "Ashura Senku": create_move_dict([
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Low Punch+Low Kick"],
        ]),
        "Hyakkishu": create_move_dict([
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Low Kick"],
        ]),
    },
    "Hugo": {
        "Shootdown Backbreaker": create_move_dict([
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Low Kick"],
        ]),
        "Ultra Throw": create_move_dict([
            MOVES["Right"],
            MOVES["Right+Down"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Low Kick"],
        ]),
        "Moonsault Press": create_move_dict([
            MOVES["Right"],
            MOVES["Right+Down"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Left+Up"],
            MOVES["Up"],
            MOVES["Right+Up"],
            MOVES["Right"],
            MOVES["Medium Punch"],
        ]),
        "Meat Squasher": create_move_dict([
            MOVES["Right"],
            MOVES["Right+Down"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Left+Up"],
            MOVES["Up"],
            MOVES["Right+Up"],
            MOVES["Right"],
            MOVES["Low Kick"],
        ]),
        "Giant Palm Bomber": create_move_dict([
            MOVES["Down"],
            MOVES["Right+Down"],
            MOVES["Right"],
            MOVES["Medium Punch"],
        ]),
        "Monster Lariat": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Low Kick"],
        ]),
    },
    "Ibuki": {
        "Raida": create_move_dict([
            MOVES["Right"],
            MOVES["Right+Down"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "Kasumi Gake": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Low Kick"],
        ]),
        "Tsuji Goe": create_move_dict([
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Medium Punch"],
        ]),
        "Kunai": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "Kubi Ori": create_move_dict([
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "Kazekiri": create_move_dict([
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Low Kick"],
        ]),
        "Hien": create_move_dict([
            MOVES["Right"],
            MOVES["Down"],
            MOVES["Right+Down"],
            MOVES["Low Kick"],
        ]),
        "Tsumuji": create_move_dict([
            MOVES["Down"],
            MOVES["Right+Down"],
            MOVES["Right"],
            MOVES["Low Kick"],
        ]),
    },
    "Ken": {
        "Hadouken": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "Shoryuken": create_move_dict([
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["High Punch"],
        ]),
        "Tatsumaki": create_move_dict([
            MOVES["Down"],
            MOVES["Right+Down"],
            MOVES["Right"],
            MOVES["Low Kick"],
        ]),
    },
    "Makoto": {
        "Karakusa": create_move_dict([
            MOVES["Right"],
            MOVES["Right+Down"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Low Kick"],
        ]),
        "Hayate": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "Fukiage": create_move_dict([
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Medium Punch"],
        ]),
        "Oroshi": create_move_dict([
            MOVES["Down"],
            MOVES["Right+Down"],
            MOVES["Right"],
            MOVES["Medium Punch"],
        ]),
    },
    "Necro": {
        "Snake Fang": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Left+Up"],
            MOVES["Up"],
            MOVES["Low Kick"],
        ]),
        "Denji Blast": create_move_dict([
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Medium Punch"],
        ]),
        "Flying Viper": create_move_dict([
            MOVES["Down"],
            MOVES["Right+Down"],
            MOVES["Right"],
            MOVES["Medium Punch"],
        ]),
        "Rising Cobra": create_move_dict([
            MOVES["Down"],
            MOVES["Right+Down"],
            MOVES["Right"],
            MOVES["Low Kick"],
        ]),
        "Tornado Hook": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Left+Up"],
            MOVES["Up"],
            MOVES["Medium Punch"],
        ]),
    },
    "Oro": {
        "Niou Riki": create_move_dict([
            MOVES["Right"],
            MOVES["Right+Down"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "Nichirin Shou": create_move_dict([
            MOVES["Right"],
            MOVES["Right"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "Oni Yanma": create_move_dict([
            MOVES["Down"],
            MOVES["Down"],
            MOVES["Up"],
            MOVES["Medium Punch"],
        ]),
        "Jinchuu Watari": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Low Kick"],
        ]),
    },
    "Q": {
        "Capture & Deadly Blow": create_move_dict([
            MOVES["Right"],
            MOVES["Right+Down"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Low Kick"],
        ]),
        "Dashing Straight": create_move_dict([
            MOVES["Right"],
            MOVES["Right"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "Dashing Head Attack": create_move_dict([
            MOVES["Right"],
            MOVES["Right"],
            MOVES["Left"],
            MOVES["High Punch"],
        ]),
        "Dashing Leg Attack": create_move_dict([
            MOVES["Right"],
            MOVES["Right"],
            MOVES["Left"],
            MOVES["Low Kick"],
        ]),
        "High Speed Barrage": create_move_dict([
            MOVES["Down"],
            MOVES["Right+Down"],
            MOVES["Right"],
            MOVES["Medium Punch"],
        ]),
    },
    "Remy": {
        "Light of Virtue": create_move_dict([
            MOVES["Right"],
            MOVES["Right"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "Light of Virtue (low)": create_move_dict([
            MOVES["Right"],
            MOVES["Right"],
            MOVES["Left"],
            MOVES["Low Kick"],
        ]),
        "Rising Rage Flash": create_move_dict([
            MOVES["Down"],
            MOVES["Down"],
            MOVES["Up"],
            MOVES["Low Kick"],
        ]),
        "Cold Blue Kick": create_move_dict([
            MOVES["Down"],
            MOVES["Right+Down"],
            MOVES["Right"],
            MOVES["Low Kick"],
        ]),
    },
    "Ryu": {
        "Hadouken": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "Shoryuken": create_move_dict([
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["High Punch"],
        ]),
        "Tatsumaki Senpukyaku": create_move_dict([
            MOVES["Down"],
            MOVES["Right+Down"],
            MOVES["Right"],
            MOVES["Low Kick"],
        ]),
        "Air Tatsumaki Senpukyaku": create_move_dict([
            MOVES["Down"],
            MOVES["Right+Down"],
            MOVES["Right"],
            MOVES["Low Kick"],
        ]),
        "Joudan Sokutou Geri": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Left+Up"],
            MOVES["Up"],
            MOVES["Low Kick"],
        ]),
    },
    "Sean": {
        "Zenten": create_move_dict([
            MOVES["Down"],
            MOVES["Right+Down"],
            MOVES["Right"],
            MOVES["Medium Punch"],
        ]),
        "Sean Tackle": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Left+Up"],
            MOVES["Up"],
            MOVES["Medium Punch"],
        ]),
        "Dragon Smash": create_move_dict([
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Medium Punch"],
        ]),
        "Tornado": create_move_dict([
            MOVES["Down"],
            MOVES["Right+Down"],
            MOVES["Right"],
            MOVES["Low Kick"],
        ]),
        "Ryuubi Kyaku": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Low Kick"],
        ]),
    },
    "Twelve": {
        "Kokuu": create_move_dict([
            MOVES["Left"],
            MOVES["Left"],
        ]),
        "N.D.L.": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "A.X.E.": create_move_dict([
            MOVES["Down"],
            MOVES["Right+Down"],
            MOVES["Right"],
            MOVES["Medium Punch"],
        ]),
        "D.R.A.": create_move_dict([
            MOVES["Down"],
            MOVES["Right+Down"],
            MOVES["Right"],
            MOVES["Low Kick"],
        ]),
    },
    "Urien": {
        "Metallic Sphere": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "Chariot Tackle": create_move_dict([
            MOVES["Right"],
            MOVES["Right"],
            MOVES["Left"],
            MOVES["Low Kick"],
        ]),
        "Violence Knee Drop": create_move_dict([
            MOVES["Down"],
            MOVES["Down"],
            MOVES["Up"],
            MOVES["Low Kick"],
        ]),
        "Dangerous Headbutt": create_move_dict([
            MOVES["Down"],
            MOVES["Down"],
            MOVES["Up"],
            MOVES["Medium Punch"],
        ]),
    },
    "Yang": {
        "Tourou Zan": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "Senkyuutai": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Low Kick"],
        ]),
        "Byakko Soushouda": create_move_dict([
            MOVES["Down"],
            MOVES["Right+Down"],
            MOVES["Right"],
            MOVES["Medium Punch"],
        ]),
        "Fake Byakko Soushouda": create_move_dict([
            MOVES["Down"],
            MOVES["Right+Down"],
            MOVES["Right"],
            MOVES["Low Punch+Low Kick"],
        ]),
        "Zenpou Tenshin": create_move_dict([
            MOVES["Right"],
            MOVES["Right+Down"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Low Kick"],
        ]),
        "Kaihou": create_move_dict([
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Low Kick"],
        ]),
    },
    "Yun": {
        "Zenpou Tenshin": create_move_dict([
            MOVES["Right"],
            MOVES["Right+Down"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Low Kick"],
        ]),
        "Kobokushi": create_move_dict([
            MOVES["Down"],
            MOVES["Right+Down"],
            MOVES["Right"],
            MOVES["Medium Punch"],
        ]),
        "Fake Kobokushi": create_move_dict([
            MOVES["Down"],
            MOVES["Right+Down"],
            MOVES["Right"],
            MOVES["Low Punch+Low Kick"],
        ]),
        "Zesshou Hohou": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "Tetsuzanko": create_move_dict([
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Medium Punch"],
        ]),
        "Nishoukyaku": create_move_dict([
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Low Kick"],
        ]),
    },
}

SPECIAL_MOVES = {
    "Alex": {
        "1 Hyper Bomb": create_move_dict([
            MOVES["Right"],
            MOVES["Right+Down"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Left+Up"],
            MOVES["Up"],
            MOVES["Right+Up"],
            MOVES["Right"],
            MOVES["Medium Punch"],
        ]),
        "2 Boomerang Raid": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Low Punch"],
        ]),
        "3 Stun Gun Headbutt": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
    },
    "Chun-Li": {
        "1 Kikou Shou": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "2 Houyoku Sen": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Kick"],
        ]),
        "3 Tensei Ranka": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Low Kick"],
        ]),
    },
    "Dudley": {
        "1 Rocket Upper": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "2 Rolling Thunder": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "3 Corkscrew Blow": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["High Punch"],
        ]),
    },
    "Elena": {
        "1 Spinning Beat": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Low Kick"],
        ]),
        "2 Brave Dance": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Low Kick"],
        ]),
        "3 Healing": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
    },
    "Gouki": {
        "1 Messatsu Gou Hadou": create_move_dict([
            MOVES["Down"],
            MOVES["Right+Down"],
            MOVES["Right"],
            MOVES["Down"],
            MOVES["Right+Down"],
            MOVES["Right"],
            MOVES["Medium Punch"],
        ]),
        "2 Messatsu Gou Shoryu": create_move_dict([
            MOVES["Down"],
            MOVES["Right+Down"],
            MOVES["Right"],
            MOVES["Down"],
            MOVES["Right+Down"],
            MOVES["Right"],
            MOVES["Medium Punch"],
        ]),
        "3 Messatsu-Gourasen": create_move_dict([
            MOVES["Down"],
            MOVES["Right+Down"],
            MOVES["Right"],
            MOVES["Down"],
            MOVES["Right+Down"],
            MOVES["Right"],
            MOVES["Low Kick"],
        ]),
        "Max Shungokusatsu (2 bars)": create_move_dict([
            MOVES["Low Punch"],
            MOVES["Low Punch"],
            MOVES["Right"],
            MOVES["Low Kick"],
            MOVES["High Punch"],
        ]),
        "Max Kongou Kokuretsuzan (2 bars)": create_move_dict([
            MOVES["Down"],
            MOVES["Down"],
            MOVES["Down"],
            MOVES["Low Punch+Low Kick"],
        ]),
    },
    "Hugo": {
        "1 Gigas Breaker": create_move_dict([
            MOVES["Right"],
            MOVES["Right+Down"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Left+Up"],
            MOVES["Up"],
            MOVES["Right+Up"],
            MOVES["Right"],
            MOVES["Right+Down"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Left+Up"],
            MOVES["Up"],
            MOVES["Right+Up"],
            MOVES["Right"],
            MOVES["Medium Punch"],
        ]),
        "2 Megaton Press": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Low Kick"],
        ]),
        "3 Hammer Frenzy": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
    },
    "Ibuki": {
        "1 Kasumi Suzaku": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "2 Yoroi Dooshi": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "3 Yami Shigure": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
    },
    "Ken": {
        "1 Shoryureppa": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "2 Shinryuken": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Low Kick"],
        ]),
        "3 Shippu Jinraikyaku": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Low Kick"],
        ]),
    },
    "Makoto": {
        "1 Seichusen Godanzuki": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Low Punch"],
        ]),
        "2 Abare Tosanami": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Low Kick"],
        ]),
        "3 Tanden Renki": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
    },
    "Necro": {
        "1 Magnetic Storm": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "2 Slam Dance": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "3 Electric Snake": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
    },
    "Oro": {
        "1 Kishin Riki": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "Max-1 EX Kishin Riki": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Low Punch+Low Kick"],
        ]),
        "2 Yagyou Dama": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "Max-2 EX Yagyou Dama (3 bars)": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Low Punch+Low Kick"],
        ]),
        "3 Tengu Stone": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "Max-3 EX Tengu Stone": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Low Punch+Low Kick"],
        ]),
    },
    "Q": {
        "1 Critical Combo Attack": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "2 Deadly Double Combination": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "3 Total Destruction": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
    },
    "Remy": {
        "1 Light of Justice": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "2 Supreme Rising Rage Flash": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Low Kick"],
        ]),
        "3 Blue Nocturne": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Low Kick"],
        ]),
    },
    "Ryu": {
        "1 Shinkuu-Hadouken": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "2 Shin Shoryuken": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "3 Denjin Hadouken": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
    },
    "Sean": {
        "1 Hadou Burst": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "2 Shoryuu Cannon": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "3 Hyper Tornado": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
    },
    "Twelve": {
        "1 X.N.D.L": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "2 X.F.L.A.T": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Low Kick"],
        ]),
        "3 X.C.O.P.Y": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
    },
    "Urien": {
        "1 Tyrant Slaughter": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "2 Temporal Thunder": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "3 Aegis Reflector": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
    },
    "Yang": {
        "1 Raishin Mahha Ken": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "2 Tenshin Senkyutai": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Low Kick"],
        ]),
        "3 Sei'ei Enbu": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
    },
    "Yun": {
        "1 You-hou": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "2 Sourai Rengeki": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
        "3 Genei-jin": create_move_dict([
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Down"],
            MOVES["Left+Down"],
            MOVES["Left"],
            MOVES["Medium Punch"],
        ]),
    },
}


# instructions


CLOSE_IN_MOVES = {
    "Move Closer": create_move_dict([MOVES["Left"]] * 4),
    "Jump Closer": create_move_dict([MOVES["Left+Up"]] * 4),
}

BASE_META_INSTRUCTIONS = {
    **CLOSE_IN_MOVES,
    **{
        move_name: create_move_dict([move_nb])
        for move_name, move_nb in MOVES.items()
        if "Punch" in move_name or "Kick" in move_name
    },
}

RECENT_MOVE_LIMIT = 8


def get_available_instructions_for_character(
    character: str,
    super_art: int,
    super_count: int,
) -> list[str]:
    instructions = []
    instructions.extend(BASE_META_INSTRUCTIONS.keys())
    instructions.extend(COMBOS[character].keys())

    if super_count > 0:
        for special_move in SPECIAL_MOVES[character].keys():
            if special_move.startswith(f"{super_art}"):
                instructions.append(str(special_move))

            bars_required = 1
            for i in range(1, 4):
                if f"({i} bars)" in special_move:
                    bars_required = i
                    break

            if super_count >= bars_required:
                # "Max-{number}" (e.g., Oro's moves)
                if special_move.startswith(f"Max-{super_art}"):
                    instructions.append(special_move)
                # Generic "Max" moves (e.g., Gouki's moves)
                elif special_move.startswith("Max "):
                    instructions.append(special_move)

    return instructions


def move_regex(available_moves: list[str]) -> str:
    alternatives = sorted(
        (re.escape(move).replace(r"\ ", " ") for move in available_moves),
        key=len,
        reverse=True,
    )
    return "(?:" + "|".join(alternatives) + ")"


@dataclass
class PlayerState:
    character: str
    super_art: int
    wins: int
    side: int
    stunned: bool
    stun_bar: int
    health: int
    super_count: int
    super_bar: int


class FrameEncoder:
    """Caches the JPEG data URL of the most recent frame. Needs the gameplay image."""

    def __init__(self) -> None:
        import numpy
        from turbojpeg import TJPF_RGB, TJSAMP_420, TurboJPEG

        self._numpy = numpy
        self._pixel_format = TJPF_RGB
        self._subsample = TJSAMP_420
        self._encoder = TurboJPEG()
        self._frame = None
        self._data_url = ""

    def data_url(self, frame: Any) -> str:
        if self._frame is not frame:
            jpeg = self._encoder.encode(
                self._numpy.ascontiguousarray(frame),
                quality=85,
                pixel_format=self._pixel_format,
                jpeg_subsample=self._subsample,
            )
            self._frame = frame
            self._data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode(
                "utf-8"
            )
        return self._data_url


def player_state(
    observation: dict[str, Any],
    settings: dict[str, Any],
    player: str,
) -> PlayerState:
    raw = observation[player]
    return PlayerState(
        character=settings["character"],
        super_art=settings["superArt"],
        wins=int(raw["wins"][0]),
        side=int(raw["side"]),
        stunned=bool(raw["stunned"]),
        stun_bar=int(raw["stun_bar"][0]),
        health=int(raw["health"][0]),
        super_count=int(raw["super_count"][0]),
        super_bar=int(raw["super_bar"][0]),
    )


NEXT_MOVE_PROMPT = "Your next move is:"


def create_messages(
    player1: PlayerState,
    player2: PlayerState,
    frames: list[str],
    recent_moves=None,
) -> tuple[list[dict], list[str]]:
    available_moves = get_available_instructions_for_character(
        player2.character, player2.super_art, player2.super_count
    )
    if recent_moves is not None:
        available_moves = [m for m in available_moves if m not in recent_moves]
        if not available_moves:
            available_moves = list(CLOSE_IN_MOVES.keys())
    moves_prompt = "You may only use the following moves:\n"
    moves_prompt += chr(10).join("- " + move for move in available_moves)

    return [
        {
            "role": "system",
            "content": dedent(
                f"""
                You are playing Street Fighter III 3rd strike.
                It's best of 3: you've won {
                    player2.wins
                } rounds, and your opponent has won {player1.wins} rounds.
                Your character is {
                    player2.character
                }, and your opponent's character is {player1.character}.

                You are on the {
                    "left side of the screen, facing right"
                    if player2.side
                    else "right side of the screen, facing left"
                }
                {moves_prompt}

                Only respond with the printed name of your next move.
                """
            ),
        },
        {
            "role": "user",
            "content": [
                *[
                    {
                        "type": "image",
                        "image": frame,
                    }
                    for frame in frames
                ],
                {
                    "type": "text",
                    "text": NEXT_MOVE_PROMPT,
                },
            ],
        },
    ], available_moves


async def generate_move(
    chat: Any,
    controlled: PlayerState,
    opponent: PlayerState,
    frame_url: str,
    recent_moves: Sequence[str] | None = None,
) -> tuple[list[int], str]:
    messages, available_moves = create_messages(
        opponent,
        controlled,
        [frame_url],
        recent_moves,
    )
    return await chat(
        messages,
        controlled.character,
        controlled.super_art,
        controlled.super_count,
        controlled.side,
        available_moves,
    )


def est_super_ct(super_bar: int) -> int:
    if super_bar == SUPER_BAR_MAX:
        return 3
    elif super_bar >= (SUPER_BAR_MAX // 3) * 2:
        return 2
    elif super_bar >= SUPER_BAR_MAX // 3:
        return 1
    else:
        return 0


def create_random_messages() -> tuple[list[dict], str, int, int, int, list[str]]:
    player1_character = random.choice(list(CHARACTER_MAPPING.values()))
    player2_character = random.choice(list(CHARACTER_MAPPING.values()))

    player1_super_art = random.randint(1, 3)
    player2_super_art = random.randint(1, 3)

    side = random.randint(0, 1)

    player1_stun_bar = random.randint(0, STUN_BAR_MAX)
    player2_stun_bar = random.randint(0, STUN_BAR_MAX)

    player1_super_bar = random.randint(0, SUPER_BAR_MAX)
    player2_super_bar = random.randint(0, SUPER_BAR_MAX)

    player1_super_count = est_super_ct(player1_super_bar)
    player2_super_count = est_super_ct(player2_super_bar)

    player1 = PlayerState(
        character=player1_character,
        super_art=player1_super_art,
        wins=random.randint(0, 2),
        side=side,
        stunned=player1_stun_bar == STUN_BAR_MAX,
        stun_bar=player1_stun_bar,
        health=random.randint(0, HEALTH_MAX),
        super_count=player1_super_count,
        super_bar=player1_super_bar,
    )
    player2 = PlayerState(
        character=player2_character,
        super_art=player2_super_art,
        wins=random.randint(0, 2),
        side=1 - side,
        stunned=player2_stun_bar == STUN_BAR_MAX,
        stun_bar=player2_stun_bar,
        health=random.randint(0, HEALTH_MAX),
        super_count=player2_super_count,
        super_bar=player2_super_bar,
    )

    warmup_frame_data_url = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAYAAAADgCAIAAACFJiYGAAABEUlEQVR42u3BMQEAAADCoPVP7WsIoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB4A/ENAAEh48SMAAAAAElFTkSuQmCC"
    )
    messages, available_moves = create_messages(
        player1,
        player2,
        frames=[warmup_frame_data_url],
    )

    return (
        messages,
        player2_character,
        player2_super_art,
        player2_super_count,
        side,
        available_moves,
    )


MAX_CONTEXT_LEN = 2048
MAX_TOKENS = 14  # "Max-2 EX Yagyou Dama (3 bars)" = 13 toks + 1 for Gemma JSON quotes


# llm post-processing


def parse_move(character: str, move_name: str, side: int) -> tuple[int, ...] | None:
    current_direction = "left" if side == 0 else "right"

    if move_name in BASE_META_INSTRUCTIONS:
        return BASE_META_INSTRUCTIONS[move_name][current_direction]
    elif move_name in COMBOS[character]:
        return COMBOS[character][move_name][current_direction]
    elif move_name in SPECIAL_MOVES[character]:
        return SPECIAL_MOVES[character][move_name][current_direction]
    return None


def resolve_move_with_fallback(
    character: str,
    move_name: str,
    side: int,
) -> tuple[list[int], str]:
    move_sequence = parse_move(character, move_name, side)
    if move_sequence is not None:
        return list(move_sequence), move_name
    return [0], "No-Move"
