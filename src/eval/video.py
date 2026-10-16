import re
import subprocess
from pathlib import Path


class MatchVideos:
    def __init__(self, directory: Path, players: tuple[str, str], prefix: str):
        self.directory = directory
        self.players = players
        self.prefix = prefix
        self.path: Path | None = None
        self.process: subprocess.Popen | None = None
        self.fight_index = 0

    def start(self, characters: list[str]) -> None:
        labels = [
            re.sub(r"[^A-Za-z0-9_-]", "_", f"{Path(player).name[:48]}-{character}")
            for player, character in zip(self.players, characters, strict=True)
        ]
        self.path = self.directory / (
            f"{self.prefix}-{self.fight_index:02d}-{'__'.join(labels)}.mp4"
        )

    def write(self, frame) -> None:
        if self.path is None:
            raise RuntimeError("Video has no active fight")
        if self.process is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            height, width = frame.shape[:2]
            self.process = subprocess.Popen(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    f"{width}x{height}",
                    "-framerate",
                    "60",
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(self.path),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        assert self.process.stdin is not None
        self.process.stdin.write(frame.tobytes())

    def finish(self) -> Path:
        if self.path is None:
            raise RuntimeError("Video has no active fight")
        path = self.path
        self.close()
        self.path = None
        self.fight_index += 1
        return path

    def step(self, env, frames):
        total_reward = 0.0
        for actions in frames:
            observation, reward, terminated, truncated, info = env.step(
                actions, self.write
            )
            total_reward += reward
            if not info.get("round_done"):
                self.write(observation["frame"])
            if info.get("round_done") or terminated or truncated:
                break
        return observation, total_reward, terminated, truncated, info

    def close(self) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        assert process.stdin is not None
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass
        try:
            code = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise
        assert process.stderr is not None
        error = process.stderr.read().decode("utf-8", errors="replace")
        process.stderr.close()
        if code != 0:
            raise RuntimeError(f"Video encoding failed: {error}")
