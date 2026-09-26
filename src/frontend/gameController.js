import { byId, setText } from "./utils.js";
import { GameState } from "./gameState.js";
import { ScreenManager } from "./screenManager.js";
import { WebRtcManager } from "./webRtcManager.js";
import { AudioManager } from "./audioManager.js";
import { GamepadManager } from "./gamepadManager.js";
import {
  getHumanSeat,
  hasHumanParticipant,
  isCpuParticipant,
} from "./participantOptions.js";
import { SOUND_KEYS } from "./constants.js";
import { gameplaySoundKey } from "./assetLoader.js";
import { setCanvasSize } from "./app.js";
import { GamepadUINavigator } from "./gamepadUINavigator.js";

const audioScenes = {
  lobby: {
    soundName: SOUND_KEYS.MAIN_MENU,
    volume: 0.2,
    loop: true,
  },
  opening: {
    soundName: SOUND_KEYS.START,
    volume: 0.5,
    loop: false,
  },
  capcom: {
    soundName: SOUND_KEYS.CAPCOM,
    volume: 1,
    loop: false,
  },
  select: {
    soundName: SOUND_KEYS.SELECT,
    volume: 0.2,
    loop: true,
  },
  transition: {
    soundName: SOUND_KEYS.TRANSITION,
    volume: 0.2,
    loop: false,
  },
  win: {
    soundName: SOUND_KEYS.WIN,
    volume: 0.4,
    loop: false,
    holdMs: 7000,
  },
  gameOver: {
    soundName: SOUND_KEYS.GAME_OVER,
    volume: 0.4,
    loop: false,
    holdMs: 6000,
  },
  continue: {
    soundName: SOUND_KEYS.CONTINUE,
    volume: 0.4,
    loop: true,
  },
  judgement: {
    soundName: SOUND_KEYS.JUDGEMENT,
    volume: 0.4,
    loop: false,
  },
  gillIntro: {
    soundName: SOUND_KEYS.GILL_INTRO,
    volume: 0.4,
    loop: false,
  },
};

const createGameController = () => {
  const remoteVideo = document.createElement("video");
  remoteVideo.autoplay = true;
  remoteVideo.muted = true;
  remoteVideo.defaultMuted = true;
  remoteVideo.playsInline = true;
  remoteVideo.setAttribute("playsinline", "");
  remoteVideo.setAttribute("webkit-playsinline", "");
  remoteVideo.setAttribute("aria-hidden", "true");
  // Chromium may pause or fail to decode MediaStream video that is not in the
  // document / viewport, which leaves drawImage() painting a black canvas.
  remoteVideo.style.cssText =
    "position:fixed;top:0;left:0;width:1px;height:1px;opacity:0;pointer-events:none;border:0;margin:0;padding:0;";
  const presentationIntervalMs = 1000 / 60;
  let renderFrameHandle = null;
  let videoFrameHandle = null;
  let pendingVideoFrame = false;
  let lastPresentationAt = null;
  let endFlow = "idle";
  let currentAudioOwner = null;
  let pendingFinishedState = null;
  let queuedContinue = false;
  let queuedGameOver = false;
  let phaseHoldTimer = null;
  const overlayStings = new Set(["win", "gameOver", "judgement"]);

  const clearPhaseHold = () => {
    if (phaseHoldTimer !== null) {
      clearTimeout(phaseHoldTimer);
      phaseHoldTimer = null;
    }
  };

  const playQueuedEndMusic = () => {
    if (queuedGameOver) {
      queuedGameOver = false;
      playAudioScene("gameOver");
      return;
    }
    if (queuedContinue) {
      queuedContinue = false;
      playAudioScene("continue");
      revealReplayIfReady();
      return;
    }
    revealReplayIfReady();
  };

  const releasePhaseOwner = (owner, extra) => {
    if (currentAudioOwner !== owner) return;
    clearPhaseHold();
    currentAudioOwner = null;
    extra?.();
    playQueuedEndMusic();
  };

  const playAudioScene = (scene, options = {}) => {
    const spec = { ...audioScenes[scene], ...options };
    if (!spec.soundName) return false;
    if (
      currentAudioOwner?.scene === scene &&
      currentAudioOwner.soundName === spec.soundName
    ) {
      return true;
    }

    clearPhaseHold();
    const owner = { scene, soundName: spec.soundName };
    currentAudioOwner = owner;
    const started = AudioManager.playPhase(spec.soundName, {
      volume: spec.volume,
      loop: spec.loop,
      onEnd: () => {
        releasePhaseOwner(owner, spec.onEnd);
      },
      onError: (error) => {
        releasePhaseOwner(owner, () => spec.onError?.(error));
      },
    });
    if (!started && currentAudioOwner === owner) {
      currentAudioOwner = null;
      playQueuedEndMusic();
      return started;
    }
    if (started && spec.holdMs) {
      phaseHoldTimer = setTimeout(() => {
        phaseHoldTimer = null;
        releasePhaseOwner(owner);
      }, spec.holdMs);
    }
    return started;
  };

  const stopAllAudio = () => {
    clearPhaseHold();
    currentAudioOwner = null;
    queuedContinue = false;
    queuedGameOver = false;
    AudioManager.stopAll();
  };

  const playGameplayMusic = (event = {}) => {
    queuedContinue = false;
    queuedGameOver = false;
    const state = GameState.get();
    const player1 = event.player1 ?? state.player1;
    const player2 = event.player2 ?? state.player2;
    // CPU story stages use the opponent theme; other modes keep the human pick.
    const character = isCpuParticipant(state.player2Participant)
      ? player2.character
      : getHumanSeat(state) === "P2"
        ? player2.character
        : player1.character;
    const soundName = gameplaySoundKey(character, event.round_number ?? 1);
    if (!soundName) return;

    playAudioScene("gameplay", {
      soundName,
      volume: 0.2,
      loop: true,
    });
  };

  const playLobbyMusic = () => {
    queuedContinue = false;
    queuedGameOver = false;
    playAudioScene("lobby");
  };

  const playSelectMusic = () => {
    queuedContinue = false;
    queuedGameOver = false;
    playAudioScene("select");
  };

  const revealReplayIfReady = () => {
    if (!pendingFinishedState) return;
    if (overlayStings.has(currentAudioOwner?.scene)) return;
    if (queuedContinue) return;
    if (queuedGameOver && currentAudioOwner?.scene !== "continue") {
      queuedGameOver = false;
      playAudioScene("gameOver");
      return;
    }
    const finishedState = pendingFinishedState;
    pendingFinishedState = null;
    handleGameFinished(finishedState);
  };

  const setCanvasLoading = (visible, message = "STARTING GAME...") => {
    const overlay = byId("canvas-loading-overlay");
    if (overlay) {
      overlay.classList.toggle("hidden", !visible);
      overlay.classList.toggle("flex", visible);
    }
    if (visible) setText("canvas-loading-status", message);
  };

  const setReplayVisible = (visible) => {
    const controls = byId("play-again-controls");
    if (controls) {
      controls.classList.toggle("hidden", !visible);
      controls.classList.toggle("flex", visible);
    }
    const gameScreen = byId("game-screen");
    if (gameScreen) {
      gameScreen.classList.toggle("play-again-visible", visible);
    }
    if (visible) {
      requestAnimationFrame(() => {
        byId("play-again-btn")?.focus();
        GamepadUINavigator.updateGamepadSections(true);
      });
    }
  };

  const canStartGame = (state) =>
    state.currentScreen === ScreenManager.screens.LOBBY &&
    state.serverReady &&
    state.gamePhase === "pregame";

  const startGame = () => {
    const state = GameState.get();
    if (!canStartGame(state)) return false;

    byId("help-overlay")?.classList.add("hidden");
    endFlow = "idle";
    pendingFinishedState = null;
    clearPhaseHold();
    currentAudioOwner = null;
    queuedContinue = false;
    queuedGameOver = false;
    AudioManager.stopTrack("phase");

    const gameConfig = {
      player1Participant: state.player1Participant,
      player2Participant: state.player2Participant,
    };
    GameState.update({
      serverReady: false,
      acceptsInput: false,
      keyState: {},
      currentCharacter: null,
      player1: { character: null, outfit: 1, superArt: 1 },
      player2: { character: null, outfit: 1, superArt: 1 },
    });

    setReplayVisible(false);
    ScreenManager.showScreen(ScreenManager.screens.GAME);
    if (document.activeElement instanceof HTMLElement) {
      document.activeElement.blur();
    }
    setCanvasLoading(false);
    GamepadManager.setUIActive(false);
    WebRtcManager.send("start_game", gameConfig);
    return true;
  };

  const cancelVideoFrameObserver = () => {
    if (
      videoFrameHandle !== null &&
      typeof remoteVideo.cancelVideoFrameCallback === "function"
    ) {
      remoteVideo.cancelVideoFrameCallback(videoFrameHandle);
      videoFrameHandle = null;
    }
  };

  const scheduleVideoPresentation = () => {
    if (renderFrameHandle !== null) return;
    const delay =
      lastPresentationAt === null
        ? 0
        : Math.max(
            0,
            lastPresentationAt + presentationIntervalMs - performance.now()
          );
    renderFrameHandle = setTimeout(() => {
      renderFrameHandle = null;
      renderVideoFrame();
    }, delay);
  };

  const scheduleVideoFrameObserver = () => {
    if (typeof remoteVideo.requestVideoFrameCallback !== "function") {
      return;
    }

    videoFrameHandle = remoteVideo.requestVideoFrameCallback(() => {
      pendingVideoFrame = true;
      handleFrameData();
      scheduleVideoPresentation();
      scheduleVideoFrameObserver();
    });
  };

  const handlePresentation = (data) => {
    if (data.name === "versus") {
      playAudioScene("transition");
      return;
    }
    if (data.name === "fight") {
      playGameplayMusic(data);
      return;
    }
    if (data.name === "capcom") {
      if (GameState.get().gamePhase !== "starting") return;
      playAudioScene("capcom", {
        onEnd: () => {
          if (GameState.get().gamePhase === "starting") {
            playAudioScene("opening");
          }
        },
      });
      return;
    }
    if (data.name === "coin") {
      AudioManager.play(SOUND_KEYS.COIN, { volume: 1, trackAs: "effect" });
      return;
    }
    if (data.name === "judgement") {
      playAudioScene("judgement");
      return;
    }
    if (data.name === "winner") {
      playAudioScene("win");
      return;
    }
    if (data.name === "continue") {
      if (overlayStings.has(currentAudioOwner?.scene)) {
        queuedContinue = true;
        return;
      }
      playAudioScene("continue");
      return;
    }
    if (data.name === "game_over") {
      if (overlayStings.has(currentAudioOwner?.scene) || queuedContinue) {
        queuedGameOver = true;
        return;
      }
      playAudioScene("gameOver");
      return;
    }
    if (data.name === "gill_intro") {
      playAudioScene("gillIntro");
    }
  };

  const handleTransportMessage = (raw) => {
    const message = JSON.parse(raw);
    if (message.type === "game_state") {
      handleGameState(message.data);
    } else if (message.type === "presentation") {
      handlePresentation(message.data);
    }
  };

  const hasRenderableFrame = () =>
    !remoteVideo.paused &&
    remoteVideo.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
    remoteVideo.videoWidth > 0 &&
    remoteVideo.videoHeight > 0;

  const handleFrameData = () => {
    const state = GameState.get();
    if (!state.loaded) {
      return;
    }

    if (!hasRenderableFrame()) {
      return;
    }

    if (!state.firstFrameReceived) {
      GameState.update({ firstFrameReceived: true });
    }
  };

  const ensureVideoPlaying = () => {
    if (!remoteVideo.srcObject) {
      return;
    }
    const playResult = remoteVideo.play();
    if (playResult && typeof playResult.catch === "function") {
      playResult.catch(() => {});
    }
  };

  const restartVideoRendering = () => {
    cancelVideoFrameObserver();
    if (renderFrameHandle !== null) {
      clearTimeout(renderFrameHandle);
      renderFrameHandle = null;
    }
    pendingVideoFrame = false;
    lastPresentationAt = null;
    ensureVideoPlaying();
    scheduleVideoFrameObserver();
    if (typeof remoteVideo.requestVideoFrameCallback !== "function") {
      pendingVideoFrame = true;
      handleFrameData();
      scheduleVideoPresentation();
    }
  };

  const renderVideoFrame = () => {
    const now = performance.now();
    if (
      lastPresentationAt !== null &&
      now - lastPresentationAt < presentationIntervalMs
    ) {
      scheduleVideoPresentation();
      return;
    }
    const canvas = byId("game-canvas");
    const state = GameState.get();
    const frameReady =
      pendingVideoFrame ||
      typeof remoteVideo.requestVideoFrameCallback !== "function";
    if (
      frameReady &&
      canvas &&
      hasRenderableFrame() &&
      state.loaded &&
      state.firstFrameReceived &&
      state.gamePhase !== "models_loading"
    ) {
      const ctx = canvas.getContext("2d");
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(remoteVideo, 0, 0, canvas.width, canvas.height);
      pendingVideoFrame = false;
      lastPresentationAt = performance.now();
    }
    if (typeof remoteVideo.requestVideoFrameCallback !== "function") {
      pendingVideoFrame = true;
      scheduleVideoPresentation();
    }
  };

  const handleRemoteStream = (stream) => {
    remoteVideo.srcObject = stream;
    remoteVideo.onloadedmetadata = restartVideoRendering;
    ensureVideoPlaying();
  };

  const handleDisconnect = (message) => {
    const state = GameState.get();
    if (state.currentScreen === ScreenManager.screens.ERROR) return;
    GameState.update({
      loaded: false,
      acceptsInput: false,
      serverReady: false,
    });
    endFlow = "idle";
    pendingFinishedState = null;
    stopAllAudio();
    ScreenManager.showError(message);
  };

  const enterLiveShell = ({ acceptsInput, uiActive }) => {
    GameState.update({
      loaded: true,
      acceptsInput,
      keyState: acceptsInput ? GameState.getKeyState() : {},
    });
    setCanvasSize();
    const state = GameState.get();
    if (state.currentScreen !== ScreenManager.screens.GAME) {
      ScreenManager.showScreen(ScreenManager.screens.GAME);
    }
    restartVideoRendering();
    GamepadManager.setUIActive(uiActive);
    GamepadUINavigator.updateGamepadSections(true);
  };

  const handleGameState = (data) => {
    const participantsAreAuthoritative = !["initializing", "pregame"].includes(
      data.status
    );
    GameState.update({
      gamePhase: data.status,
      ...(participantsAreAuthoritative
        ? {
            player1Participant: data.player1_participant,
            player2Participant: data.player2_participant,
          }
        : {}),
    });
    switch (data.status) {
      case "initializing":
        setText("loading-status", "Starting game...");
        playLobbyMusic();
        if (GameState.getCurrentScreen() === null) {
          ScreenManager.showScreen(ScreenManager.screens.LOBBY);
        }
        break;

      case "pregame": {
        const waitingForReplay = endFlow === "ending" || endFlow === "replay";
        GameState.update({
          loaded: true,
          serverReady: true,
          acceptsInput: false,
        });
        if (!waitingForReplay) setReplayVisible(false);
        if (!waitingForReplay) {
          ScreenManager.showScreen(ScreenManager.screens.LOBBY);
          GamepadManager.setUIActive(true);
          playLobbyMusic();
        }
        setCanvasSize();
        restartVideoRendering();
        break;
      }

      case "starting":
        setCanvasLoading(false);
        enterLiveShell({
          acceptsInput: false,
          uiActive: true,
        });
        break;

      case "selecting": {
        ScreenManager.showScreen(ScreenManager.screens.GAME);
        enterLiveShell({
          acceptsInput: !!data.accepts_input,
          uiActive: false,
        });
        const selections = {};
        if (data.player1_selection) {
          selections.player1 = data.player1_selection;
        }
        if (data.player2_selection) {
          selections.player2 = data.player2_selection;
        }
        const humanSeat = getHumanSeat(GameState.get());
        if (humanSeat === "P2" && data.player2_selection?.character) {
          selections.currentCharacter = data.player2_selection.character;
        } else if (data.player1_selection?.character) {
          selections.currentCharacter = data.player1_selection.character;
        }
        if (Object.keys(selections).length > 0) {
          GameState.update(selections);
        }
        setCanvasLoading(false);
        if (!(data.player1_selection && data.player2_selection)) {
          playSelectMusic();
        }
        break;
      }

      case "models_loading":
        GameState.update({
          loaded: true,
          acceptsInput: false,
          keyState: {},
        });
        ScreenManager.showScreen(ScreenManager.screens.GAME);
        setCanvasLoading(true, "Models cold-starting...");
        GamepadManager.setUIActive(true);
        GamepadUINavigator.updateGamepadSections(true);
        break;

      case "transitioning":
        setCanvasLoading(false);
        enterLiveShell({ acceptsInput: false, uiActive: true });
        break;

      case "running": {
        const state = GameState.get();
        const identity = data.match_identity;
        const resumedMatch =
          currentAudioOwner?.scene !== "gameplay" &&
          !overlayStings.has(currentAudioOwner?.scene);
        if (state.currentScreen !== ScreenManager.screens.GAME) {
          ScreenManager.showScreen(ScreenManager.screens.GAME);
        }
        GameState.update({
          loaded: true,
          acceptsInput: !!data.accepts_input,
          ...(identity
            ? {
                player1: identity.player1,
                player2: identity.player2,
                currentCharacter:
                  getHumanSeat(state) === "P2"
                    ? identity.player2.character
                    : identity.player1.character,
              }
            : {}),
        });
        setCanvasSize();
        setCanvasLoading(false);
        restartVideoRendering();
        if (resumedMatch && identity) {
          playGameplayMusic({
            player1: identity.player1,
            player2: identity.player2,
            round_number: data.round_number,
          });
        }
        GamepadManager.setUIActive(!hasHumanParticipant(state));
        GamepadUINavigator.updateGamepadSections(true);
        break;
      }

      case "finished":
        GameState.update({
          loaded: true,
          acceptsInput: false,
          keyState: {},
        });
        ScreenManager.showScreen(ScreenManager.screens.GAME);
        setCanvasSize();
        restartVideoRendering();
        endFlow = "ending";
        pendingFinishedState = data;
        revealReplayIfReady();
        break;

      case "error":
        GameState.update({ loaded: false, acceptsInput: false });
        endFlow = "idle";
        pendingFinishedState = null;
        stopAllAudio();
        ScreenManager.showError(data.error || "Unknown game error");
        break;
    }
  };

  const handleGameFinished = () => {
    endFlow = "replay";
    GameState.update({
      loaded: true,
      acceptsInput: false,
      keyState: {},
    });
    setCanvasLoading(false);
    setReplayVisible(true);
    GamepadManager.setUIActive(true);
    GamepadUINavigator.updateGamepadSections(true);
  };

  const init = () => {
    if (!remoteVideo.isConnected) {
      document.body.appendChild(remoteVideo);
    }

    WebRtcManager.init({
      onMessage: handleTransportMessage,
      onRemoteStream: handleRemoteStream,
      onDisconnect: handleDisconnect,
    });

    const startGameButton = byId("start-game-btn");
    if (startGameButton) {
      startGameButton.addEventListener("click", startGame);
    }

    document.addEventListener("gameStartRequest", () => {
      startGame();
    });

    document.addEventListener("keydown", (event) => {
      if (event.repeat) return;

      if (event.code === "Enter" || event.code === "Space") {
        const target =
          event.target instanceof Element ? event.target : null;
        if (
          target?.closest(
            "select, button, a, input, textarea, [contenteditable]"
          )
        ) {
          return;
        }
        if (startGame()) {
          event.preventDefault();
        }
      }
    });

    const playAgainBtn = byId("play-again-btn");
    if (playAgainBtn) {
      playAgainBtn.addEventListener("click", () => {
        endFlow = "rematch";
        playLobbyMusic();
        setReplayVisible(false);
        GameState.update({
          currentCharacter: null,
          player1: { character: null, outfit: 1, superArt: 1 },
          player2: { character: null, outfit: 1, superArt: 1 },
        });
        ScreenManager.showScreen(ScreenManager.screens.LOBBY);
        GamepadManager.setUIActive(true);
        requestAnimationFrame(() => byId("start-game-btn")?.focus());
      });
    }

    const errorBackBtn = byId("error-back-btn");
    if (errorBackBtn) {
      errorBackBtn.addEventListener("click", async () => {
        AudioManager.playSound(SOUND_KEYS.CLICK);
        WebRtcManager.clearStoredGameId();
        const launcherUrl = await WebRtcManager.getLauncherUrl();
        if (launcherUrl) {
          window.location.assign(launcherUrl);
        } else {
          window.location.reload();
        }
      });
    }
    playLobbyMusic();
  };

  const cleanup = () => {
    if (renderFrameHandle !== null) {
      clearTimeout(renderFrameHandle);
      renderFrameHandle = null;
    }
    cancelVideoFrameObserver();
    stopAllAudio();
    WebRtcManager.close();
  };

  return {
    init,
    startGame,
    cleanup,
  };
};

export const GameController = createGameController();
