import { GameState } from "./gameState.js";
import { GamepadManager } from "./gamepadManager.js";
import { WebRtcManager } from "./webRtcManager.js";
import { hasHumanParticipant } from "./participantOptions.js";

const createInputController = () => {
  const BITS = {
    left: 1,
    right: 2,
    up: 4,
    down: 8,
    LP: 16,
    MP: 32,
    HP: 64,
    LK: 128,
    MK: 256,
    HK: 512,
  };
  const DIRECTION_MASK = BITS.left | BITS.right | BITS.up | BITS.down;
  const ATTACK_KEYS = ["LP", "MP", "HP", "LK", "MK", "HK"];
  const ATTACK_MASK = ATTACK_KEYS.reduce((bits, key) => bits | BITS[key], 0);
  const DIRECTION_GLYPHS = {
    [BITS.left]: "←",
    [BITS.left | BITS.up]: "↖",
    [BITS.up]: "↑",
    [BITS.up | BITS.right]: "↗",
    [BITS.right]: "→",
    [BITS.right | BITS.down]: "↘",
    [BITS.down]: "↓",
    [BITS.down | BITS.left]: "↙",
  };
  const ATTACK_GLYPHS = {
    keyboard: { LP: "J", MP: "K", HP: "L", LK: "U", MK: "I", HK: "O" },
    gamepad: { LP: "A", MP: "B", HP: "RB", LK: "X", MK: "Y", HK: "LB" },
  };

  const directionBits = ({ left, right, up, down }) =>
    (left ? BITS.left : right ? BITS.right : 0) |
    (up ? BITS.up : down ? BITS.down : 0);

  const attackBits = (attacks) =>
    ATTACK_KEYS.reduce(
      (bits, key) => bits | (attacks[key] ? BITS[key] : 0),
      0
    );

  const displayFrame = (frame, isGamepad) => {
    if (!frame) return "";
    const glyphs = isGamepad
      ? ATTACK_GLYPHS.gamepad
      : ATTACK_GLYPHS.keyboard;
    const symbols = [];
    const direction = DIRECTION_GLYPHS[frame & DIRECTION_MASK];
    if (direction) symbols.push(direction);
    for (const key of ATTACK_KEYS) {
      if (frame & BITS[key]) symbols.push(glyphs[key]);
    }
    return symbols.join(" + ");
  };

  let combos = {};
  let specialMoves = {};
  let gamepadWasSelecting = false;
  let gamepadSelectionConfirmPressed = false;
  let gamepadHeldAttack = 0;
  let touchDirection = 0;
  let touchAttacks = 0;
  let pointerDownAt = null;
  let lastSelectTapAt = 0;

  const POINTER_TAP_DISTANCE_PX = 20;
  const DOUBLE_TAP_MS = 280;

  const getActionFromKeys = () => {
    const state = GameState.get();
    const keyState = GameState.getKeyState();
    const selecting = state.gamePhase === "selecting";

    const attacks = selecting
      ? {
          LP: keyState["Enter"],
          MP: false,
          HP: false,
          LK: false,
          MK: false,
          HK: false,
        }
      : {
          LP: keyState["KeyJ"] || (touchAttacks & BITS.LP) !== 0,
          MP: keyState["KeyK"] || (touchAttacks & BITS.MP) !== 0,
          HP: keyState["KeyL"] || (touchAttacks & BITS.HP) !== 0,
          LK: keyState["KeyU"] || (touchAttacks & BITS.LK) !== 0,
          MK: keyState["KeyI"] || (touchAttacks & BITS.MK) !== 0,
          HK: keyState["KeyO"] || (touchAttacks & BITS.HK) !== 0,
        };

    const directions = {
      left:
        keyState["KeyA"] ||
        keyState["ArrowLeft"] ||
        (touchDirection & BITS.left) !== 0,
      right:
        keyState["KeyD"] ||
        keyState["ArrowRight"] ||
        (touchDirection & BITS.right) !== 0,
      up:
        keyState["KeyW"] ||
        keyState["ArrowUp"] ||
        (touchDirection & BITS.up) !== 0,
      down:
        keyState["KeyS"] ||
        keyState["ArrowDown"] ||
        (touchDirection & BITS.down) !== 0,
    };

    return directionBits(directions) | attackBits(attacks);
  };

  const processGamepadInput = (currentState) => {
    const gameState = GameState.get();
    if (!gameState.acceptsInput) {
      return;
    }

    const threshold = GamepadManager.gameplayThreshold;
    const selecting = gameState.gamePhase === "selecting";
    const selectionConfirmPressed = currentState.buttons[0] || false;
    if (selecting && !gamepadWasSelecting) {
      gamepadSelectionConfirmPressed = selectionConfirmPressed;
    }
    const selectionConfirm =
      selecting &&
      selectionConfirmPressed &&
      !gamepadSelectionConfirmPressed;
    gamepadSelectionConfirmPressed = selecting
      ? selectionConfirmPressed
      : false;
    gamepadWasSelecting = selecting;

    const stickX = currentState.axes.left.x;
    const stickY = currentState.axes.left.y;

    const directions = {
      left: stickX < -threshold || currentState.buttons[14],
      right: stickX > threshold || currentState.buttons[15],
      up: stickY < -threshold || currentState.buttons[12],
      down: stickY > threshold || currentState.buttons[13],
    };

    const attacks = selecting
      ? {
          LP: selectionConfirm,
          MP: false,
          LK: false,
          MK: false,
          HK: false,
          HP: false,
        }
      : {
          LP: currentState.buttons[0],
          MP: currentState.buttons[1],
          LK: currentState.buttons[2],
          MK: currentState.buttons[3],
          HK: currentState.buttons[4],
          HP: currentState.buttons[5],
        };

    const attack = attackBits(attacks);
    const attackPressed = attack !== 0 && attack !== gamepadHeldAttack;
    gamepadHeldAttack = attack;

    const direction = directionBits(directions);
    handleActionFromInput(attackPressed ? attack | direction : direction);
  };

  const handleActionFromInput = (action) => {
    if (!GameState.get().acceptsInput) return;
    WebRtcManager.send("player_action", { action });
  };

  const handleKeyboardEvent = (e, isDown) => {
    const state = GameState.get();
    if (
      !state.loaded ||
      !state.acceptsInput ||
      (state.gamePhase !== "selecting" && !hasHumanParticipant(state))
    )
      return;
    if (state.gamePhase === "selecting" && e.code === "Enter" && e.repeat) {
      return;
    }

    GameState.setKeyState(e.code, isDown);
    const action = getActionFromKeys();
    handleActionFromInput(action);
    e.preventDefault();
  };

  const setExtraMoves = (extraCombos, extraSpecialMoves) => {
    combos = extraCombos;
    specialMoves = extraSpecialMoves;
  };

  const setTouchDirection = (mask) => {
    if (mask === touchDirection) return;
    touchDirection = mask;
    handleActionFromInput(getActionFromKeys() & ~ATTACK_MASK);
  };

  const setTouchAttacks = (mask) => {
    if (mask === touchAttacks) return;
    touchAttacks = mask;
    handleActionFromInput(getActionFromKeys());
  };

  const releaseAllKeys = () => {
    GameState.update({ keyState: {} });
    touchDirection = 0;
    touchAttacks = 0;
    handleActionFromInput(0);
  };

  const handlePointerSelect = (e) => {
    const state = GameState.get();
    if (!state.loaded) return;
    const target = e.target instanceof Element ? e.target : null;
    if (
      target?.closest(
        "select, button, a, input, textarea, [contenteditable], #help-overlay"
      ) ||
      (pointerDownAt &&
        Math.hypot(e.clientX - pointerDownAt.x, e.clientY - pointerDownAt.y) >=
          POINTER_TAP_DISTANCE_PX)
    )
      return;
    if (state.gamePhase === "pregame") {
      document.dispatchEvent(new Event("gameStartRequest"));
      return;
    }
    if (state.gamePhase === "starting") {
      WebRtcManager.send("fast_forward", {});
      return;
    }
    if (
      state.gamePhase === "selecting" &&
      state.acceptsInput &&
      target?.closest("#game-frame")
    ) {
      const now = performance.now();
      const isDouble = now - lastSelectTapAt < DOUBLE_TAP_MS;
      lastSelectTapAt = now;
      if (isDouble) handleActionFromInput(BITS.LP);
    }
  };

  const init = () => {
    document.addEventListener("keydown", (e) => handleKeyboardEvent(e, true));
    document.addEventListener("keyup", (e) => handleKeyboardEvent(e, false));
    document.addEventListener("pointerdown", (e) => {
      pointerDownAt = { x: e.clientX, y: e.clientY };
    });
    document.addEventListener("click", handlePointerSelect);
    window.addEventListener("blur", releaseAllKeys);
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) releaseAllKeys();
    });
  };

  const initGamepadInput = () => {
    const originalOnInput = GamepadManager.onInput;
    GamepadManager.onInput = (state) => {
      const gameState = GameState.get();
      if (
        gameState.loaded &&
        gameState.acceptsInput &&
        (gameState.gamePhase === "selecting" ||
          hasHumanParticipant(gameState))
      ) {
        processGamepadInput(state);
      } else if (originalOnInput) {
        originalOnInput(state);
      }
    };
  };

  return {
    BITS,
    displayFrame,
    init,
    initGamepadInput,
    handleActionFromInput,
    processGamepadInput,
    setExtraMoves,
    setTouchDirection,
    setTouchAttacks,
    getCombos: () => combos,
    getSpecialMoves: () => specialMoves,
  };
};

export const InputController = createInputController();
