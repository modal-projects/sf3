import { byId } from "./utils.js";
import { GameState } from "./gameState.js";
import { InputController } from "./inputController.js";

const STICK_DEAD_ZONE = 20;
const STICK_NUB_RANGE = 34;
const SWIPE_DISTANCE = 36;
const HOLD_MS = 350;
const DOUBLE_TAP_MS = 280;
const ATTACK_RELEASE_MS = 80;

const createTouchController = () => {
  const { BITS } = InputController;
  // clockwise from right, screen coords (y grows downward)
  const OCTANT_MASKS = [
    BITS.right,
    BITS.right | BITS.down,
    BITS.down,
    BITS.down | BITS.left,
    BITS.left,
    BITS.left | BITS.up,
    BITS.up,
    BITS.up | BITS.right,
  ];
  const SWIPE_ATTACKS = { up: "HK", left: "MK", right: "MK", down: "LK" };

  let movePointer = null; // { id, ox, oy }
  const attackPointers = new Map(); // id -> { ox, oy, t0, fired, holdTimer }
  let attackHeld = 0;
  let lastTapAt = 0;

  const setStickPosition = (x, y) => {
    const nub = byId("touch-stick-nub");
    if (!nub) return;
    nub.style.transform = `translate(${x}px, ${y}px)`;
  };

  const showStick = (x, y) => {
    const base = byId("touch-stick-base");
    if (!base) return;
    base.style.left = `${x}px`;
    base.style.top = `${y}px`;
    base.classList.remove("hidden");
    setStickPosition(0, 0);
  };

  const hideStick = () => {
    byId("touch-stick-base")?.classList.add("hidden");
  };

  const stickDirection = (dx, dy) => {
    if (Math.hypot(dx, dy) < STICK_DEAD_ZONE) return 0;
    const octant = Math.round(Math.atan2(dy, dx) / (Math.PI / 4));
    return OCTANT_MASKS[(octant + 8) % 8];
  };

  const flashAttack = (name, x, y) => {
    const zone = byId("touch-attack");
    if (!zone) return;
    const flash = document.createElement("div");
    flash.className = "touch-flash";
    flash.textContent = name;
    flash.style.left = `${x - zone.getBoundingClientRect().left}px`;
    flash.style.top = `${y - zone.getBoundingClientRect().top}px`;
    zone.appendChild(flash);
    setTimeout(() => flash.remove(), 500);
  };

  const fireAttack = (name, x, y) => {
    attackHeld |= BITS[name];
    InputController.setTouchAttacks(attackHeld);
    flashAttack(name, x, y);
    setTimeout(() => {
      attackHeld &= ~BITS[name];
      InputController.setTouchAttacks(attackHeld);
    }, ATTACK_RELEASE_MS);
  };

  const bindMovement = (zone) => {
    zone.addEventListener("pointerdown", (e) => {
      if (movePointer) return;
      e.preventDefault();
      zone.setPointerCapture(e.pointerId);
      movePointer = { id: e.pointerId, ox: e.clientX, oy: e.clientY };
      showStick(e.clientX - zone.getBoundingClientRect().left, e.clientY - zone.getBoundingClientRect().top);
    });
    zone.addEventListener("pointermove", (e) => {
      if (movePointer?.id !== e.pointerId) return;
      e.preventDefault();
      const dx = e.clientX - movePointer.ox;
      const dy = e.clientY - movePointer.oy;
      const len = Math.hypot(dx, dy) || 1;
      const scale = Math.min(len, STICK_NUB_RANGE) / len;
      setStickPosition(dx * scale, dy * scale);
      InputController.setTouchDirection(stickDirection(dx, dy));
    });
    const release = (e) => {
      if (movePointer?.id !== e.pointerId) return;
      movePointer = null;
      hideStick();
      InputController.setTouchDirection(0);
    };
    zone.addEventListener("pointerup", release);
    zone.addEventListener("pointercancel", release);
    zone.addEventListener("contextmenu", (e) => e.preventDefault());
  };

  const swipeDirection = (dx, dy) => {
    if (Math.hypot(dx, dy) < SWIPE_DISTANCE) return null;
    return Math.abs(dy) > Math.abs(dx) ? (dy < 0 ? "up" : "down") : dx < 0 ? "left" : "right";
  };

  const bindAttacks = (zone) => {
    zone.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      zone.setPointerCapture(e.pointerId);
      const pointer = {
        ox: e.clientX,
        oy: e.clientY,
        fired: false,
        holdTimer: null,
      };
      pointer.holdTimer = setTimeout(() => {
        if (pointer.fired) return;
        pointer.fired = true;
        fireAttack("HP", e.clientX, e.clientY);
      }, HOLD_MS);
      attackPointers.set(e.pointerId, pointer);
    });
    zone.addEventListener("pointermove", (e) => {
      const pointer = attackPointers.get(e.pointerId);
      if (!pointer || pointer.fired) return;
      const dir = swipeDirection(e.clientX - pointer.ox, e.clientY - pointer.oy);
      if (!dir) return;
      pointer.fired = true;
      clearTimeout(pointer.holdTimer);
      fireAttack(SWIPE_ATTACKS[dir], e.clientX, e.clientY);
    });
    const release = (e) => {
      const pointer = attackPointers.get(e.pointerId);
      if (!pointer) return;
      attackPointers.delete(e.pointerId);
      clearTimeout(pointer.holdTimer);
      if (pointer.fired || e.type === "pointercancel") return;
      const dir = swipeDirection(e.clientX - pointer.ox, e.clientY - pointer.oy);
      if (dir) {
        fireAttack(SWIPE_ATTACKS[dir], e.clientX, e.clientY);
        return;
      }
      const now = performance.now();
      const isDouble = now - lastTapAt < DOUBLE_TAP_MS;
      lastTapAt = now;
      fireAttack(isDouble ? "MP" : "LP", e.clientX, e.clientY);
    };
    zone.addEventListener("pointerup", release);
    zone.addEventListener("pointercancel", release);
    zone.addEventListener("contextmenu", (e) => e.preventDefault());
  };

  const releaseAll = () => {
    movePointer = null;
    for (const pointer of attackPointers.values()) clearTimeout(pointer.holdTimer);
    attackPointers.clear();
    attackHeld = 0;
    hideStick();
    InputController.setTouchDirection(0);
    InputController.setTouchAttacks(0);
  };

  const refreshVisibility = () => {
    const controls = byId("touch-controls");
    if (!controls) return;
    const inGame = GameState.getCurrentScreen() === "game";
    controls.classList.toggle(
      "hidden",
      !(inGame && GameState.get().acceptsInput)
    );
  };

  const init = () => {
    const controls = byId("touch-controls");
    if (!controls || navigator.maxTouchPoints === 0) return;

    const move = byId("touch-move");
    const attack = byId("touch-attack");
    if (move) bindMovement(move);
    if (attack) bindAttacks(attack);

    GameState.subscribe((changeType) => {
      if (
        changeType === "screenChange" ||
        changeType === "gameModeChange" ||
        changeType === "update"
      ) {
        refreshVisibility();
      }
    });
    window.addEventListener("blur", releaseAll);
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) releaseAll();
    });
    refreshVisibility();
  };

  return { init, refreshVisibility };
};

export const TouchController = createTouchController();
