import { byId } from "./utils.js";
import { GameState } from "./gameState.js";
import { InputController } from "./inputController.js";
import { hasHumanParticipant } from "./participantOptions.js";

const createTouchController = () => {
  const { BITS } = InputController;
  const DIRECTION_MASKS = {
    "up-left": BITS.left | BITS.up,
    up: BITS.up,
    "up-right": BITS.up | BITS.right,
    left: BITS.left,
    right: BITS.right,
    "down-left": BITS.down | BITS.left,
    down: BITS.down,
    "down-right": BITS.down | BITS.right,
  };

  const activeDirections = new Map();
  const activeAttacks = new Map();

  const emitDirection = () => {
    let mask = 0;
    for (const entry of activeDirections.values()) mask |= entry.mask;
    InputController.setTouchDirection(mask);
  };

  const emitAttacks = () => {
    let mask = 0;
    for (const entry of activeAttacks.values()) mask |= entry.mask;
    InputController.setTouchAttacks(mask);
  };

  const setDirection = (pointerId, el) => {
    const active = activeDirections.get(pointerId);
    if (active?.el === el) return;
    active?.el?.classList.remove("touch-active");
    activeDirections.set(
      pointerId,
      el
        ? { el, mask: DIRECTION_MASKS[el.dataset.touchDir] ?? 0 }
        : { el: null, mask: 0 }
    );
    el?.classList.add("touch-active");
    emitDirection();
  };

  const releaseDirection = (e) => {
    const active = activeDirections.get(e.pointerId);
    if (!active) return;
    active.el?.classList.remove("touch-active");
    activeDirections.delete(e.pointerId);
    emitDirection();
  };

  const setAttack = (pointerId, el) => {
    const active = activeAttacks.get(pointerId);
    if (el && active) return;
    if (el) {
      activeAttacks.set(pointerId, {
        el,
        mask: BITS[el.dataset.touchBtn] ?? 0,
      });
      el.classList.add("touch-active");
    } else {
      active?.el.classList.remove("touch-active");
      activeAttacks.delete(pointerId);
    }
    emitAttacks();
  };

  const releaseAll = () => {
    for (const entry of activeDirections.values()) {
      entry.el?.classList.remove("touch-active");
    }
    for (const entry of activeAttacks.values()) {
      entry.el?.classList.remove("touch-active");
    }
    activeDirections.clear();
    activeAttacks.clear();
    emitDirection();
    emitAttacks();
  };

  const refreshVisibility = () => {
    const controls = byId("touch-controls");
    if (!controls) return;
    const state = GameState.get();
    const inGame = GameState.getCurrentScreen() === "game";
    const usable =
      state.gamePhase === "selecting" ||
      (hasHumanParticipant(state) && state.gamePhase !== "finished");
    controls.classList.toggle("hidden", !(inGame && usable));
  };

  const bindDirections = (dpad) => {
    dpad.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      dpad.setPointerCapture(e.pointerId);
      setDirection(e.pointerId, e.target.closest("[data-touch-dir]"));
    });
    dpad.addEventListener("pointermove", (e) => {
      if (!activeDirections.has(e.pointerId)) return;
      e.preventDefault();
      const hit = document.elementFromPoint(e.clientX, e.clientY);
      if (hit && !hit.closest("#touch-dpad")) {
        setDirection(e.pointerId, null);
        return;
      }
      const el = hit?.closest("[data-touch-dir]");
      if (el) setDirection(e.pointerId, el);
    });
    dpad.addEventListener("pointerup", releaseDirection);
    dpad.addEventListener("pointercancel", releaseDirection);
    dpad.addEventListener("contextmenu", (e) => e.preventDefault());
  };

  const bindAttacks = (attacks) => {
    const handle = (e, press) => {
      const el = press ? e.target.closest("[data-touch-btn]") : null;
      if (press && !el) return;
      e.preventDefault();
      if (press) el.setPointerCapture(e.pointerId);
      setAttack(e.pointerId, el);
    };
    attacks.addEventListener("pointerdown", (e) => handle(e, true));
    attacks.addEventListener("pointerup", (e) => handle(e, false));
    attacks.addEventListener("pointercancel", (e) => handle(e, false));
    attacks.addEventListener("contextmenu", (e) => e.preventDefault());
  };

  const init = () => {
    const controls = byId("touch-controls");
    if (!controls || navigator.maxTouchPoints === 0) return;

    const dpad = byId("touch-dpad");
    const attacks = byId("touch-attacks");
    if (dpad) bindDirections(dpad);
    if (attacks) bindAttacks(attacks);

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
