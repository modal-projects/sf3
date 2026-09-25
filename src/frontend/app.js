import { AssetLoader } from "./assetLoader.js";
import { AudioManager } from "./audioManager.js";
import { GameState } from "./gameState.js";
import { ScreenManager } from "./screenManager.js";
import { InputController } from "./inputController.js";
import { GameController } from "./gameController.js";
import { UIController } from "./uiController.js";
import { GamepadManager } from "./gamepadManager.js";
import { GamepadUINavigator } from "./gamepadUINavigator.js";
import { WebRtcManager } from "./webRtcManager.js";
import { byId, setText } from "./utils.js";
import { GAME_SOURCE_SIZE } from "./constants.js";

export const setCanvasSize = () => {
  const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
  const canvas = byId("game-canvas");
  if (canvas) {
    const scale = isMobile ? 2 : 3;
    canvas.width = GAME_SOURCE_SIZE.width * scale;
    canvas.height = GAME_SOURCE_SIZE.height * scale;
  }
};

const initApp = async () => {
  const reconnecting = WebRtcManager.hasStoredGameId();
  AudioManager.init();
  WebRtcManager.init();
  if (reconnecting) {
    ScreenManager.showScreen(ScreenManager.screens.LOADING);
    setText("loading-status", "Reconnecting...");
  }

  await AssetLoader.loadAllAssets();
  if (
    reconnecting &&
    GameState.getCurrentScreen() === ScreenManager.screens.LOADING
  ) {
    setText("loading-status", "Reconnecting...");
  }

  GamepadManager.init({
    onStatusChange: () => {
      UIController.updateControlsDisplay();
      UIController.updateCombosDisplay(GameState.get().currentCharacter);
      UIController.updateSuperArtsDisplay(GameState.get().currentCharacter);
      ScreenManager.refreshVisibility();
      UIController.updateGamepadNavVisibility();
      GamepadUINavigator.updateGamepadSections(true);
    },
    onInput: () => {},
    onUIAction: () => {},
  });

  GamepadManager.setUIActive(true);

  InputController.init();
  InputController.initGamepadInput();
  GameController.init();
  UIController.init();
  GamepadUINavigator.init();
  if (GameState.getCurrentScreen() === null) {
    ScreenManager.showScreen(ScreenManager.screens.LOBBY);
  }

  window.addEventListener("beforeunload", () => {
    AudioManager.stopAll();
    GameController.cleanup();
  });
};

document.addEventListener("DOMContentLoaded", () => {
  const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
  let hasKeyboard = false;
  let hasGamepad = false;

  const checkMobileRequirements = () => {
    const isPortrait = window.innerHeight > window.innerWidth;
    const rotateScreen = byId("rotate-device");
    const requirementTitle = byId("requirement-title");
    const requirementText = byId("requirement-text");
    const requirementIcon = byId("requirement-icon");
    const inputIcons = byId("input-icons");

    if (rotateScreen && isMobile) {
      if (isPortrait) {
        rotateScreen.classList.remove("hidden");
        rotateScreen.classList.add("flex");
        requirementTitle.textContent = "Please Rotate Your Device";
        requirementText.textContent =
          "This game requires landscape orientation on mobile devices.";
        requirementIcon.classList.remove("hidden");
        inputIcons.classList.add("hidden");
        inputIcons.classList.remove("flex");
      } else if (!hasKeyboard && !hasGamepad) {
        rotateScreen.classList.remove("hidden");
        rotateScreen.classList.add("flex");
        requirementTitle.textContent = "Controller Required";
        requirementText.textContent =
          "Please connect a Bluetooth keyboard or gamepad to play on mobile.";
        requirementIcon.classList.add("hidden");
        inputIcons.classList.remove("hidden");
        inputIcons.classList.add("flex");
      } else {
        rotateScreen.classList.add("hidden");
        rotateScreen.classList.remove("flex");
      }
    } else if (rotateScreen) {
      rotateScreen.classList.add("hidden");
      rotateScreen.classList.remove("flex");
    }
  };

  window.addEventListener("keydown", () => {
    if (isMobile && !hasKeyboard) {
      hasKeyboard = true;
      checkMobileRequirements();
    }
  });

  window.addEventListener("gamepadStatusChange", (e) => {
    if (isMobile) {
      hasGamepad = e.detail.connected;
      checkMobileRequirements();
    }
  });

  setCanvasSize();
  checkMobileRequirements();
  window.addEventListener("orientationchange", checkMobileRequirements);
  window.addEventListener("resize", checkMobileRequirements);

  initApp();
});
