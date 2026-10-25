import { AssetLoader } from "./assetLoader.js";
import { AudioManager } from "./audioManager.js";
import { GameState } from "./gameState.js";
import { ScreenManager } from "./screenManager.js";
import { InputController } from "./inputController.js";
import { TouchController } from "./touchController.js";
import { GameController } from "./gameController.js";
import { UIController } from "./uiController.js";
import { GamepadManager } from "./gamepadManager.js";
import { GamepadUINavigator } from "./gamepadUINavigator.js";
import { WebRtcManager } from "./webRtcManager.js";
import { byId } from "./utils.js";
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

  await AssetLoader.loadAllAssets();

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
  TouchController.init();
  GameController.init();
  UIController.init();
  GamepadUINavigator.init();
  if (!reconnecting && GameState.getCurrentScreen() === null) {
    ScreenManager.showScreen(ScreenManager.screens.LOBBY);
  }

  window.addEventListener("beforeunload", () => {
    AudioManager.stopAll();
    GameController.cleanup();
  });
};

document.addEventListener("DOMContentLoaded", () => {
  const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);

  const checkMobileRequirements = () => {
    const isPortrait = window.innerHeight > window.innerWidth;
    const rotateScreen = byId("rotate-device");
    if (!rotateScreen) return;

    if (isMobile && isPortrait) {
      rotateScreen.classList.remove("hidden");
      rotateScreen.classList.add("flex");
    } else {
      rotateScreen.classList.add("hidden");
      rotateScreen.classList.remove("flex");
    }
  };

  setCanvasSize();
  checkMobileRequirements();
  window.addEventListener("orientationchange", checkMobileRequirements);
  window.addEventListener("resize", checkMobileRequirements);

  initApp();
});
