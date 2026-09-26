import { byId } from "./utils.js";
import { GameState } from "./gameState.js";
import { AudioManager } from "./audioManager.js";
import { AssetLoader } from "./assetLoader.js";
import { GamepadManager } from "./gamepadManager.js";
import { InputController } from "./inputController.js";
import { MovesDisplay } from "./movesEngine.js";
import {
  GamepadUINavigator,
  directionFromAxes,
} from "./gamepadUINavigator.js";
import { ScreenManager } from "./screenManager.js";
import {
  getParticipantsForSeat,
  hasHumanParticipant,
  isHumanParticipant,
} from "./participantOptions.js";
import { SOUND_KEYS } from "./constants.js";

const createUIController = () => {
  const defaultModelParticipant = "qwen35_9b";

  const focusLobbyStart = () => {
    requestAnimationFrame(() => {
      const lobby = byId("lobby-screen");
      const start = byId("start-game-btn");
      if (
        !lobby ||
        !start ||
        GameState.getCurrentScreen() !== ScreenManager.screens.LOBBY ||
        lobby.contains(document.activeElement)
      ) {
        return;
      }
      start.focus();
    });
  };

  const setupKeyboardNavigation = () => {
    const arrowAxes = {
      ArrowUp: [0, -1],
      ArrowDown: [0, 1],
      ArrowLeft: [-1, 0],
      ArrowRight: [1, 0],
    };

    document.addEventListener("keydown", (event) => {
      if (event.defaultPrevented || !arrowAxes[event.code]) {
        return;
      }

      const screen = GameState.getCurrentScreen();
      const state = GameState.get();
      if (
        screen === ScreenManager.screens.GAME &&
        (state.gamePhase === "selecting" || hasHumanParticipant(state)) &&
        state.acceptsInput
      ) {
        return;
      }

      const [inputX, inputY] = arrowAxes[event.code];
      event.preventDefault();

      // Lobby keyboard arrows use the same neighbor graph as the gamepad.
      if (screen === ScreenManager.screens.LOBBY) {
        GamepadUINavigator.navigateLobbyDirection(
          directionFromAxes(inputX, inputY)
        );
        return;
      }

      GamepadUINavigator.handleDirectionalNavigation(inputX, inputY);
    });
  };

  const syncParticipantUI = () => {
    const state = GameState.get();
    document.querySelectorAll("[data-participant-seat]").forEach((button) => {
      const seat = button.dataset.participantSeat;
      const selected =
        seat === "P1"
          ? state.player1Participant
          : state.player2Participant;
      button.setAttribute(
        "aria-pressed",
        String(button.dataset.participant === selected)
      );
    });
  };

  const selectParticipant = (seat, participant) => {
    const state = GameState.get();
    if (isHumanParticipant(participant)) {
      if (
        seat === "P1" &&
        isHumanParticipant(state.player2Participant)
      ) {
        GameState.setPlayer2Participant(defaultModelParticipant);
      } else if (
        seat === "P2" &&
        isHumanParticipant(state.player1Participant)
      ) {
        GameState.setPlayer1Participant(defaultModelParticipant);
      }
    }

    if (seat === "P1") {
      GameState.setPlayer1Participant(participant);
    } else {
      GameState.setPlayer2Participant(participant);
    }
    ScreenManager.refreshVisibility();
    AudioManager.playSound(SOUND_KEYS.CLICK);
    syncParticipantUI();
    GamepadUINavigator.updateGamepadSections(true);
  };

  const setupParticipantLists = () => {
    for (const seat of ["P1", "P2"]) {
      const list = byId(`participant-list-${seat.toLowerCase()}`);
      if (!list) continue;

      list.replaceChildren();
      getParticipantsForSeat(seat).forEach(
        ({ participant, label, logo }) => {
          const button = document.createElement("button");
          button.type = "button";
          button.id = `participant-${seat.toLowerCase()}-${participant}`;
          button.className = `participant-card participant-card-${seat.toLowerCase()}`;
          button.dataset.participantSeat = seat;
          button.dataset.participant = participant;

          const image = document.createElement("img");
          image.src = logo;
          image.alt = "";
          image.setAttribute("aria-hidden", "true");

          const name = document.createElement("span");
          name.className = "min-w-0 break-words text-xs md:text-sm lg:text-base";
          name.textContent = label;

          button.append(image, name);
          button.addEventListener("click", () =>
            selectParticipant(seat, participant)
          );
          button.addEventListener("mouseenter", () =>
            AudioManager.playSound(SOUND_KEYS.HOVER)
          );
          list.appendChild(button);
        }
      );
    }
    syncParticipantUI();
  };

  const setupHelpOverlay = () => {
    const helpButton = byId("controls-help");
    const overlay = byId("help-overlay");
    const closeBtn = byId("help-overlay-close");

    if (!helpButton || !overlay || !closeBtn) return;

    let savedNavPosition = null;

    const openOverlay = () => {
      const navState = GameState.getGamepadUIState();
      savedNavPosition = {
        section: navState.currentSection,
        element: navState.currentElement,
      };

      overlay.classList.remove("hidden");
      updateControlsDisplay();
      updateGamepadNavVisibility();
      updateCharacterHelpVisibility();

      AudioManager.playSound(SOUND_KEYS.CLICK);
      GamepadManager.setUIActive(true);
      GamepadUINavigator.updateGamepadSections(true);
    };

    const closeOverlay = () => {
      overlay.classList.add("hidden");
      AudioManager.playSound(SOUND_KEYS.CLICK);
      GamepadManager.setUIActive(true);
      GamepadUINavigator.updateGamepadSections(true);

      if (savedNavPosition) {
        GamepadUINavigator.restoreNavPosition(savedNavPosition);
        savedNavPosition = null;
      }
    };

    helpButton.addEventListener("click", openOverlay);
    closeBtn.addEventListener("click", closeOverlay);
  };

  const updateLobbyReadiness = () => {
    const state = GameState.get();
    const button = byId("start-game-btn");
    if (!button) return;

    const ready = state.serverReady && state.gamePhase === "pregame";
    const status = ready ? "START GAME" : "BOOTING EMULATOR...";
    button.disabled = false;
    button.textContent = status;
    button.setAttribute("aria-disabled", String(!ready));
    button.setAttribute(
      "aria-label",
      ready ? "Emulator ready. Start game." : status
    );
    focusLobbyStart();
  };

  const updateCharacterHelpVisibility = () => {
    const state = GameState.get();
    const panel = byId("character-help-panel");
    if (!panel) return;

    const hasSelection = Boolean(
      state.currentCharacter && state.player1?.superArt
    );
    panel.classList.toggle("hidden", !hasSelection);
  };

  const updateControlsDisplay = () => {
    const onPad = GamepadManager.isConnected();
    const controls = {
      "movement-display": onPad ? "Left Stick / D-Pad" : "WASD / Arrow Keys",
      "lp-display": onPad ? "A" : "J",
      "mp-display": onPad ? "B" : "K",
      "hp-display": onPad ? "RB" : "L",
      "lk-display": onPad ? "X" : "U",
      "mk-display": onPad ? "Y" : "I",
      "hk-display": onPad ? "LB" : "O",
      "lplk-display": onPad ? "A + X" : "J + U",
      "mpmk-display": onPad ? "B + Y" : "K + I",
      "hphk-display": onPad ? "RB + LB" : "L + O",
    };

    Object.entries(controls).forEach(([id, text]) => {
      const element = byId(id);
      if (element) element.textContent = text;
    });
  };

  const updateCombosDisplay = (character) => {
    const combosList = byId("combos-list");
    if (!combosList) return;

    if (!character || !InputController.getCombos()[character]) {
      combosList.innerHTML =
        '<p class="text-sf-beige-dark">Character selected in-game</p>';
      return;
    }

    const state = GameState.get();
    const combos = InputController.getCombos()[character];

    const moves = Object.entries(combos).map(([name, data]) => ({
      name,
      sequence: data[state.playerDirection],
    }));

    combosList.innerHTML = MovesDisplay.generateMovesHTML(
      moves,
      InputController.idxToMove,
      GamepadManager.isConnected()
    );
  };

  const updateSuperArtsDisplay = (character) => {
    const superArtsList = byId("super-arts-list");
    if (!superArtsList) return;

    const state = GameState.get();
    character = character || state.currentCharacter;

    if (!character || !InputController.getSpecialMoves()[character]) {
      superArtsList.innerHTML =
        '<p class="text-sf-beige-dark">Character selected in-game</p>';
      return;
    }

    const selectedSuperArt = String(state.player1?.superArt || 1);
    const characterMoves = InputController.getSpecialMoves()[character];
    const moves = [];

    for (const [moveKey, moveData] of Object.entries(characterMoves)) {
      if (moveKey.startsWith(`${selectedSuperArt} `)) {
        moves.push({
          name: MovesDisplay.getSpecialMoveDisplayName(moveKey),
          sequence: moveData[state.playerDirection],
        });
        break;
      }
    }

    for (const [moveKey, moveData] of Object.entries(characterMoves)) {
      if (typeof moveKey === "string" && moveKey.startsWith("Max")) {
        if (
          moveKey.startsWith(`Max-${selectedSuperArt} `) ||
          moveKey.startsWith("Max ")
        ) {
          moves.push({
            name: MovesDisplay.getSpecialMoveDisplayName(moveKey),
            sequence: moveData[state.playerDirection],
          });
        }
      }
    }

    superArtsList.innerHTML = MovesDisplay.generateMovesHTML(
      moves,
      InputController.idxToMove,
      GamepadManager.isConnected()
    );
  };

  const updateGamepadNavVisibility = () => {
    const gamepadNavSection = byId("gamepad-nav-section");
    if (gamepadNavSection) {
      const showGamepadNav = GamepadManager.isConnected();
      gamepadNavSection.classList.toggle("hidden", !showGamepadNav);
    }
  };

  const loadExtraMovesDisplay = async () => {
    const elements = {
      combosLoading: byId("combos-loading"),
      superArtsLoading: byId("super-arts-loading"),
      combosList: byId("combos-list"),
      superArtsList: byId("super-arts-list"),
    };

    if (!elements.combosLoading || !elements.superArtsLoading) return;

    try {
      const data = await AssetLoader.loadExtraMoves();

      InputController.setExtraMoves(data.combos, data.specialMoves);

      const state = GameState.get();
      updateCombosDisplay(state.currentCharacter);
      updateSuperArtsDisplay(state.currentCharacter);

      elements.combosLoading.classList.add("hidden");
      elements.superArtsLoading.classList.add("hidden");
    } catch (error) {
      console.error("Failed to load combos:", error);
      elements.combosLoading.classList.add("hidden");
      elements.superArtsLoading.classList.add("hidden");
      elements.combosList.innerHTML =
        '<p class="text-sf-red">Failed to load combos</p>';
      elements.superArtsList.innerHTML =
        '<p class="text-sf-red">Failed to load super arts</p>';
    }
  };

  const setupHoverSounds = () => {
    const buttons = [
      "start-game-btn",
      "play-again-btn",
      "error-back-btn",
      "help-overlay-close",
      "controls-help",
    ];

    buttons.forEach((elemId) => {
      const elem = byId(elemId);
      if (elem) {
        elem.addEventListener("mouseenter", () =>
          AudioManager.playSound(SOUND_KEYS.HOVER)
        );
      }
    });
  };

  const init = () => {
    setupParticipantLists();
    setupKeyboardNavigation();
    setupHelpOverlay();
    setupHoverSounds();
    loadExtraMovesDisplay();
    updateLobbyReadiness();
    updateCharacterHelpVisibility();

    GameState.subscribe((changeType, data) => {
      if (changeType === "gameModeChange") {
        syncParticipantUI();
      }
      if (changeType === "update") {
        if ("serverReady" in data || "gamePhase" in data) {
          updateLobbyReadiness();
        }
        if ("player1Participant" in data || "player2Participant" in data) {
          syncParticipantUI();
        }
        if (
          "currentCharacter" in data ||
          "player1" in data ||
          "gamePhase" in data
        ) {
          updateCharacterHelpVisibility();
        }
        if ("currentCharacter" in data || "player1" in data) {
          const state = GameState.get();
          updateCombosDisplay(state.currentCharacter);
          updateSuperArtsDisplay(state.currentCharacter);
        }
      }
    });
  };

  return {
    init,
    updateControlsDisplay,
    updateCombosDisplay,
    updateSuperArtsDisplay,
    updateGamepadNavVisibility,
    updateLobbyReadiness,
    updateCharacterHelpVisibility,
  };
};

export const UIController = createUIController();
