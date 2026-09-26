import { AudioManager } from "./audioManager.js";
import { GameState } from "./gameState.js";
import { setText } from "./utils.js";
import { PARTICIPANT_SPECS } from "./participantOptions.js";
import { SOUND_KEYS } from "./constants.js";

const createAssetLoader = () => {
  const characters = [
    "Alex",
    "Chun-Li",
    "Dudley",
    "Elena",
    "Gouki",
    "Hugo",
    "Ibuki",
    "Ken",
    "Makoto",
    "Necro",
    "Oro",
    "Q",
    "Remy",
    "Ryu",
    "Sean",
    "Twelve",
    "Urien",
    "Yang",
    "Yun",
  ];

  const staticImages = {
    MUTE: "/icons/mute.svg",
    UNMUTE: "/icons/unmute.svg",
    HELP: "/icons/help.webp",
    CLOSE: "/icons/close.webp",
    GAMEPAD: "/icons/gamepad.webp",
    KEYBOARD: "/icons/keyboard.webp",
    MODAL: "/logos/modal.webp",
  };

  const soundFiles = {
    [SOUND_KEYS.HOVER]: "extras/hover.mp3",
    [SOUND_KEYS.CLICK]: "extras/click.mp3",
    [SOUND_KEYS.COIN]: "extras/coin.mp3",
    [SOUND_KEYS.GAMEPAD_CONNECT]: "extras/gamepad-connect.mp3",
    [SOUND_KEYS.GAMEPAD_DISCONNECT]: "extras/gamepad-disconnect.mp3",
    [SOUND_KEYS.CAPCOM]: "album/01. Capcom Logo.mp3",
    [SOUND_KEYS.START]: "album/02. Opening.mp3",
    [SOUND_KEYS.MAIN_MENU]: "album/04. Select Your Opponent.mp3",
    [SOUND_KEYS.SELECT]: "album/03. Character Select.mp3",
    [SOUND_KEYS.TRANSITION]: "album/05. Versus.mp3",
    [SOUND_KEYS.WIN]: "album/58. Winner.mp3",
    [SOUND_KEYS.GAME_OVER]: "album/63. Game Over.mp3",
    [SOUND_KEYS.CONTINUE]: "album/59. Continue.mp3",
    [SOUND_KEYS.GILL_INTRO]: "album/55. Gill Intro.mp3",
    [SOUND_KEYS.JUDGEMENT]: "album/57. Judgement.mp3",
  };

  const gameplayRoundTracks = {
    Gouki: [
      "album/21. Akuma (Round 1).mp3",
      "album/22. Akuma (Round 2).mp3",
    ],
    Alex: [
      "album/23. Alex & Ken (Round 1).mp3",
      "album/24. Alex & Ken (Round 2).mp3",
    ],
    Ken: [
      "album/23. Alex & Ken (Round 1).mp3",
      "album/24. Alex & Ken (Round 2).mp3",
    ],
    "Chun-Li": [
      "album/25. Chun-Li (Round 1).mp3",
      "album/26. Chun-Li (Round 2).mp3",
    ],
    Dudley: [
      "album/27. Dudley (Round 1).mp3",
      "album/28. Dudley (Round 2).mp3",
    ],
    Elena: [
      "album/29. Elena (Round 1).mp3",
      "album/30. Elena (Round 2).mp3",
    ],
    Gill: [
      "album/31. Gill (Round 1).mp3",
      "album/32. Gill (Round 2).mp3",
    ],
    Hugo: [
      "album/33. Hugo (Round 1).mp3",
      "album/34. Hugo (Round 2).mp3",
    ],
    Ibuki: [
      "album/35. Ibuki (Round 1).mp3",
      "album/36. Ibuki (Round 2).mp3",
    ],
    Makoto: [
      "album/37. Makoto (Round 1).mp3",
      "album/38. Makoto (Round 2).mp3",
    ],
    Necro: [
      "album/39. Necro & Twelve (Round 1).mp3",
      "album/40. Necro & Twelve (Round 2).mp3",
    ],
    Twelve: [
      "album/39. Necro & Twelve (Round 1).mp3",
      "album/40. Necro & Twelve (Round 2).mp3",
    ],
    Q: ["album/41. Q (Round 1).mp3", "album/42. Q (Round 2).mp3"],
    Remy: [
      "album/43. Remy (Round 1).mp3",
      "album/44. Remy (Round 2).mp3",
    ],
    Ryu: [
      "album/45. Ryu (Round 1).mp3",
      "album/46. Ryu (Round 2).mp3",
    ],
    Sean: [
      "album/47. Sean & Oro (Round 1).mp3",
      "album/48. Sean & Oro (Round 2).mp3",
    ],
    Oro: [
      "album/47. Sean & Oro (Round 1).mp3",
      "album/48. Sean & Oro (Round 2).mp3",
    ],
    Urien: [
      "album/49. Urien (Round 1).mp3",
      "album/50. Urien (Round 2).mp3",
    ],
    Yun: [
      "album/51. Yun & Yang (Round 1).mp3",
      "album/52. Yun & Yang (Round 2).mp3",
    ],
    Yang: [
      "album/51. Yun & Yang (Round 1).mp3",
      "album/52. Yun & Yang (Round 2).mp3",
    ],
  };

  const gameplayMusicMap = Object.fromEntries(
    Object.entries(gameplayRoundTracks).flatMap(([character, [round1, round2]]) => [
      [`${character}_r1`, round1],
      [`${character}_r2`, round2],
    ])
  );

  const gameplaySoundKey = (character, roundNumber) => {
    if (!character || !gameplayRoundTracks[character]) return null;
    const variant = Number(roundNumber) > 1 ? 2 : 1;
    return `${character}_r${variant}`;
  };

  const preloadImage = (src) => {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = resolve;
      img.onerror = resolve;
      img.src = src;
    });
  };

  const loadAllAssets = async () => {
    setText("loading-status", "Loading assets...");

    const imageSources = [
      ...Object.values(staticImages),
      ...Object.values(PARTICIPANT_SPECS).map(({ logo }) => logo),
    ];
    const imagePromises = imageSources.map(preloadImage);

    await Promise.all([
      AudioManager.preloadSounds(soundFiles, gameplayMusicMap),
      ...imagePromises,
    ]);

    GameState.update({ assetsLoaded: true });
    setText("loading-status", "Connecting to server...");
  };

  const loadExtraMoves = async () => {
    try {
      const response = await fetch("/api/extra-moves");
      const data = await response.json();
      return {
        combos: data.combos || {},
        specialMoves: data.special_moves || {},
      };
    } catch (error) {
      console.error("Failed to load extra moves:", error);
      return {
        combos: {},
        specialMoves: {},
      };
    }
  };

  return {
    characters,
    staticImages,
    soundFiles,
    gameplayRoundTracks,
    gameplayMusicMap,
    gameplaySoundKey,
    loadAllAssets,
    loadExtraMoves,
  };
};

export const AssetLoader = createAssetLoader();
export const gameplaySoundKey = (character, roundNumber) =>
  AssetLoader.gameplaySoundKey(character, roundNumber);
