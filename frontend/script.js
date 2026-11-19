const API_BASE = "http://127.0.0.1:8000/api";
const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const FILES = "abcdefgh";
const PIECES = {
  K: "♔", Q: "♕", R: "♖", B: "♗", N: "♘", P: "♙",
  k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟"
};
const PROMOTION_PIECES = { q: "Queen", r: "Rook", b: "Bishop", n: "Knight" };
const STORAGE = {
  history: "house-chess-history-v1",
  settings: "house-chess-settings-v1",
  activeGame: "house-chess-active-game-v1"
};
const INITIAL_CLOCKS = { w: 0, b: 0 };

let game = null;
let legalMoves = [];
let selectedSquare = null;
let flipped = false;
let busy = false;
let clockTimer = null;
let lastClockTick = Date.now();
let lastClockSave = Date.now();
let apiConnected = false;
let playerColorChoice = "white";

const $ = selector => document.querySelector(selector);
const boardElement = $("#chessboard");

function readStorage(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch (error) {
    console.error(`Could not read ${key} from browser storage.`, error);
    return fallback;
  }
}

function writeStorage(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch (error) {
    showError(error.message || "Could not save your game in this browser.");
  }
}

function getSettings() {
  const saved = readStorage(STORAGE.settings, {});
  return {
    theme: ["classic", "slate", "blue"].includes(saved.theme) ? saved.theme : "classic",
    darkMode: saved.darkMode === true,
    sound: saved.sound === true
  };
}

let settings = getSettings();

async function api(path, body) {
  const options = body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  };
  const response = await fetch(`${API_BASE}${path}`, options);
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error(`The chess server returned an invalid response (${response.status}).`);
  }
  if (!response.ok) {
    const detail = typeof data.detail === "string" ? data.detail : `Request failed (${response.status}).`;
    throw new Error(detail);
  }
  return data;
}

function saveSettings() {
  writeStorage(STORAGE.settings, settings);
  applyTheme();
}

function applyTheme() {
  document.body.dataset.theme = settings.theme;
  document.body.dataset.mode = settings.darkMode ? "dark" : "light";
  document.querySelector('meta[name="theme-color"]').content = settings.darkMode ? "#151a17" : "#f5f4f0";
  const themeButton = $("#theme-quick-button");
  themeButton.textContent = settings.darkMode ? "☼" : "◐";
  themeButton.setAttribute("aria-label", settings.darkMode ? "Enable light mode" : "Enable dark mode");
  themeButton.title = settings.darkMode ? "Enable light mode" : "Enable dark mode";
  $("#dark-mode-toggle").checked = settings.darkMode;
  document.querySelectorAll(".theme-choice").forEach(button => {
    button.classList.toggle("selected", button.dataset.theme === settings.theme);
    button.setAttribute("aria-pressed", String(button.dataset.theme === settings.theme));
  });
}

function showScreen(name) {
  const titles = { home: "Home", game: "Play", history: "Game history", settings: "Settings" };
  for (const screen of ["home", "game", "history", "settings"]) {
    $(`#${screen}-screen`).classList.toggle("hidden", screen !== name);
  }
  $("#page-title").textContent = titles[name];
  document.querySelectorAll(".nav-link").forEach(button => {
    button.classList.toggle("active", button.dataset.screen === name);
  });
  if (name === "home") renderHome();
  if (name === "history") renderHistory();
}

function getHistory() {
  const saved = readStorage(STORAGE.history, []);
  return Array.isArray(saved) ? saved.filter(entry => entry && typeof entry.id === "string") : [];
}

function getActiveColor() {
  return game?.fen.split(" ")[1] || "w";
}

function getPosition(fen) {
  const rows = fen.split(" ")[0].split("/");
  const position = {};
  rows.forEach((row, rowIndex) => {
    let file = 0;
    for (const token of row) {
      if (/\d/.test(token)) {
        file += Number(token);
      } else {
        position[`${FILES[file]}${8 - rowIndex}`] = token;
        file += 1;
      }
    }
  });
  return position;
}

function getSquareColor(square) {
  const piece = getPosition(game.fen)[square];
  return piece && piece === piece.toUpperCase() ? "w" : "b";
}

function renderBoard() {
  if (!game) return;
  const position = getPosition(game.fen);
  const targets = new Set(legalMoves.filter(move => move.from === selectedSquare).map(move => move.to));
  const rows = flipped ? [1, 2, 3, 4, 5, 6, 7, 8] : [8, 7, 6, 5, 4, 3, 2, 1];
  const files = flipped ? [...FILES].reverse() : [...FILES];
  boardElement.replaceChildren();

  rows.forEach((rank, rowIndex) => {
    files.forEach((file, columnIndex) => {
      const square = `${file}${rank}`;
      const cell = document.createElement("button");
      cell.type = "button";
      cell.className = `square ${(rank + FILES.indexOf(file)) % 2 === 0 ? "dark" : "light"}`;
      cell.setAttribute("role", "gridcell");
      cell.setAttribute("aria-label", `${square}${position[square] ? `, ${pieceName(position[square])}` : ", empty"}`);
      cell.dataset.square = square;
      if (columnIndex === 0) {
        const rankLabel = document.createElement("span");
        rankLabel.className = "coordinate rank-coordinate";
        rankLabel.textContent = String(rank);
        cell.append(rankLabel);
      }
      if (rowIndex === rows.length - 1) {
        const fileLabel = document.createElement("span");
        fileLabel.className = "coordinate file-coordinate";
        fileLabel.textContent = file;
        cell.append(fileLabel);
      }
      if (square === selectedSquare) cell.classList.add("selected");
      if (game.lastMove?.includes(square)) cell.classList.add("last-move");
      if (square === game.checkSquare) cell.classList.add("in-check");
      if (targets.has(square)) cell.classList.add(position[square] ? "capture-target" : "legal-target");
      if (position[square]) {
        const piece = document.createElement("span");
        piece.className = `piece ${position[square] === position[square].toUpperCase() ? "white-piece" : "black-piece"}`;
        piece.textContent = PIECES[position[square]];
        cell.append(piece);
      }
      cell.addEventListener("click", () => handleSquareClick(square));
      boardElement.append(cell);
    });
  });
}

function pieceName(symbol) {
  const names = { k: "king", q: "queen", r: "rook", b: "bishop", n: "knight", p: "pawn" };
  return `${symbol === symbol.toUpperCase() ? "White" : "Black"} ${names[symbol.toLowerCase()]}`;
}

function renderPlayers() {
  if (!game) return;
  const aiMode = game.mode === "ai";
  const playerColor = game.playerColor || "w";
  const opponentColor = playerColor === "w" ? "b" : "w";
  $("#game-heading").textContent = aiMode ? "Play vs AI" : "Two-player game";
  $("#opponent-name").textContent = aiMode ? "Neural Network" : "Player";
  $("#opponent-caption").textContent = opponentColor === "w" ? "White" : "Black";
  $("#self-name").textContent = "You";
  $("#self-caption").textContent = playerColor === "w" ? "White" : "Black";
  $("#opponent-piece").textContent = opponentColor === "w" ? "♔" : "♚";
  $("#self-piece").textContent = playerColor === "w" ? "♔" : "♚";
  $("#opponent-strip").classList.toggle("active-player", getActiveColor() === opponentColor);
  $(".self-strip").classList.toggle("active-player", getActiveColor() === playerColor);
  $("#model-indicator").textContent = aiMode ? "AI" : "OFF";
  $("#evaluation-panel").classList.toggle("disabled-panel", !aiMode);
}

function renderMoves() {
  const list = $("#move-list");
  list.replaceChildren();
  $("#move-count").textContent = String(Math.ceil(game.moves.length / 2));
  if (!game.moves.length) {
    const empty = document.createElement("div");
    empty.className = "empty-moves";
    empty.textContent = "Your opening move starts here.";
    list.append(empty);
    return;
  }
  for (let index = 0; index < game.moves.length; index += 2) {
    const row = document.createElement("div");
    row.className = "move-row";
    const number = document.createElement("span");
    number.className = "move-number";
    number.textContent = `${Math.floor(index / 2) + 1}.`;
    row.append(number);
    for (const move of game.moves.slice(index, index + 2)) {
      const san = document.createElement("span");
      san.className = "move-san";
      san.textContent = move.san;
      row.append(san);
    }
    list.append(row);
  }
  list.scrollTop = list.scrollHeight;
}

function renderMaterial() {
  const takenBy = { w: [], b: [] };
  game.moves.forEach(move => {
    if (move.captured && move.capturedColor && takenBy[move.color]) {
      takenBy[move.color].push(move.captured);
    }
  });
  for (const color of ["w", "b"]) {
    const output = color === "w" ? $("#white-captured") : $("#black-captured");
    output.replaceChildren();
    takenBy[color].forEach(symbol => {
      const piece = document.createElement("span");
      piece.textContent = PIECES[symbol];
      piece.className = symbol === symbol.toUpperCase() ? "white-piece" : "black-piece";
      output.append(piece);
    });
    if (!takenBy[color].length) output.textContent = "—";
  }
}

function renderClocks() {
  if (!game) return;
  for (const color of ["w", "b"]) {
    const element = color === "w" ? $("#white-clock") : $("#black-clock");
    const remaining = Math.max(0, Math.ceil(game.clocks[color]));
    const minutes = Math.floor(remaining / 60);
    const seconds = remaining % 60;
    element.textContent = game.clockEnabled
      ? `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`
      : "--:--";
    element.classList.toggle("low-time", game.clockEnabled && remaining <= 30);
    element.classList.toggle("active-clock", game.clockEnabled && getActiveColor() === color && !game.finished);
  }
}

function renderStatus() {
  if (!game) return;
  const active = getActiveColor() === "w" ? "White" : "Black";
  const isAiTurn = game.mode === "ai" && getActiveColor() !== game.playerColor;
  const text = game.finished
    ? `${game.resultText}${game.termination ? ` · ${game.termination}` : ""}`
    : isAiTurn
      ? "Neural network is thinking…"
      : `${active} to move${game.checkSquare ? " · Check" : ""}`;
  $("#game-status").textContent = text;
  const waitingForBlackToMove = game.mode === "ai" &&
    game.playerColor === "b" &&
    game.moves.length === 1 &&
    game.snapshots.length === 1;
  $("#undo-button").disabled = game.finished || game.snapshots.length === 0 || waitingForBlackToMove || busy;
  $("#draw-button").disabled = game.finished || busy;
  $("#draw-button").lastChild.textContent = canClaimDraw() ? " Claim draw" : " Offer draw";
  $("#resign-button").disabled = game.finished || busy;
  $("#board-overlay").classList.toggle("hidden", !game.finished);
  if (game.finished) {
    $("#result-title").textContent = game.resultText;
    $("#result-detail").textContent = game.termination || "Game complete";
    $("#result-icon").textContent = game.result === "1/2-1/2" ? "½" : "♔";
  }
  renderClocks();
}

function renderGame() {
  if (!game) return;
  renderPlayers();
  renderBoard();
  renderMoves();
  renderMaterial();
  renderStatus();
}

function persistActiveGame() {
  if (game && !game.finished) writeStorage(STORAGE.activeGame, game);
}

async function refreshPosition() {
  const legal = await api("/legal-moves", { fen: game.fen });
  legalMoves = legal.moves;
  selectedSquare = null;
  renderGame();
  if (game.mode === "ai") {
    try {
      const evaluation = await api("/evaluate", { fen: game.fen });
      $("#score-value").textContent = evaluation.white_score.toFixed(3);
      $("#score-marker").style.left = `${((evaluation.white_score + 1) / 2) * 100}%`;
      $("#score-note").textContent = `Side to move: ${evaluation.side_to_move_score.toFixed(3)}.`;
    } catch (error) {
      $("#score-note").textContent = "Position score is temporarily unavailable.";
      throw error;
    }
  } else {
    $("#score-value").textContent = "—";
    $("#score-marker").style.left = "50%";
    $("#score-note").textContent = "Position score is available in games against the neural network.";
  }
  renderGame();
}

function storeGameResult() {
  if (game.saved) return;
  game.saved = true;
  const history = getHistory();
  history.unshift({
    id: game.id,
    mode: game.mode,
    playerColor: game.playerColor,
    result: game.result,
    resultText: game.resultText,
    termination: game.termination,
    moves: game.moves.map(move => ({ san: move.san, color: move.color })),
    startedAt: game.startedAt,
    endedAt: new Date().toISOString(),
    clockSeconds: game.initialClock
  });
  writeStorage(STORAGE.history, history.slice(0, 100));
  localStorage.removeItem(STORAGE.activeGame);
}

function finishGame(result, termination) {
  if (!game || game.finished) return;
  game.finished = true;
  game.result = result;
  game.termination = termination;
  game.resultText = result === "1/2-1/2"
    ? "Draw"
    : result === "1-0" ? "White wins" : "Black wins";
  selectedSquare = null;
  clearInterval(clockTimer);
  storeGameResult();
  renderGame();
}

function clockStep() {
  if (!game || game.finished || !game.clockEnabled || busy) {
    lastClockTick = Date.now();
    return;
  }
  const now = Date.now();
  const elapsed = (now - lastClockTick) / 1000;
  lastClockTick = now;
  const color = getActiveColor();
  game.clocks[color] = Math.max(0, game.clocks[color] - elapsed);
  renderClocks();
  if (game.clocks[color] <= 0) {
    finishGame(color === "w" ? "0-1" : "1-0", "Time expired");
  } else if (Date.now() - lastClockSave >= 5000) {
    lastClockSave = Date.now();
    persistActiveGame();
  }
}

function startClock() {
  clearInterval(clockTimer);
  lastClockTick = Date.now();
  lastClockSave = Date.now();
  if (game?.clockEnabled) clockTimer = setInterval(clockStep, 250);
}

function createGame(mode, color, clockMinutes) {
  const chosenColor = color === "random"
    ? (Math.random() < 0.5 ? "w" : "b")
    : color === "white" ? "w" : color === "black" ? "b" : color;
  const seconds = Number(clockMinutes) * 60;
  game = {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    mode,
    playerColor: mode === "ai" ? chosenColor : "w",
    fen: START_FEN,
    moves: [],
    snapshots: [],
    lastMove: null,
    checkSquare: null,
    clocks: { w: seconds, b: seconds },
    initialClock: seconds,
    clockEnabled: seconds > 0,
    startedAt: new Date().toISOString(),
    finished: false,
    saved: false
  };
  legalMoves = [];
  flipped = mode === "ai" && chosenColor === "b";
  showScreen("game");
  startClock();
  $("#game-error").textContent = "";
  runGameAction(async () => {
    await refreshPosition();
    persistActiveGame();
    await maybePlayAiTurn();
  });
}

async function startNewGame(mode, color, clockMinutes) {
  const saved = readStorage(STORAGE.activeGame, null);
  if (saved && !saved.finished) {
    const confirmed = await showConfirmation({
      eyebrow: "UNFINISHED GAME",
      title: "Start a new game?",
      message: "Your unfinished game will be replaced. Finished games in your history will stay saved.",
      actionLabel: "Start new game"
    });
    if (!confirmed) return;
  }
  createGame(mode, color, clockMinutes);
}

function snapshotGame() {
  game.snapshots.push({
    fen: game.fen,
    moves: game.moves.slice(),
    clocks: { ...game.clocks },
    lastMove: game.lastMove,
    checkSquare: game.checkSquare
  });
}

async function applyMove(uci) {
  const moverColor = getActiveColor();
  const matchingMove = legalMoves.find(move => move.uci === uci);
  if (!matchingMove) throw new Error("That move is no longer legal. Please select it again.");
  snapshotGame();
  const result = await api("/play", { fen: game.fen, uci });
  const capturedPiece = result.captured_piece || null;
  game.moves.push({
    uci,
    san: result.san,
    color: moverColor,
    captured: capturedPiece,
    capturedColor: capturedPiece ? result.captured_color : null,
    fen: result.fen
  });
  game.fen = result.fen;
  game.lastMove = [matchingMove.from, matchingMove.to];
  game.checkSquare = result.is_check ? findKingSquare(getPosition(game.fen), getActiveColor()) : null;
  selectedSquare = null;
  if (settings.sound) playMoveSound();
  if (result.game_over) {
    finishGame(result.result || "1/2-1/2", result.is_checkmate ? "Checkmate" : "Draw");
  } else if (countPositionOccurrences(game) >= 5) {
    finishGame("1/2-1/2", "Fivefold repetition");
  } else if (Number(game.fen.split(" ")[4]) >= 150) {
    finishGame("1/2-1/2", "75-move rule");
  }
  renderGame();
  persistActiveGame();
  if (!game.finished) {
    await refreshPosition();
    renderStatus();
  }
}

function findKingSquare(position, color) {
  const king = color === "w" ? "K" : "k";
  return Object.keys(position).find(square => position[square] === king) || null;
}

function countPositionOccurrences(currentGame) {
  const currentKey = currentGame.fen.split(" ").slice(0, 4).join(" ");
  const keys = [START_FEN, ...currentGame.moves.map(move => move.fen).filter(Boolean)];
  return keys.filter(key => key.split(" ").slice(0, 4).join(" ") === currentKey).length;
}

function canClaimDraw() {
  if (!game) return false;
  return Number(game.fen.split(" ")[4]) >= 100 || countPositionOccurrences(game) >= 3;
}

async function maybePlayAiTurn() {
  if (!game || game.finished || game.mode !== "ai" || getActiveColor() === game.playerColor) return;
  $("#game-status").textContent = "Neural network is thinking…";
  const gameId = game.id;
  const position = game.fen;
  await new Promise(resolve => setTimeout(resolve, 1000));
  if (!game || game.finished || game.id !== gameId || game.fen !== position) return;
  const move = await api("/best-move", { fen: position });
  if (!game || game.finished || game.id !== gameId || game.fen !== position) return;
  await applyMove(move.uci);
}

async function playSelectedMove(move) {
  await runGameAction(async () => {
    await applyMove(move.uci);
    await maybePlayAiTurn();
  });
}

async function handleSquareClick(square) {
  if (!game || game.finished || busy) return;
  if (game.mode === "ai" && getActiveColor() !== game.playerColor) return;
  const selectedMove = legalMoves.find(move => move.from === selectedSquare && move.to === square);
  if (selectedMove) {
    const candidates = legalMoves.filter(move => move.from === selectedSquare && move.to === square);
    if (candidates.some(move => move.promotion)) {
      const promotion = await choosePromotion(getSquareColor(selectedMove.from));
      if (!promotion) return;
      const chosen = candidates.find(move => move.promotion === promotion);
      if (chosen) await playSelectedMove(chosen);
    } else {
      await playSelectedMove(selectedMove);
    }
    return;
  }
  const piece = getPosition(game.fen)[square];
  const sideToMove = getActiveColor();
  selectedSquare = piece && getSquareColor(square) === sideToMove ? square : null;
  renderBoard();
}

function choosePromotion(color) {
  const dialog = $("#promotion-dialog");
  const options = $("#promotion-options");
  options.replaceChildren();
  return new Promise(resolve => {
    let resolved = false;
    const finish = value => {
      if (resolved) return;
      resolved = true;
      dialog.removeEventListener("close", onClose);
      resolve(value);
    };
    const onClose = () => finish(dialog.returnValue || null);
    dialog.addEventListener("close", onClose);
    for (const [symbol, name] of Object.entries(PROMOTION_PIECES)) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "promotion-choice";
      button.innerHTML = "";
      const piece = document.createElement("span");
      const letter = color === "w" ? symbol.toUpperCase() : symbol;
      piece.textContent = PIECES[letter];
      button.append(piece, document.createTextNode(name));
      button.addEventListener("click", () => {
        dialog.close(symbol);
        finish(symbol);
      });
      options.append(button);
    }
    dialog.showModal();
  });
}

async function runGameAction(action) {
  if (busy) return;
  busy = true;
  $("#game-error").textContent = "";
  renderStatus();
  try {
    await action();
  } catch (error) {
    $("#game-error").textContent = error.message || "The move could not be completed.";
  } finally {
    busy = false;
    lastClockTick = Date.now();
    renderStatus();
    renderBoard();
  }
}

async function undoMove() {
  if (!game || game.finished || !game.snapshots.length) return;
  await runGameAction(async () => {
    const plies = game.mode === "ai" && game.moves.length >= 2 ? 2 : 1;
    let snapshot = null;
    for (let index = 0; index < plies; index += 1) {
      snapshot = game.snapshots.pop();
      if (!snapshot) break;
    }
    if (!snapshot) return;
    game.fen = snapshot.fen;
    game.moves = snapshot.moves;
    game.clocks = snapshot.clocks;
    game.lastMove = snapshot.lastMove;
    game.checkSquare = snapshot.checkSquare;
    selectedSquare = null;
    await refreshPosition();
    persistActiveGame();
  });
}

function showConfirmation({ eyebrow, title, message, actionLabel, danger = true }) {
  const dialog = $("#confirm-dialog");
  $("#confirm-eyebrow").textContent = eyebrow;
  $("#confirm-title").textContent = title;
  $("#confirm-message").textContent = message;
  const actionButton = $("#confirm-action-button");
  actionButton.textContent = actionLabel;
  actionButton.classList.toggle("danger-button", danger);
  actionButton.classList.toggle("primary-button", !danger);
  dialog.showModal();
  return new Promise(resolve => {
    dialog.addEventListener("close", () => resolve(dialog.returnValue === "confirm"), { once: true });
  });
}

async function resignGame() {
  if (!game || game.finished) return;
  const confirmed = await showConfirmation({
    eyebrow: "END GAME",
    title: "Resign this game?",
    message: "Your opponent will be recorded as the winner.",
    actionLabel: "Resign"
  });
  if (!confirmed) return;
  const winner = getActiveColor() === "w" ? "0-1" : "1-0";
  finishGame(winner, "Resignation");
}

async function offerDraw() {
  if (!game || game.finished) return;
  if (canClaimDraw()) {
    const reason = Number(game.fen.split(" ")[4]) >= 100
      ? "50-move rule"
      : "Threefold repetition";
    const confirmedClaim = await showConfirmation({
      eyebrow: "DRAW CLAIM",
      title: `Claim draw by ${reason.toLowerCase()}?`,
      message: `The current position qualifies for a draw by the ${reason}.`,
      actionLabel: "Claim draw",
      danger: false
    });
    if (confirmedClaim) finishGame("1/2-1/2", reason);
    return;
  }
  const confirmed = await showConfirmation({
    eyebrow: "DRAW",
    title: "Agree to a draw?",
    message: "This local game will be recorded as a draw.",
    actionLabel: "Agree to draw",
    danger: false
  });
  if (confirmed) finishGame("1/2-1/2", "Draw agreed");
}

async function maybeResumeGame() {
  const saved = readStorage(STORAGE.activeGame, null);
  if (!saved || saved.finished || typeof saved.fen !== "string" || !Array.isArray(saved.moves)) return false;
  const confirmed = await showConfirmation({
    eyebrow: "GAME IN PROGRESS",
    title: "Continue your game?",
    message: "An unfinished game was saved in this browser.",
    actionLabel: "Continue",
    danger: false
  });
  if (!confirmed) {
    localStorage.removeItem(STORAGE.activeGame);
    return false;
  }
  game = saved;
  game.snapshots = Array.isArray(game.snapshots) ? game.snapshots : [];
  game.clocks = game.clocks || { ...INITIAL_CLOCKS };
  game.playerColor = game.playerColor || "w";
  game.lastMove = game.lastMove || null;
  game.checkSquare = game.checkSquare || null;
  flipped = game.mode === "ai" && game.playerColor === "b";
  showScreen("game");
  startClock();
  await runGameAction(async () => {
    await refreshPosition();
    await maybePlayAiTurn();
  });
  return true;
}

function formatDate(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? "Unknown date"
    : new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", year: "numeric" }).format(date);
}

function formatDuration(startedAt, endedAt) {
  const seconds = Math.max(0, Math.floor((new Date(endedAt) - new Date(startedAt)) / 1000));
  if (!Number.isFinite(seconds)) return "";
  return `${Math.floor(seconds / 60)} min`;
}

function createHistoryRow(entry, detailed = false) {
  const row = document.createElement("article");
  row.className = "history-row";
  const gameCell = document.createElement("div");
  gameCell.className = "history-game-cell";
  const icon = document.createElement("span");
  icon.className = `history-mode-icon ${entry.mode === "ai" ? "history-ai" : "history-friend"}`;
  icon.textContent = entry.mode === "ai" ? "✳" : "♟";
  const details = document.createElement("span");
  details.className = "history-game-name";
  const name = document.createElement("strong");
  name.textContent = entry.mode === "ai" ? "vs Neural Network" : "Two-player game";
  const sub = document.createElement("small");
  sub.textContent = `${entry.moves?.length || 0} moves · ${formatDuration(entry.startedAt, entry.endedAt)}`;
  details.append(name, sub);
  gameCell.append(icon, details);

  const result = document.createElement("span");
  result.className = `history-result ${entry.result === "1/2-1/2" ? "draw-result" : entry.result === "1-0" ? "white-result" : "black-result"}`;
  result.textContent = entry.resultText || "Game complete";
  const date = document.createElement("span");
  date.className = "history-date";
  date.textContent = formatDate(entry.endedAt);
  row.append(gameCell, result, date);

  if (detailed) {
    const detailsElement = document.createElement("details");
    detailsElement.className = "history-moves";
    const summary = document.createElement("summary");
    summary.textContent = "Moves";
    const moves = document.createElement("p");
    moves.textContent = (entry.moves || []).map((move, index) =>
      index % 2 === 0
        ? `${Math.floor(index / 2) + 1}. ${move.san}`
        : move.san
    ).join("  ");
    detailsElement.append(summary, moves);
    const moveCell = document.createElement("div");
    moveCell.className = "history-detail-cell";
    moveCell.append(detailsElement);
    row.append(moveCell);
  }
  return row;
}

function renderHistory() {
  const history = getHistory();
  const list = $("#history-list");
  list.replaceChildren();
  $("#history-empty").classList.toggle("hidden", history.length > 0);
  $(".history-panel").classList.toggle("hidden", history.length === 0);
  const whiteWins = history.filter(entry => entry.result === "1-0").length;
  const draws = history.filter(entry => entry.result === "1/2-1/2").length;
  const stats = [
    ["Games played", history.length],
    ["White wins", whiteWins],
    ["Draws", draws]
  ];
  const summary = $("#history-summary");
  summary.replaceChildren();
  stats.forEach(([label, value]) => {
    const card = document.createElement("div");
    card.className = "stat-card";
    const strong = document.createElement("strong");
    strong.textContent = String(value);
    const caption = document.createElement("span");
    caption.textContent = label;
    card.append(strong, caption);
    summary.append(card);
  });
  history.forEach(entry => list.append(createHistoryRow(entry, true)));
}

function renderHome() {
  const history = getHistory();
  const list = $("#recent-games");
  list.replaceChildren();
  if (!history.length) {
    const empty = document.createElement("div");
    empty.className = "recent-empty";
    empty.innerHTML = "<span class=\"empty-board-icon\">♘</span><span><strong>No games yet</strong><small>Choose a game above to get started.</small></span>";
    list.append(empty);
  } else {
    history.slice(0, 3).forEach(entry => list.append(createHistoryRow(entry)));
  }
  const active = readStorage(STORAGE.activeGame, null);
  const existing = $("#continue-card");
  if (existing) existing.remove();
  if (active && !active.finished) {
    const continueCard = document.createElement("button");
    continueCard.id = "continue-card";
    continueCard.className = "continue-card";
    continueCard.type = "button";
    continueCard.innerHTML = "<span>↗</span><span><strong>Continue your game</strong><small>Your unfinished match is saved on this device.</small></span>";
    continueCard.addEventListener("click", async () => {
      game = active;
      game.snapshots = Array.isArray(game.snapshots) ? game.snapshots : [];
      game.clocks = game.clocks || { ...INITIAL_CLOCKS };
      flipped = game.mode === "ai" && game.playerColor === "b";
      showScreen("game");
      startClock();
      await runGameAction(async () => {
        await refreshPosition();
        await maybePlayAiTurn();
      });
    });
    $(".welcome-row").after(continueCard);
  }
}

function renderConnection(connected) {
  apiConnected = connected;
}

function playMoveSound() {
  try {
    const context = new AudioContext();
    const oscillator = context.createOscillator();
    const gain = context.createGain();
    oscillator.frequency.value = 520;
    gain.gain.setValueAtTime(0.035, context.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, context.currentTime + 0.08);
    oscillator.connect(gain);
    gain.connect(context.destination);
    oscillator.start();
    oscillator.stop(context.currentTime + 0.08);
    oscillator.addEventListener("ended", () => context.close(), { once: true });
  } catch (error) {
    console.warn("Move sound could not be played.", error);
  }
}

function initializeSettings() {
  applyTheme();
  $("#sound-toggle").checked = settings.sound;
}

document.querySelectorAll("[data-screen]").forEach(button => {
  button.addEventListener("click", event => {
    event.preventDefault();
    showScreen(button.dataset.screen);
  });
});

document.querySelectorAll("[data-color]").forEach(button => {
  button.addEventListener("click", () => {
    playerColorChoice = button.dataset.color;
    document.querySelectorAll("[data-color]").forEach(choice => {
      choice.classList.toggle("selected", choice === button);
      choice.setAttribute("aria-pressed", String(choice === button));
    });
  });
});

$("#start-ai-button").addEventListener("click", () => {
  if (!apiConnected) {
    $("#home-error").textContent = "Start the FastAPI backend before starting a game.";
    return;
  }
  $("#home-error").textContent = "";
  startNewGame("ai", playerColorChoice, $("#ai-time-control").value);
});

$("#start-friend-button").addEventListener("click", () => {
  if (!apiConnected) {
    $("#home-error").textContent = "Start the FastAPI backend before starting a game.";
    return;
  }
  $("#home-error").textContent = "";
  startNewGame("friend", "w", $("#friend-time-control").value);
});

$("#flip-board-button").addEventListener("click", () => {
  flipped = !flipped;
  renderBoard();
});
$("#new-game-button").addEventListener("click", () => showScreen("home"));
$("#undo-button").addEventListener("click", undoMove);
$("#resign-button").addEventListener("click", resignGame);
$("#draw-button").addEventListener("click", offerDraw);
$("#result-home-button").addEventListener("click", () => showScreen("home"));
$("#theme-quick-button").addEventListener("click", () => {
  settings.darkMode = !settings.darkMode;
  saveSettings();
});

document.querySelectorAll(".theme-choice").forEach(button => {
  button.addEventListener("click", () => {
    settings.theme = button.dataset.theme;
    saveSettings();
  });
});

$("#sound-toggle").addEventListener("change", event => {
  settings.sound = event.target.checked;
  saveSettings();
});
$("#dark-mode-toggle").addEventListener("change", event => {
  settings.darkMode = event.target.checked;
  saveSettings();
});

$("#clear-history-button").addEventListener("click", async () => {
  const confirmed = await showConfirmation({
    eyebrow: "CLEAR RECORD",
    title: "Clear your game history?",
    message: "This removes all finished game records stored in this browser. This cannot be undone.",
    actionLabel: "Clear history"
  });
  if (confirmed) {
    localStorage.removeItem(STORAGE.history);
    renderHistory();
    renderHome();
  }
});

$("#confirm-dialog").addEventListener("click", event => {
  if (event.target === $("#confirm-dialog")) $("#confirm-dialog").close("cancel");
});
$("#promotion-dialog").addEventListener("click", event => {
  if (event.target === $("#promotion-dialog")) $("#promotion-dialog").close("cancel");
});

function showError(message) {
  const target = $("#game-screen").classList.contains("hidden") ? $("#home-error") : $("#game-error");
  target.textContent = message;
}

async function initialize() {
  initializeSettings();
  renderHome();
  try {
    await api("/health");
    renderConnection(true);
  } catch {
    renderConnection(false);
    $("#home-error").textContent = "Chess server is offline. Start FastAPI to play.";
  }
  if (apiConnected) await maybeResumeGame();
}

initialize();
