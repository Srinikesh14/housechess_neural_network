# Chess ANN Lab

A compact educational chess project built from scratch with Python, NumPy, FastAPI, and python-chess. It encodes a board into 773 features, sends them through a three-layer MLP, and provides a responsive browser chess application for playing locally.

> **Important:** The included model contains partial outcome-supervised training weights and is not a strong or benchmarked chess engine. Training progress is saved separately in `backend/training_checkpoint.npz`; the PGN archive used for training is not included in this repository.

## Project layout

- `backend/encoder.py` — maps each piece and the five board metadata bits into a 773-value vector.
- `backend/mlp.py` — NumPy implementation of the forward pass, mean squared error, backpropagation, and a single gradient-descent step.
- `backend/gradient_check.py` — compares analytical gradients with finite-difference estimates.
- `backend/train.py` — streams PGN games, samples outcome-labeled positions, trains, and saves the model.
- `backend/main.py` — FastAPI health, evaluation, legal-move, play, and best-move endpoints.
- `frontend/` — static chess application with human-vs-AI and local two-player modes, clocks, move history, saved results, and board themes.

## Chess application features

- Start an AI game as White, Black, or a random color, or play a local two-player game.
- Choose no clock, 5, 10, or 15 minutes. A timed game resumes its clock when continued in the same browser.
- Make moves by selecting a piece and a highlighted legal destination. Promotions prompt for a piece.
- Undo moves, offer/agree to a draw, resign, and flip the board.
- See move notation, captured pieces, check status, and the experimental model's position score in AI games.
- Finished-game results and move lists are kept in browser-local storage; theme and sound preferences are saved there too.
- Choose Classic, Slate, or Blue board colors in Settings. The game is local to this browser and does not provide online multiplayer or account synchronization.
- Toggle dark mode from the top-right appearance button or Settings. The appearance preference persists on this device. The AI waits one second before each reply to make its turn feel deliberate.

The AI uses the saved NumPy model and the current one-ply move selector. Its playing strength has not been benchmarked; legal moves and polished game controls do not imply strong chess play.

## Setup and run

From the `chess-ann-project` directory, create and activate a virtual environment, then install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Start the API in one terminal:

```powershell
python -m uvicorn backend.main:app --reload
```

In a second terminal, from the same project directory, serve the frontend:

```powershell
python -m http.server 5500 --directory frontend
```

Open <http://127.0.0.1:5500>. The API docs are at <http://127.0.0.1:8000/docs>. For the board UI, use the New game button, click a piece and a highlighted legal destination, or try AI move. FEN can be loaded in the Position FEN disclosure.

## Representation and network

The first 768 inputs are 12 planes of 64 squares: one plane per color and piece type. Squares follow python-chess indexing (`a1` is 0 and `h8` is 63). The final five inputs are side-to-move and four castling-right bits. En-passant, move counters, and repetition history are not encoded.

The network computes:

```text
773 -> 256 (ReLU) -> 64 (ReLU) -> 1 (tanh)
```

Each dense layer multiplies a vector by a weight matrix and adds a bias. For the input-to-hidden layer, the board's mostly-zero piece features mean the multiplication sums only the active square columns for each hidden unit. Backpropagation applies the chain rule from the scalar squared error back through tanh, both ReLUs, and each matrix multiplication.

The output is a scalar in `[-1, 1]`, interpreted from White's perspective. The API also returns a score from the side-to-move perspective. Without training, these are just outputs of random weights.

## Try the gradient check

From the project directory:

```powershell
python -m backend.gradient_check
```

It uses a small network to keep numerical finite differences fast while testing the same activation and backpropagation code. It should report `Gradient check passed.`

## API overview

- `GET /api/health`
- `POST /api/evaluate` with `{"fen": "..."}`
- `POST /api/legal-moves` with `{"fen": "..."}`
- `POST /api/play` with `{"fen": "...", "uci": "e2e4"}`
- `POST /api/best-move` with `{"fen": "..."}`

## Learning and training next steps

1. Collect PGN games you have permission to use, parse them with python-chess, and encode positions with `encode_board`.
2. Create labels on a consistent scale. For example, game outcomes can be `1` for a White win, `0` for a draw, and `-1` for a White loss. A position-level teacher evaluation can provide more nuanced labels.
3. Split positions by game into training and validation sets so positions from one game do not leak across both sets.
4. Train using `MLP.train_step(features, target, learning_rate)` and track held-out mean squared error. The current example trains one position at a time; batching is a useful extension.
5. Add model save/load and expose a carefully validated training or model-management workflow before using learned weights in the API.
6. Compare predictions with held-out labels and a baseline before interpreting scores as useful chess evaluations.
## Train from a PGN file

The training script reads plain PGN files or streams `.pgn.zst` files without extracting them. By default it reads the entire Lichess archive beside this project, processes one epoch, and samples up to four positions per game to keep raw NumPy training practical. It does not keep the archive or all positions in memory. Set `--pgn` to another path if needed.

From the project directory, train on the complete archive:

```powershell
python -m backend.train --pgn ..\lichess_db_standard_rated_2016-11.pgn.zst
```

The script uses the final game result as a coarse White-perspective target (`1`, `0`, or `-1`), and holds every fifth parsed game out for validation. It saves weights to `backend/model.npz` after a successful pass. For a quick test, add `--max-games 1000 --max-positions 4000`; for more positions from each game, adjust `--samples-per-game`. Increase `--epochs` to re-read and train on the archive multiple times. The FastAPI app loads the saved file when it starts; restart the server after training to use updated weights.

Training checkpoints are atomically saved every 10,000 games by default. Progress is printed every 50,000 games. If interrupted, rerun with `--resume` to load the checkpoint and continue after its last completed game; if there is no checkpoint yet, `--resume` starts from the existing `backend/model.npz` weights and scans the archive from the beginning. An interrupted run loses at most the work since its last checkpoint. A bounded run using `--max-games` or `--max-positions` is intentionally left resumable rather than marked complete.

For example:

```powershell
python -m backend.train --pgn ..\lichess_db_standard_rated_2016-11.pgn.zst --resume
```

Game results are noisy labels for intermediate positions: a position may be good even if its player later loses. Validation MSE measures agreement with game outcomes, not playing strength. Split by whole games, inspect the label balance, and compare with a meaningful baseline before drawing conclusions. Check the dataset's source and permitted use for your intended purpose.

7. As a later phase, implement the same encoder and model in PyTorch, compare outputs and gradients, then use batching and GPU acceleration if the data size justifies them.
