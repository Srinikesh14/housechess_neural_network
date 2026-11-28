# Chess ANN Lab

An educational chess project built with Python, NumPy, FastAPI, and python-chess. It includes a Stockfish-supervised PGN training pipeline and a browser chess application.

> **Model status:** V1 is the preserved original outcome-supervised model. V2 is the separately saved Stockfish-supervised model trained on a 400,000-game subset. Neither model should be described as stronger until held-out evaluation and playing benchmarks support that conclusion.

## Project layout

- `backend/encoder.py` — V2 838-feature encoder and legacy 773-feature V1 encoder.
- `backend/mlp.py` — NumPy forward pass, mean squared error, backpropagation, gradient descent, and version-aware model saves.
- `backend/gradient_check.py` — finite-difference gradient validation.
- `backend/stockfish_eval.py` — reusable UCI engine integration, White-perspective normalization, and settings-aware SQLite cache.
- `backend/train.py` — streaming PGN training, validation, atomic checkpointing, and resume support.
- `backend/evaluate.py` — V1/V2 regression comparison and optional paired playing benchmarks.
- `backend/wsp.py` — guarded Weight-Space Programming experiments via OpenRouter; LLM tools are local, bounded, and do not execute generated code.
- `backend/wsp_system_prompt.txt` — research and safety instructions used for WSP sessions.
- `backend/main.py` — FastAPI health, evaluation, legal-move, play, and best-move endpoints.
- `backend/models/` — preserved V1 model/checkpoint and separate V2 outputs/cache.
- `frontend/` — local browser chess application.

The browser app supports AI or local two-player games, clocks, undo, draw/resign, move history, themes, and saved results in browser-local storage. The AI move selector is intentionally a simple one-ply evaluator, not minimax or alpha-beta search.

The UI lets you choose White, Black, or a random color against the AI, or start a local two-player game. It supports no clock or 5/10/15-minute clocks, legal-move highlighting, promotion choice, undo, draw/resign, board flipping, captured pieces, check status, and move notation. Finished games, theme, and sound preferences are stored in browser-local storage. This is local play, not online multiplayer or account synchronization.

On the Play vs AI card, select either the preserved **V1 Original** model or the **V2 Stockfish-trained** model. The selection is sent with each evaluation and best-move request, so both versions remain loaded and available in the same backend process; it is saved with the active game and its history entry. The **INFO** button describes their architectures and input representations. V2 is not guaranteed to play stronger than V1.

## Install and run the application

From the `chess-ann-project` directory, create/activate the environment and install dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Start the API in one terminal:

```powershell
python -m uvicorn backend.main:app --reload
```

In a second terminal, from the same directory:

```powershell
python -m http.server 5500 --directory frontend
```

Open <http://127.0.0.1:5500>. The API docs are at <http://127.0.0.1:8000/docs>.

## Network and input representation

The architecture is unchanged aside from the V2 input width:

```text
V1: 773 -> 256 (ReLU) -> 64 (ReLU) -> 1 (tanh)
V2: 838 -> 256 (ReLU) -> 64 (ReLU) -> 1 (tanh)
```

All outputs are in `[-1, 1]` and interpreted from White's perspective. V2 input ordering:

1. `0–767`: 12 piece planes × 64 squares, White pawn through king then Black pawn through king; python-chess square order (`a1=0`, `h8=63`).
2. `768`: side to move (`1` for White).
3. `769–772`: White king-side, White queen-side, Black king-side, Black queen-side castling rights.
4. `773–836`: one-hot en-passant target square, all zero if no target exists.
5. `837`: halfmove clock clipped to 100 and divided by 100.

The encoder does not represent repetition history. A FEN does not contain complete history; the cache key distinguishes available PGN move history to avoid conflating repeated-position contexts.

V1's saved 773-input weights remain unchanged and use the matching legacy encoder. The V1 and V2 first-layer weights are never mixed or migrated.

## Stockfish position labels

The trainer samples at most `--samples-per-game` positions from each game. It obtains a Stockfish score for each position, converts the evaluation explicitly to White's perspective, and uses:

```text
centipawn target = tanh(white_score_cp / K)    # K defaults to 400
```

Mate evaluations are not treated as centipawns. Forced mates map to signed targets from 0.9 to 1.0; closer mates have larger magnitude. The score, type, White perspective, target, normalization scale, engine identity, and search settings are persisted in SQLite.

One CPU Stockfish process is reused during a run. Cache entries are only reused for matching engine/search settings (depth/time, threads, and hash). If `K` changes, the cached raw score is reused and its normalized target is recalculated. PGNs are streamed, including `.pgn.zst`; only one game's sampled positions are held at once. Every fifth parsed game is held out as validation, and validation never updates weights. Training MSE is the mean online per-position SGD loss; validation MSE is measured on the final epoch weights after the training pass.

## Install Stockfish

Install a native Stockfish executable for Windows separately. `python-chess` already provides the Python UCI engine integration; no GPU or additional engine package is required.

```powershell
python -m pip install -r requirements.txt
$env:STOCKFISH_EXECUTABLE = "C:\path\to\stockfish\stockfish-windows-x86-64-avx2.exe"
```

Replace that example with your actual Stockfish executable path. The default configuration uses depth 8, one thread, and 64 MB of hash. Optionally add `--time-limit 0.1`; when depth and time are both specified, the first reached limit ends the search.

## Start training on the supplied PGN

Run from the `chess-ann-project` directory. The supplied PGN is one directory above it:

```powershell
python -m backend.train --pgn ..\lichess_db_standard_rated_2016-11.pgn.zst --stockfish "C:\path\to\stockfish\stockfish-windows-x86-64-avx2.exe" --max-games 1000 --max-positions 4000 --samples-per-game 4 --depth 8 --threads 1 --hash-mb 64
```

This writes:

- `backend/models/model_v2_stockfish.npz`
- `backend/models/training_checkpoint_v2.npz`
- `backend/models/stockfish_cache.sqlite3`

`--max-positions` is applied between games, so the final game can exceed the requested cap by up to one game's sample count. Invalid PGN games are skipped; incomplete but legal games can still contribute positions. Engine failures are reported and affected positions are skipped.

To resume after an interruption or expand the subset, keep the same PGN, Stockfish executable/version, evaluation settings, seed, samples per game, and learning rate; increase the limits and add `--resume`. If the checkpoint is already in a later epoch, pass `--epochs` with at least that epoch number:

```powershell
python -m backend.train --pgn ..\lichess_db_standard_rated_2016-11.pgn.zst --stockfish "C:\path\to\stockfish\stockfish-windows-x86-64-avx2.exe" --max-games 10000 --max-positions 40000 --samples-per-game 4 --depth 8 --threads 1 --hash-mb 64 --resume
```

Checkpoints are saved atomically every 1,000 games (configurable with `--checkpoint-interval`); an interruption may require replaying work since the last checkpoint. The SQLite cache is persistent independently of model checkpoints. `--overwrite` explicitly starts over at the selected V2 paths; it cannot overwrite V1 artifacts. Never use `backend/training_checkpoint.npz` to resume V2.

To train the full archive, omit both `--max-games` and `--max-positions`. Increase `--epochs` as desired. A full run can take a long time because Stockfish evaluates each previously unseen sampled position; start with the small subset first. The cache may require substantial disk space as it grows.

## Evaluate V1 against V2

Regression evaluation uses the same held-out sampled positions for both models and reports MSE, MAE, prediction correlation, prediction consistency, runtime per position, and number of positions. Optional playing benchmarks use the standard starting position, alternate colors, and report wins/losses/draws, points-based win rate, and average game length. Self-play games are paired with colors swapped.

```powershell
python -m backend.evaluate --pgn ..\lichess_db_standard_rated_2016-11.pgn.zst --stockfish "C:\path\to\stockfish\stockfish-windows-x86-64-avx2.exe" --max-games 10000 --max-positions 1000 --samples-per-game 4 --depth 8 --self-play-pairs 2 --stockfish-games 2
```

The V2 model file must exist before running the comparison. A lower training loss alone is not evidence that V2 improved.

## Select the model in the API

The API keeps V1 as its default for older clients, and also loads V2 when its checkpoint is available. The frontend explicitly requests its selected version. To change the default used by API clients that omit `model_version`:

```powershell
$env:CHESS_ANN_MODEL_VERSION = "v2"
python -m uvicorn backend.main:app --reload
```

Set `CHESS_ANN_MODEL_VERSION` to `v1` or `v2` for the default model used by API clients that omit `model_version`. `POST /api/evaluate` and `POST /api/best-move` accept an optional `model_version` of `v1` or `v2`. The health endpoint lists loaded models. `wsp` remains available only when explicitly configured to an accepted-model path.

## Experimental LLM Weight-Space Programming

The WSP experiment is deliberately separate from training. It uses the exact configured OpenRouter model identifier and does not silently substitute another model. Current availability, free-tier quota, and this model's response-format reliability are not guaranteed; no request is sent until you run `backend.wsp analyze`. The CLI uses a constrained JSON action protocol rather than trusting model-generated Python or unrestricted function calls.

The LLM receives bounded summaries and tool results, not full weight matrices or the complete PGN. It can inspect architecture/statistics, neuron weights and activations, and prediction errors. In proposal mode it can record one small edit (maximum absolute per-weight change 0.02 and at most 64 coordinates), but it cannot apply it. Python validates the proposal and only writes to a separate candidate after you explicitly pass `--approve-candidate-edit`. No backpropagation or optimizer is available.

When you start `analyze`, the system prompt, weight summaries, and any specifically requested position FENs/Stockfish targets are sent to OpenRouter. Use only data you are authorized to send; do not use private PGNs if that sharing is not permitted.

### Configure the OpenRouter key

From `chess-ann-project`, create the local ignored `.env` file and enter your own key:

```powershell
Copy-Item .env.example .env
notepad .env
```

Keep the exact model identifier unless you intentionally change it:

```text
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
```

Do not share or commit `.env`. Alternatively, set `OPENROUTER_API_KEY` in the current PowerShell session. The model's current availability is checked only when the request is made. HTTP errors (including unavailable-model or quota errors) stop the experiment; there is no automatic model fallback.

### Create fixed, game-disjoint datasets

The model's first 400,000 games have already been used for training/validation, so WSP excludes them. Eligible PGN games after game 400,000 through 410,000 form development data; later games form the sealed set. Only every fifth parsed game without PGN errors is sampled. Each split is capped at 2,000 positions by default, sampled deterministically, and labeled using the existing Stockfish cache/settings. The manifest records hashes and split boundaries. Prepare the sealed set now, but do not inspect it during LLM analysis or candidate iteration. If the PGN has too few games beyond game 410,000, use a later archive or deliberately choose new boundaries beyond every game previously used for training/validation.

```powershell
python -m backend.wsp prepare-data --pgn ..\lichess_db_standard_rated_2016-11.pgn.zst --stockfish ..\stockfish\stockfish-windows-x86-64-universal.exe --max-games 420000 --development-after-game 400000 --sealed-after-game 410000 --max-positions-per-split 2000 --samples-per-game 4 --depth 8 --threads 1 --hash-mb 64
```

Existing data is not overwritten unless `--overwrite-data` is explicitly supplied. This step may use the cache and may perform additional CPU Stockfish evaluations.

### Start with read-only analysis

This is the recommended first experiment; it does not permit edit proposals:

```powershell
python -m backend.wsp analyze --model backend\models\model_v2_stockfish.npz --dataset backend\wsp_data\development.jsonl --max-steps 8 --question "Investigate whether the largest prediction errors share an activation pattern; report evidence and uncertainty without proposing edits."
```

To allow the LLM to record one *proposal* after reviewing its report, use a separate run:

```powershell
python -m backend.wsp analyze --model backend\models\model_v2_stockfish.npz --dataset backend\wsp_data\development.jsonl --max-steps 10 --propose-edit --question "Investigate one systematic error pattern and, only if the local evidence supports it, propose one small direct weight edit."
```

Runs, bounded tool logs, token usage, and any unapplied proposal are saved under `backend\wsp_experiments\<run-id>\`. Read `experiment.json` and `proposal.json` before deciding whether to test an edit. The proposal is tied to the exact baseline model hash.

### Test, evaluate, and optionally accept a candidate

Replace `<run-id>` and the output candidate name with the paths for your experiment:

```powershell
python -m backend.wsp apply-proposal --proposal backend\wsp_experiments\<run-id>\proposal.json --baseline backend\models\model_v2_stockfish.npz --dataset backend\wsp_data\development.jsonl --candidate backend\wsp_experiments\<run-id>\candidate.npz --approve-candidate-edit
python -m backend.wsp evaluate --baseline backend\models\model_v2_stockfish.npz --candidate backend\wsp_experiments\<run-id>\candidate.npz --dataset backend\wsp_data\development.jsonl --split development --output backend\wsp_experiments\<run-id>\development-report.json
```

If development results justify one final test, evaluate the candidate on the sealed set **once**. Do not feed sealed results back to the LLM or use them to choose among many candidates:

```powershell
python -m backend.wsp evaluate --baseline backend\models\model_v2_stockfish.npz --candidate backend\wsp_experiments\<run-id>\candidate.npz --dataset backend\wsp_data\sealed.jsonl --split sealed --output backend\wsp_experiments\<run-id>\sealed-report.json
```

Compare playing strength separately with paired, color-swapped games; this may take substantially longer:

```powershell
python -m backend.wsp benchmark --baseline backend\models\model_v2_stockfish.npz --candidate backend\wsp_experiments\<run-id>\candidate.npz --self-play-pairs 2 --stockfish-games 2 --stockfish ..\stockfish\stockfish-windows-x86-64-universal.exe --depth 8 --output backend\wsp_experiments\<run-id>\playing-report.json
```

Only after reviewing the sealed metrics and the separate playing report—and if the sealed report has at least 100 positions, lower MSE, no more than 2% MAE regression, and no more than 0.01 correlation regression—may you explicitly save a versioned accepted copy:

```powershell
python -m backend.wsp accept --report backend\wsp_experiments\<run-id>\sealed-report.json --output backend\models\wsp_accepted\model_wsp_<run-id>.npz --confirm-accept
```

Acceptance never overwrites V1 or V2. To serve an accepted model, explicitly select its path:

```powershell
$env:CHESS_ANN_MODEL_VERSION = "wsp"
$env:CHESS_ANN_WSP_MODEL_PATH = "backend\models\wsp_accepted\model_wsp_<run-id>.npz"
python -m uvicorn backend.main:app --reload
```

WSP results are exploratory. Lower Stockfish regression error alone does not demonstrate stronger chess play; the LLM may misunderstand the network or propose harmful edits, and a free API may be unavailable or rate-limited.

## Tests

```powershell
python -m unittest backend.test_training_pipeline backend.test_wsp -v
python -m backend.gradient_check
```

The focused tests use synthetic PGN games and a fake evaluator; they do not need Stockfish or the large archive. Stockfish uses CPU and the configured hash memory; NumPy training is CPU-only. Check free disk space before scaling up the persistent evaluation cache.

## API overview

- `GET /api/health`
- `POST /api/evaluate` with `{"fen": "..."}`
- `POST /api/legal-moves` with `{"fen": "..."}`
- `POST /api/play` with `{"fen": "...", "uci": "e2e4"}`
- `POST /api/best-move` with `{"fen": "..."}`
