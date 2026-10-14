# Setup — see the demo, then run the agent from the terminal

Everything here runs locally against in-memory fakes. No API keys, no model, no
network. A UI comes later; for now the agent is driven from the terminal.

---

## Just want to see it work? (one click)

**Windows:** double-click **`run_demo.bat`**.
**macOS / Linux:** open a terminal in this folder and run **`./run_demo.sh`**.

The launcher does everything on its own — creates the virtual environment, installs
the project, generates the data the demo needs, and runs the narrated walkthrough.
First run takes about a minute; later runs start immediately. The only requirement
is Python 3.11+ installed and on PATH ([python.org](https://python.org)).

The demo has six sections and runs in a few seconds:

| # | Section | What you will see |
|---|---|---|
| 1 | The FR5 safety gate | Code tries to delete a calendar event without asking — and is blocked |
| 2 | Scenario suite | 66 scenarios across 10 failure families, all passing, 0 safety violations |
| 3 | NLU pipeline | How spoken commands are understood; held-out accuracy; refusing to guess dates |
| 4 | Sequence diagram, end to end | A real cancellation email → agent asks permission → user says yes → event deleted |
| 5 | What the agent refuses | Promo emails, stray "yes", ambiguous bookings, unreadable mail |
| 6 | What the agent does | Voice booking, duplicate emails, reschedule, status questions |

To run just one section: `run_demo.bat --only 4` (or `./run_demo.sh --only 4`).

For what the project is and why it is built this way, read **`PROJECT_SUMMARY.md`**.

---

## Manual setup (for development)

### 0. Prerequisites

- Python **3.11 or newer** (`python --version`)
- Git

### 1. Clone and create the virtual environment

```powershell
# Windows (PowerShell)
git clone <this-repo>
cd booking-sync
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
git clone <this-repo>
cd booking-sync
python -m venv .venv
source .venv/bin/activate
```

If PowerShell refuses to run `Activate.ps1`:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 2. Install

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### 3. Verify the install

```bash
python -m pytest tests/ -q          # 797 tests, all should pass
```

### 4. Run the agent

There is no interactive shell yet. The agent is exercised through the narrated
demo, which feeds it spoken commands and booking emails and prints every
decision it makes.

```bash
python -m packages.nlp build        # once: generates data/nlu/splits/ (git-ignored;
                                    # demo section 3 crashes without it)
python -m packages.demo             # full end-to-end walkthrough
python -m packages.demo --list      # list the sections
python -m packages.demo --only 1    # run a single section by number
```

Run the FR5 safety gate + the 66-scenario regression suite (exits 2 on any
unconfirmed destructive action):

```bash
python -m packages.bench --all
python -m packages.bench --all --dump-trajectories data/trajectories
```

### 5. Optional: NLP, evaluation, dashboard

```bash
# NLU data + scores
python -m packages.nlp build
python -m packages.nlp evaluate --split test

# Evaluation and research
python -m packages.bench.latency
python -m packages.bench.compare
python -m packages.bench.recommend
python -m packages.analysis.rq1
python -m packages.analysis.rq2

# Dashboard (static, no build step)
python dashboard/build_index.py
# then open dashboard/index.html in a browser, or:
python dashboard/serve.py
```

### 6. Driving the agent programmatically

Until a UI exists, this is the direct way to send a command to the agent:

```python
from datetime import datetime
from packages.sync_agent.agent import SyncAgent
# see packages/demo/__main__.py for how the agent is constructed with fakes

agent = ...                        # build as in the demo
result = agent.handle_command("cancel my flight to Delhi on Friday", now=datetime.now())
print(result)
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| Double-clicking `run_demo.bat` flashes and closes | Python is not on PATH; reinstall Python and tick "Add python to PATH" |
| `FileNotFoundError: ...data/nlu/splits/train.jsonl` | Run `python -m packages.nlp build` (the launcher does this for you) |
| `ModuleNotFoundError: packages` | Run commands from the repo root with the venv activated |
| `pip install -e .` fails | Upgrade pip: `python -m pip install --upgrade pip` |
| Tests can't find scenarios | Don't move `data/scenarios/` — it is source, not generated data |
| `data/trajectories/` is empty | Run `python -m packages.bench --all --dump-trajectories data/trajectories` |
