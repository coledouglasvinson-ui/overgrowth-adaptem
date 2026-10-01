# Overgrowth adaptive combat training

This prototype exposes one Adaptem Red or Adaptem Blue combat AI as a lockstep Gymnasium environment. The selected Adaptem's strategy actions are controlled by a Python PPO policy; other Adaptems use their own persistent in-game learner, and ordinary NPCs keep their normal behavior. Red and Blue have separate memories, each saved to the game config after every learned decision, so each improves from its own fights. The game remains paused while Python chooses each action. Training is opt-in and only connects to `127.0.0.1`.

## Requirements

Use Python 3.10–3.14 and install the trainer dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r Tools\adaptive_ai\requirements.txt
```

## Enable the game bridge

Build and run the game from this checkout using `COMPILING.md`; the bridge adds native AngelScript bindings and a dedicated simulation pause. In the game's writable `Data\config.txt` (on Windows, `My Documents\Wolfire\Data\config.txt`), set:

```text
adaptive_ai_training: true
```

The default TCP port is `47852`. If you change the trainer's `--port`, set the same `adaptive_ai_training_port: <port>` in the game config. Start training before loading a level so the local Python listener is ready before the game connects. Use a sandbox level containing an Adaptem and an opponent; the trainer waits until an Adaptem is actively fighting, then selects the first eligible one for that episode. Training is disabled in multiplayer.

## Train PPO

From the repository root, start the listener and trainer:

```powershell
python Tools\adaptive_ai\train.py --timesteps 100000
```

Then launch the game, load the sandbox, and let the process run. The trainer writes resumable checkpoints to `runs\adaptive_combat.pt` and episode returns to `runs\adaptive_combat.csv`. Resume a run with:

```powershell
python Tools\adaptive_ai\train.py --resume runs\adaptive_combat.pt --timesteps 200000
```

The single-agent action space is `0=wait/attack`, `1=rush/attack`, `2=defend`, `3=provoke`; the 8 discrete observations combine near/far range, whether the target is armed, and whether it is attacking. Rewards come from the existing combat feedback (damage, failed attacks, blocks, dodges, and defeat). Reset ends the current sandbox episode and reloads the level.

Only localhost connections are accepted. Keep the bridge disabled during normal play by omitting the setting or setting it to `false`. If the trainer exits, the game releases the training pause and falls back to local AI.

## Tests

Run the protocol-only tests without installing Gymnasium or PyTorch:

```powershell
python -m unittest discover -s Tools\adaptive_ai\tests -p test_protocol.py
```

After installing the requirements, run all environment and PPO tests:

```powershell
python -m unittest discover -s Tools\adaptive_ai\tests
```
