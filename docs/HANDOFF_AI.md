# HANDOFF_AI — SceneBridge booth (v1.2), 2026-10-26

For an AI agent resuming this work. The binding rules are in `AGENTS.md`. Reply to the user in Korean. Human docs:
- `docs/HANDOFF.md` (operations)
- `docs/INSTALL.md`
- `docs/ASSETS.md` (separately delivered files)
- `docs/METHODOLOGY.md`

## What this repo is

This is a clean extract of the v1.2 booth path from the private development repo. It has a fresh history and no recordings.

```
cameras x4 / recorded AVI -> sync (100 ms) -> A1 RF-DETR-Seg nano (10 prop_id classes, allowed = booth list)
 -> object_centric association + triangulation (support plane, class heights from catalog)
 -> merge_split 0.03 m / hide_leftover 0.15 m -> keep_counted(counts) -> tracker (hold_seconds 1.5, time-based only)
 -> OrientationRunner (DINOv2-small + linear 24 cls, every 0.5 s, circular mean, keep 0.3) -> display_name
 -> UDP 127.0.0.1:7001 JSON -> Isaac ext gist.tabletop_twin
agent: Isaac panel -> HTTP 7080 agent_service -> SGLang 30000 (Gemma 4 26B-A4B NVFP4) -> tools -> Isaac PhysX
```

### Key files

| Area | Path |
|---|---|
| Core | `smartphone_mocap/perception.py` |
| Recognition | `recognition/{objects.py (keep_counted), orientation.py}` |
| Reconstruction | `object_reconstruction.py`, `object_centric_association.py` |
| Booth list | `booth_props.py` |
| Venues | `venues.py` |
| Output | `publisher.py` |
| Agent | `scene_agent.py`, `experiment_planner.py`, `physics_scenario.py`, `agent_service.py` |
| Smoke test rules | `smoke_test.py` |

### Configs

- `config/booth_props.json`: the student-edited booth list.
- `assets/props/catalog.json`: model, height and symmetry per prop.
- `config/venues.json`: only `dongshin_ai_classroom`.
- `smartphone_mocap/config/booth.json`: tracker config. It names the venue, the calibration (`venue_20261025`) and the models (`outputs/models/`).
- `smartphone_mocap/config/tabletop_agent.json`: agent config.

### Launchers and tools

- `tools/booth_start.sh` runs `booth_check`, then the venue check, then SGLang, then `run_tabletop_demo.sh`.
- `run_tabletop_demo.sh` starts Isaac, the live tracker and the agent.
- Other tools: `run_tabletop_replay.sh`, `booth_check.py`, `booth_smoke_test.py`, `venue_args.py` (`--venues` for another presets file).
- Calibration tools: see `docs/HANDOFF.md` §6.
- Synthetic data and training: see `docs/METHODOLOGY.md` §2.

### Message and table frame

UDP object fields:
- Always present: `id`, `label` (= prop_id), `state` (`candidate`, `active` or `partial`), `xyz` (table frame, m, bottom centre), `source_cameras`, `asset_uri`.
- Present only when set: `yaw_deg` (upright and non-symmetric only), `pose` (recorded only), `display_name`.
- The message also carries `rig_to_scene` (venue yaw).

Table frame:
- Origin at the front-left corner on the B side; table size 1.25 × 0.75 m.
- +X from B to D.
- +Y toward the wall.
- +Z up.

## Not in git (and where it comes from)

| What | Where it goes | Source |
|---|---|---|
| Model weights | `outputs/models/{prop_detector_a1.pth, orientation_head.pt}` | Delivered separately (not public) |
| Meshy prop models | `assets/props/<prop_id>/model.usdz` | Delivered separately. `model_placed.usda` references `./model.usdz` |
| Dongshin background | `usd/venues/source/Dongshin_Final/Dongshin_Final.usda` (+ folder) | Delivered separately. The wrapper `usd/venues/dongshin_ai_classroom.usda` sublayers it |
| Recordings | `recordings/` | Never published (people in frame) |
| DINOv2-small | HF cache (`~/.cache/huggingface-gist/hub` or `HF_HUB_CACHE`) | Auto-download |

- `load_venue(..., check_files=False)` lets the tracker read the venue yaw without the background.
- Isaac launch (`venue_args.py`) still refuses a missing background or source USD, with a Korean message.

## Verified (2026-10-26, lab, NVIDIA GB10 aarch64)

- Suite: `.venv-mocap/bin/python -m pytest tests -q` gives 419 passed in this repo.
  - The development repo has more tests, for tools that are not shipped here.
- Calibration (`venue_20261025`): reprojection error 0.10–0.14 px; jitter over 40 shots 1.3–9.3 mm. This is calibration quality, not object accuracy.
- Live, placement watch (6 min): 7 of 8 props shown exactly once from 3–4 cameras. Lying giraffe and dinosaur were classified `lying`. `stainless_mug` was never detected. The mascot on the table was an untrained doll.
- Live, movement test (9 min, 1775 frames, about 3.2 Hz): 3 frames had a duplicated label. Position jumps every 0.3–1 s while an object is carried.
- Smoke test:
  - Recognition (first 30 s of the still recording): all 7 props exactly once 99% of frames, spread 0.1–1.0 cm.
  - Agent: Korean-name QA passes; force-without-duration returns `needs_clarification`.
- Synthetic comparison (120 scenes): A1 "exactly one + right name" 0.919 at 72 ms. Untrained objects are found 3–4% of the time. Orientation: pose 0.89, yaw 11°. These are SYNTHETIC numbers that favor A1 (`docs/eval/20261025_recognition_comparison.md`).

**Not verified:**
- Position error in cm.
- Yaw accuracy.
- Real vanish delay.
- Physics experiments end-to-end on v1.2 live data.
- Any machine other than GB10.

Count/name match is not position or ID accuracy. Report them separately.

## Behaviours that surprise people

- **Physics route needs 시뮬/물리 in the prompt** (`physics_scenario.simulation_requested`). Without it, "오리를 … 밀어 줘" is treated as layout, and the model says physics is unsupported.
- **Physics planning needs the Isaac `twin_state`** (USD bounds, viewport capture). A tracking-only snapshot fails with "USD geometry bounds are unavailable". So `booth_smoke_test.py` checks only QA and the duration clarification. Planning and real contact are covered by the live 5-minute check.
- **Korean names**: each object carries `name_ko` (booth `name_ko`, else the builtin Korean word, else the label). The summary has `counts_by_name_ko`, and the system prompt forbids internal ids in replies.
- **Restart after agent changes**: the running `agent_service` must be restarted to pick up code changes. SIGINT may not stop it; use SIGTERM by PID.
- **Isaac exits at start with "TSC ran backwards" (139)**: transient; retry.
- **Short viewport cuts the table**: the camera panel makes the viewport short. Override `camera_eye_*` / `camera_target_*` (see `docs/HANDOFF.md` §7).
- **Shared GPU**: Isaac RTX, RF-DETR and SGLang share one GPU. Isaac has crashed with Xid 109 / DEVICE_LOST under load. Avoid tracker restarts during the booth, and do not run synthetic rendering while the demo Isaac is up.
- **SGLang constrained decoding**: regex/grammar decoding crashed the server (NCCL) in this container. Do not use it.
- **Dark room**: lights off means only one camera sees the table, and objects stay `partial`.

## Deferred minors from the v1.2 final review

- Table yaw arrives only with the first tracking message, so the table is unrotated for about 20 s at startup.
- `OrientationRunner._state` is never pruned, so a stale yaw shows briefly on reid.
- A backward timestamp freezes orientation updates.
- The stored yaw is overwritten with None when all of a frame's bins are zero.
- An orientation head load failure stops perception instead of disabling orientation.
- `keep_counted` caps per frame, not tracker output, so a fast move can briefly show 2 tracks.
- The venue silently replaces the `rig.scene_placement` translation.
- `isaac_args` uses `:g` (6 significant digits).
- The "frame table" button jumps to the venue camera instead of fitting the table.
- `make_venue` groups walls by `round(coord, 1)`.

## Open work (ordered)

1. Position accuracy test with ruler reference points. Write it to `docs/eval/`, keeping count/name, position and yaw separate.
2. Retrain `stainless_mug` from a real-photo Meshy model.
3. Later:
   - COCO-80 + props joint training (merge overlapping classes).
   - Class-agnostic finder (empty-table background subtraction) for on-site props.
   - Other venue presets.
   - Pose display.
   - Student guide.
