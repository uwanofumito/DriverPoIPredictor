# driver-state-vla

Driver-state (alertness/distraction) classifier built by repurposing
[Wild-Drive](https://github.com/wangzihanggg/Wild-Drive)'s ViT + MoRo-Former
tokenizer. Wild-Drive fuses camera + LiDAR into tokens for an LLM captioner
and a trajectory planner; this project keeps only the tokenizer, feeds it
road-facing + driver-facing camera instead of camera + LiDAR, adds a control
(steering/throttle/brake/speed + acceleration) token stream and an optional
text token stream, and replaces the LLM/planner with a small transformer
classification head.

## Architecture

```
road image   ─┐                      ┌─ MoRo-Former "camera" slot ─┐
               ├─ shared ViT ─────────┤                             ├─ fusion tokens ─┐
driver image ─┘   (frozen)            └─ MoRo-Former "lidar" slot ──┘                  │
                                                                                         │
control series (steer/throttle/brake/speed + accel) ─ ControlEncoder ─ control tokens ─┤
                                                                                         ├─ [CLS] + concat ─ 2-layer
optional free text ─ frozen DistilBERT + compressor ─ text tokens (or learned "no text")┤   transformer  ─ MLP ─ class logits
                                                                                         │
driver embedding × road ViT patches ─ GazeHead (unsupervised attention) ────────────────┴─ heatmap (interpretability only)
```

- `models/moroformer.py` — copied unmodified from Wild-Drive. It's modality-agnostic:
  whatever fills its `lidar_features` argument gets routed through the same
  task/expert queries as whatever fills `camera_features`. Here, driver-facing
  camera fills the first slot, road-facing camera the second.
- `models/control_encoder.py` — MLP + the same learned-query compression trick
  MoRo-Former uses internally (`BranchCompressor`), turning a variable-length
  control/IMU window into a fixed token count.
- `models/text_encoder.py` — optional branch, off by default in spirit but
  wired in (`use_text=True` in `config.py`). Brain4Cars has no free-text
  annotations, so at least at first every sample's `text` field will be `""`
  and the model will use the learned "no-text" placeholder tokens — but the
  slot is there for later (weather, road type, free-form notes, etc.).
- `models/gaze_head.py` — **not a calibrated gaze estimator.** There's no
  gaze-target ground truth in Brain4Cars to supervise it with. It's a
  cross-attention head (driver embedding as query, road ViT patch grid as
  key/value) trained only indirectly through the classification loss. Treat
  its output as "where the model weighted the road image when it made its
  decision," not as a physical point of regard.

## Getting a real gaze estimate later

If the unsupervised heatmap isn't good enough, the accurate path discussed
was: pretrain (or fine-tune) `GazeHead` with real synced gaze data —
[DR(eye)VE](http://imagelab.ing.unimore.it/dreyeve) has driver eye-tracking
+ forward-road video — then fine-tune the rest of the model (classifier
head) on Brain4Cars with the gaze head frozen or jointly trained. That needs
a second dataset loader and a multi-task loss; not implemented here.

## Brain4Cars manifest format

`data/dataset.py`'s `Brain4CarsDataset` expects a JSON manifest:

```json
[
  {
    "road_image": "relative/or/absolute/path/road.jpg",
    "driver_image": "relative/or/absolute/path/driver.jpg",
    "control": [[steering, throttle, brake, speed, accel_x, accel_y, accel_z], ...],
    "text": "highway, clear weather",
    "label": 0
  }
]
```

Brain4Cars ships driver/road video + vehicle dynamics, not this JSON
directly — write a small preprocessing script that extracts frames and syncs
telemetry into this format. `control` is a short time window (T steps); a
single snapshot (T=1) also works.

## Running

```bash
pip install -r requirements.txt

# Smoke-test the pipeline with synthetic data (no dataset, no download of
# facebook/dinov2-base weights needed if you swap in a tiny stand-in model —
# see train.py's --vision-model-path-equivalent note below)
python train.py --mock-samples 32 --epochs 1 --batch-size 2

# Train on real data once you have a manifest
python train.py --manifest /path/to/manifest.json --epochs 20
```

Mock mode (`--manifest` omitted) skips image loading/preprocessing and feeds
random tensors of the right shape straight into the model — it verifies the
forward/backward pass and tensor shapes, not model quality. The ViT
(`facebook/dinov2-base` by default) and, if `use_text=True`, DistilBERT still
get downloaded from Hugging Face on first run either way, since they're part
of the model itself.

## Class labels

`config.num_classes` defaults to 4; map them to whatever driver-state taxonomy
you're targeting (e.g. alert / drowsy / distracted / cognitively-loaded).
Brain4Cars' native labels are maneuver classes (lane change, turn, etc.), not
alertness states — if you're using Brain4Cars for the state-estimation task
itself rather than as a stand-in for a proprietary dataset, you'll need to
relabel or find/derive alertness annotations separately.
