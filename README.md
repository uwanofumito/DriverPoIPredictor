# driver-state-vla

Driver-state estimation model built by repurposing
[Wild-Drive](https://github.com/wangzihanggg/Wild-Drive)'s ViT + MoRo-Former
tokenizer. Wild-Drive fuses camera + LiDAR into tokens for an LLM captioner
and a trajectory planner; this project keeps only the tokenizer, feeds it
road-facing + driver-facing camera instead of camera + LiDAR, adds a control
(steering/throttle/brake/speed + acceleration) token stream and an optional
text token stream, and replaces the LLM/planner with a small transformer
classification head plus a gaze-heatmap head.

## Project status (read this first)

What's actually working and validated end-to-end on real data:
- The tokenizer/fusion pipeline (road + driver camera → MoRo-Former → tokens)
- Control-signal encoding (real steering/throttle/brake/speed/accel from ETS2)
- **GazeHead, supervised against a Gaze360-pretrained gaze estimator** — this
  is the part of the model that's actually learning something meaningful
  (loss drops steadily across training). See the results gallery for a
  before/after visual.

What's explicitly de-scoped or on hold, and why:
- **Distraction/alertness classification** (the `logits` output) — kept in
  the architecture, but not a near-term goal. ETS2 essentially contains no
  real distraction behavior (subjects had no secondary task), so the
  classifier reliably collapses to the majority class no matter how the
  proxy label is tuned. Not a bug to fix; a data problem with no cheap fix.
- **"Looked-but-failed-to-see" driver inattention** (hazard visible on the
  road, e.g. a crossing pedestrian, but driving behavior/control doesn't
  respond to it) — the actual target concept behind the classifier, but
  requires (a) hazard/pedestrian detection on the road camera and (b) a
  behavior-mismatch label, neither of which exist yet. Checked whether ETS2
  could support this first: ran a COCO-pretrained detector
  (`check_pedestrians.py`) over all 113 road-camera frames — **zero real
  pedestrian detections** (one 0.57-confidence hit, visually confirmed to be
  a false positive). ETS2 is a highway/rural trucking sim; this scenario
  just doesn't occur in it. On hold pending a dataset that actually has it
  (CARLA-scripted scenarios were discussed as the most promising route,
  since CARLA can supply ground-truth hazard-visibility rather than needing
  a separate detector — not implemented).
- **Pedestrian-side state estimation** (the pedestrian's own crossing
  intention / attention-to-traffic, as opposed to the driver's) — a
  separate, better-supported research direction (JAAD dataset — 346 clips,
  3.1GB, verified downloadable, no registration, with real per-frame
  "looking at traffic" / "crossing" / "walking-vs-standing" annotations —
  no proxy labels needed here, unlike the driver side). **In progress** as
  a standalone model (`models/pedestrian_state_model.py`,
  `train_pedestrian.py`), deliberately not wired into `DriverStateModel` yet
  — see "Pedestrian-state estimation (JAAD)" and "Integrating pedestrian-
  state estimation later" below for results and how it's meant to attach
  without changing what's already built.

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
driver embedding × road ViT patches ─ GazeHead (cross-attention) ───────────────────────┴─ heatmap, supervised against
                                                                                            a Gaze360-derived proxy target
```

- `models/moroformer.py` — copied unmodified from Wild-Drive. It's modality-agnostic:
  whatever fills its `lidar_features` argument gets routed through the same
  task/expert queries as whatever fills `camera_features`. Here, driver-facing
  camera fills the first slot, road-facing camera the second.
- `models/control_encoder.py` — MLP + the same learned-query compression trick
  MoRo-Former uses internally (`BranchCompressor`), turning a variable-length
  control/IMU window into a fixed token count.
- `models/text_encoder.py` — optional branch (`use_text=True` in `config.py`).
  Neither dataset used here ships free-text annotations, so by default every
  sample's `text` field is `""` and the model uses the learned "no-text"
  placeholder tokens — the slot is there for later (weather, road type, notes).
- `models/gaze_head.py` — cross-attention head (driver embedding as query,
  road ViT patch grid as key/value) producing a heatmap over the road image.
  When a manifest provides `gaze_target`, `train.py` adds a soft-target loss
  (`gaze_target_heatmap()`) so it's actually supervised, not just an
  incidental byproduct of the classification loss.

## Data: Brain4Cars is unavailable, ETS2 is the working stand-in

The original plan was Brain4Cars (Jain et al., ICCV 2015) — maneuver
anticipation dataset with driver-facing + road-facing video. **Its raw video
is no longer retrievable**: the official Dropbox share's folder tree is
still browsable, but every event folder is empty (files removed at some
point after the link was published).

The working pipeline instead targets the Zenodo-hosted **ETS2 (Euro Truck
Simulator 2) driving dataset** (Tonutti & Ruffaldi, 2017, DOI
[10.5281/zenodo.1009540](https://doi.org/10.5281/zenodo.1009540)) — same
shape (paired road/driver video + per-event vehicle telemetry), 113 events,
~528MB, actually downloadable. Its native labels are maneuver classes
(lchange/lturn/rchange/rturn/straight), not alertness states — see below.

```bash
# download+extract face_camera.zip / road_camera.zip / car_data.zip from the
# Zenodo record above into some directory, then:
python data/prepare_ets2_manifest.py \
  --src-root /path/to/extracted \
  --out-dir data/raw/ets2/prepared
```

This extracts the last frame of each road/driver clip, builds a
`control` series from the telemetry JSONs (steering/throttle/brake/speed +
3-axis acceleration — this dataset actually has real control signals, unlike
Brain4Cars which only ever had GPS speed), and computes a `label` +
`gaze_target` via gaze estimation (next section). It writes
`data/raw/ets2/prepared/manifest.json` in the schema `data/dataset.py`
expects (see `ManifestDataset`).

## Gaze estimation (`data/gaze_estimation.py`)

No dataset here has real distraction/gaze-target ground truth, so both are
derived from a **pretrained gaze estimator**:
[yakhyo/gaze-estimation](https://github.com/yakhyo/gaze-estimation)'s
resnet18, pretrained on **Gaze360** (real gaze annotations from many
subjects; MIT license, architecture vendored into `models/gaze_resnet.py`).
MediaPipe FaceLandmarker is used only to detect/crop the face before it goes
into that model — its head-pose output is not used directly (an earlier
version of this pipeline used head pose alone as the proxy; the gaze model
is a strict improvement — same idea, but trained on real gaze data instead
of a hand-picked field-of-view heuristic).

Required model assets (both gitignored under `models_data/`, download once):

```bash
mkdir -p models_data
curl -L "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" \
  -o models_data/face_landmarker.task
curl -L "https://github.com/yakhyo/gaze-estimation/releases/download/weights/resnet18.pt" \
  -o models_data/gaze_resnet18.pt
```

What's still a heuristic even with a real gaze model:
- **Session-relative baseline**: each recording session has its own neutral
  gaze pitch (camera mounting + how each face reads to the model varies), so
  the distraction proxy uses deviation from that session's own median pitch,
  not an absolute threshold.
- **Distraction label**: pitch more than 10° below the session baseline (looking
  down — phone/lap/dashboard) for over 30% of sampled frames in the 5s clip.
  Yaw (sideways) is excluded on purpose — mirror/blind-spot checks before a
  turn or lane change are legitimate, not distraction, and there's no ground
  truth to tell the two apart otherwise.
- **Gaze target point**: yaw/pitch projected onto the road image assuming a
  windshield field of view of ±35° horizontal / ±25° vertical. The gaze
  *angles* are model output from real training data; this angle→pixel
  projection is not calibrated to this dataset's actual camera geometry.

## Manifest format

`data/dataset.py`'s `ManifestDataset` expects a JSON manifest (source-agnostic —
`data/prepare_ets2_manifest.py` is one example producer):

```json
[
  {
    "road_image": "relative/or/absolute/path/road.jpg",
    "driver_image": "relative/or/absolute/path/driver.jpg",
    "control": [[steering, throttle, brake, speed, accel_x, accel_y, accel_z], ...],
    "text": "highway, clear weather",
    "label": 0,
    "gaze_target": [0.42, 0.55]
  }
]
```

`control` is a short time window (T steps); a single snapshot (T=1) also
works. `text` and `gaze_target` are both optional.

## Running

```bash
pip install -r requirements.txt

# Smoke-test the pipeline with synthetic data (no dataset, no manifest)
python train.py --mock-samples 32 --epochs 1 --batch-size 2

# Train on real data, save a checkpoint for inference/visualization
python train.py --manifest data/raw/ets2/prepared/manifest.json \
  --num-classes 2 --epochs 8 --save-checkpoint outputs/driver_state_model.pt

# Visualize results (writes outputs/*.json consumed by an HTML gallery)
python make_results_gallery.py
python make_animated_examples.py
```

Mock mode (`--manifest` omitted) skips image loading/preprocessing and feeds
random tensors of the right shape straight into the model — it verifies the
forward/backward pass and tensor shapes, not model quality. The ViT
(`facebook/dinov2-base` by default) and, if `use_text=True`, DistilBERT still
get downloaded from Hugging Face on first run either way, since they're part
of the model itself.

## Class labels

`config.num_classes` defaults to 4 (a generic alert/drowsy/distracted/
cognitively-loaded taxonomy placeholder); pass `--num-classes 2` for the
ETS2 manifest's attentive/distracted proxy label, or `5` if you regenerate
the manifest with `--skip-gaze` to train on its native maneuver classes
instead.

**Known limitation:** only ~1 of 113 ETS2 events cross the distraction
threshold (was 6/112 with the earlier head-pose-only proxy — the Gaze360
model appears to be less noisy, which if anything makes this worse for
training). The classifier reliably collapses to predicting the majority
class; there simply isn't enough real distraction behavior in this
simulator study (no secondary task was assigned to subjects) to learn a
decision boundary from. The gaze heatmap is the part of this pipeline that
trains meaningfully — its loss drops steadily across epochs. See the
results gallery for a concrete before/after comparison.

## Scripts

| Script | Purpose |
|---|---|
| `train.py` | Train (mock or real manifest), optionally save a checkpoint |
| `data/prepare_ets2_manifest.py` | Build a manifest.json from raw ETS2 video+telemetry |
| `data/gaze_estimation.py` | Gaze360-pretrained gaze estimation (library, not a script) |
| `make_results_gallery.py` | Run a checkpoint over real events → static overlay images + JSON |
| `make_animated_examples.py` | Same, but per-frame across a clip → animated GIFs |
| `visualize_gaze.py` | Quick sanity-check on synthetic images (no real data needed) |
| `check_pedestrians.py` | Diagnostic: scans road-camera frames for pedestrian detections (COCO detector, no extra data). Used to confirm ETS2 has none. |
| `data/prepare_jaad_manifest.py` | Build a pedestrian-crop manifest.json from raw JAAD video+XML annotations |
| `train_pedestrian.py` | Train the standalone `PedestrianStateModel` on a JAAD manifest |
| `make_pedestrian_gallery.py` | Run a pedestrian-state checkpoint over sample crops → JSON for a results gallery |

## Pedestrian-state estimation (JAAD)

A **standalone** model (`models/pedestrian_state_model.py`: frozen
`facebook/dinov2-base` — same ViT as `DriverStateModel`, on purpose — plus
three small classification heads) trained on real JAAD annotations, not a
proxy. No dataset access issues here: JAAD's clips + XML annotations
download directly (3.1GB, no registration — verified), and every pedestrian
track already has real per-frame `look` (looking/not-looking at traffic),
`cross` (crossing/not-crossing), and `action` (walking/standing) labels.

```bash
# after downloading+extracting JAAD_clips.zip and cloning ykotseruba/JAAD's
# annotations/ folder:
python data/prepare_jaad_manifest.py \
  --annotations-dir /path/to/JAAD/annotations \
  --clips-dir /path/to/JAAD_clips \
  --out-dir data/raw/jaad/prepared --frame-stride 15

python train_pedestrian.py --manifest data/raw/jaad/prepared/manifest.json \
  --epochs 5 --save-checkpoint outputs/pedestrian_state_model.pt
```

**Results** (9,074 crops from 346 clips, held-out validation split):

| Attribute | Majority baseline | Val accuracy after 5 epochs |
|---|---|---|
| `cross` (crossing intention) | 55.7% | **79.6%** — genuinely learned |
| `action` (walking vs. standing) | 85.0% | 87.9% — modest improvement |
| `look` (looking at traffic) | 82.0% | 83.5% — barely above baseline |

Crossing intention trains well from a single frame. Attention (`look`) does
not — visually, the model mostly predicts "not-looking" regardless of the
true label. Single-frame appearance may just be a weak cue for gaze
direction on typically-small, low-resolution pedestrian crops; the
head-pose-specialized approach used for the driver side (`gaze_resnet18`,
see above) might transfer better here than a generic frozen-ViT classifier
— not tried yet.

## Integrating pedestrian-state estimation later

`models/pedestrian_state_model.py` exists now and trains on its own (see
above), but it is **not wired into `DriverStateModel`** — that integration
is still just a plan, deliberately deferred until there's a dataset with
road camera + driver camera + control *and* pedestrians together (JAAD has
no driver-facing camera or control telemetry; ETS2 has no pedestrians).
The plan is to add it as an **additional branch**, not a rework of what's
here — the same pattern `ControlEncoder` and `TextEncoder` already follow:

1. `PedestrianStateModel`'s frozen ViT + pooled features (or a thin
   adapter over them) feed a new `models/pedestrian_encoder.py`, producing
   a fixed-size token set, the same way `ControlEncoder.forward()` turns a
   variable-length control window into `num_control_tokens` tokens. Reusing
   `facebook/dinov2-base` for both was a deliberate choice for this reason.
2. `DriverStateModel.__init__` gains one more optional branch behind a
   `config.use_pedestrian` flag (mirrors `config.use_text`), and
   `DriverStateModel.forward()` appends its output to `token_streams`
   before the `[CLS]` + transformer classifier — no change to
   `MoRoFormer`, `GazeHead`, `ControlEncoder`, or the classifier itself.
3. The manifest schema gains an optional field (e.g. `"pedestrians": [...]`)
   the same way `gaze_target` was added without breaking manifests that
   don't have it (`ManifestDataset`/`collate_driver_state` both already
   handle optional fields this way — see `dataset.py`).

This keeps today's architecture and trained components (gaze estimation
pipeline, fusion tokenizer) reusable as-is when that work starts.
