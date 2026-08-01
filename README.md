# driver-state-vla

Driver-state (alertness/distraction) classifier built by repurposing
[Wild-Drive](https://github.com/wangzihanggg/Wild-Drive)'s ViT + MoRo-Former
tokenizer. Wild-Drive fuses camera + LiDAR into tokens for an LLM captioner
and a trajectory planner; this project keeps only the tokenizer, feeds it
road-facing + driver-facing camera instead of camera + LiDAR, adds a control
(steering/throttle/brake/speed + acceleration) token stream and an optional
text token stream, and replaces the LLM/planner with a small transformer
classification head plus a gaze-heatmap head.

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
