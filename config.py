"""Config for the driver-state estimation model.

Reuses Wild-Drive's ViT + MoRo-Former as a modality-agnostic tokenizer:
the "lidar" branch slot is repurposed for the driver-facing camera, and
the "camera" branch slot stays the road-facing camera. Text and control
(steering/throttle/brake + acceleration) are separate token streams
fused before the classifier head.
"""


class DriverStateConfig:
    def __init__(
        self,
        # ---- vision tokenizer (ViT) ----
        vision_model_path="facebook/dinov2-base",
        freeze_vision_model=True,
        share_vision_encoder=True,  # one ViT for both road + driver cameras

        # ---- MoRo-Former fusion ----
        token_dim=768,              # common hidden dim for all token streams
        moro_num_tasks=5,
        moro_queries_per_task=64,
        moro_num_heads=8,
        moro_num_layers=2,
        moro_tokens_per_branch=4,

        # ---- control encoder (steering/throttle/brake/speed + accel) ----
        control_input_dim=7,
        num_control_tokens=4,

        # ---- optional text encoder ----
        use_text=True,
        text_model_path="distilbert-base-uncased",
        freeze_text_model=True,
        num_text_tokens=4,

        # ---- gaze head (unsupervised attention visualization) ----
        use_gaze_head=True,
        gaze_heatmap_size=16,       # ViT patch grid side (e.g. 16x16 for 224px/14px patches)

        # ---- classifier head ----
        num_classes=4,              # e.g. alert / drowsy / distracted / cognitively-loaded
        classifier_num_layers=2,
        classifier_num_heads=8,
        classifier_dropout=0.1,

        **kwargs,
    ):
        self.vision_model_path = vision_model_path
        self.freeze_vision_model = freeze_vision_model
        self.share_vision_encoder = share_vision_encoder

        self.token_dim = token_dim
        self.moro_num_tasks = moro_num_tasks
        self.moro_queries_per_task = moro_queries_per_task
        self.moro_num_heads = moro_num_heads
        self.moro_num_layers = moro_num_layers
        self.moro_tokens_per_branch = moro_tokens_per_branch
        self.num_fusion_tokens = moro_num_tasks * 3 * moro_tokens_per_branch

        self.control_input_dim = control_input_dim
        self.num_control_tokens = num_control_tokens

        self.use_text = use_text
        self.text_model_path = text_model_path
        self.freeze_text_model = freeze_text_model
        self.num_text_tokens = num_text_tokens

        self.use_gaze_head = use_gaze_head
        self.gaze_heatmap_size = gaze_heatmap_size

        self.num_classes = num_classes
        self.classifier_num_layers = classifier_num_layers
        self.classifier_num_heads = classifier_num_heads
        self.classifier_dropout = classifier_dropout

        for k, v in kwargs.items():
            setattr(self, k, v)
