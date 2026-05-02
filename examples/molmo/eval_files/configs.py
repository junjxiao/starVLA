from molmo_spaces.configs.policy_configs import BasePolicyConfig
from .policy import STARVLA_Policy

class StarVLAPolicyConfig(BasePolicyConfig):
    checkpoint_path: str = ""
    remote_config: dict = dict(host="localhost", port=8000)
    prompt_object_word_num: str = 1  # number of words as the object name
    prompt_templates: list[str] | None = None
    grasping_type: str = "binary"
    grasping_threshold: float = 0.5

    policy_cls: type = None
    policy_type: str = "learned"

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        if self.policy_cls is None:
            self.policy_cls = STARVLA_Policy


from molmo_spaces.configs.robot_configs import FrankaRobotConfig
from molmo_spaces.evaluation.configs.evaluation_configs import JsonBenchmarkEvalConfig

class StarVLAEvalConfig(JsonBenchmarkEvalConfig):
    robot_config: FrankaRobotConfig = FrankaRobotConfig()
    policy_config: StarVLAPolicyConfig = StarVLAPolicyConfig(
        checkpoint_path=""
    )
    policy_dt_ms: float = 66.0  # Match your model's expected control rate

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.robot_config.action_noise_config.enabled = False