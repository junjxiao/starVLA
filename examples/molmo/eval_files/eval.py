from pathlib import Path
from molmo_spaces.evaluation.eval_main import run_evaluation
from examples.molmo.eval_files.configs import StarVLAEvalConfig
import dataclasses
import tyro

@dataclasses.dataclass
class Args:
    host: str = "localhost"
    port: int = 8000
    benchmark_dir: str = ""
    checkpoint_path: str = ""
    output_dir: str = ""
    task_horizon_steps: int = 450

def eval_molmo(args: Args) -> None:
    results = run_evaluation(
        eval_config_cls=StarVLAEvalConfig,
        benchmark_dir=Path(args.benchmark_dir),
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        task_horizon_steps=args.task_horizon_steps,
        use_wandb=False,
        host=args.host,
        port=args.port,
    )

    print(f"Success rate: {results.success_rate:.1%}")
    for r in results.episode_results:
        print(f"{r.house_id}/ep{r.episode_idx}: {'pass' if r.success else 'fail'}")


if __name__ == "__main__":
    tyro.cli(eval_molmo)