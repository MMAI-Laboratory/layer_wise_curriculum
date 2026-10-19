"""End-to-end layer-wise curriculum training from a YAML experiment config.

    python scripts/run_curriculum.py --config configs/experiments/llama3.2-3b_28to19.yaml \
        --work_dir outputs/llama3.2-3b_28to19 --gpus 0,1,2,3

For every curriculum phase p = 1..K (see ``layer_curriculum/curriculum.py``):

    1. extract features   teacher features (p = 1) or features of the stacked student of phase p-1 (p >= 2)
    2. train layers       one process per active layer, distributed over ``--gpus``
    3. stack layers       -> <work_dir>/stacked  (a standard HF model, the final output after phase K)

With ``--post_train`` the ``post_train`` section of the config is run afterwards (-> <work_dir>/post_train).

Finished steps are recorded in ``<work_dir>/.done`` so an interrupted run resumes where it stopped.
Configs with ``mode: online`` (BERT / GPT-2) skip feature caching and compute the features on the fly; their
layers are trained one after another, each data-parallel over all ``--gpus``. An epoch is then one pass over
``train_file`` instead of over the cached samples.

Batch sizes given as ``global_batch_size`` are independent of the number of GPUs: the gradient accumulation
steps are derived from it at launch time.
"""
import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time

import yaml
from transformers import AutoConfig

from layer_curriculum.architectures import get_arch
from layer_curriculum.curriculum import build_phases
from layer_curriculum.layer_selection import resolve_layer_config

SCRIPTS = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--work_dir", required=True, help="Features, layer weights, logs and the stacked model")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU ids, e.g. 0,1,2,3")
    parser.add_argument("--model", default=None, help="Override `model` of the config (e.g. a local path)")
    parser.add_argument("--train_file", default=None, help="Override `train_file` of the config")
    parser.add_argument("--jobs_per_gpu", type=int, default=1,
                        help="Layer-training processes sharing one GPU (one layer needs about 1 GB for a 3B model)")
    parser.add_argument("--start_phase", type=int, default=1, help="Skip phases before this one")
    parser.add_argument("--post_train", action="store_true",
                        help="Run the `post_train` section of the config on the final model (all --gpus, DDP)")
    parser.add_argument("--post_train_only", action="store_true", help="Only run the post-training stage")
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--project", default="layer-curriculum")
    parser.add_argument("--cleanup_features", action="store_true",
                        help="Delete the feature cache of a phase once its layers are trained (saves disk)")
    parser.add_argument("--dry_run", action="store_true", help="Only print the commands")
    return parser.parse_args()


def to_cli(options: dict):
    cli = []
    for key, value in options.items():
        if value is None or value is False:
            continue
        cli.append(f"--{key}")
        if value is not True:
            cli.append(str(value))
    return cli


def accelerate_launcher(num_processes: int):
    return [sys.executable, "-m", "accelerate.commands.launch", "--num_processes", str(num_processes),
            "--num_machines", "1", "--mixed_precision", "no", "--dynamo_backend", "no",
            *(["--multi_gpu"] if num_processes > 1 else [])]


class Runner:
    def __init__(self, args):
        self.args = args
        self.done_dir = os.path.join(args.work_dir, ".done")
        self.log_dir = os.path.join(args.work_dir, "logs")
        os.makedirs(self.done_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def is_done(self, name):
        return os.path.exists(os.path.join(self.done_dir, name))

    def mark_done(self, name):
        if not self.args.dry_run:
            open(os.path.join(self.done_dir, name), "w").close()

    def _popen(self, name, cmd, gpus):
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=gpus)
        log_path = os.path.join(self.log_dir, f"{name}.log")
        print(f"[run] {name} (GPU {gpus}) -> {log_path}\n      CUDA_VISIBLE_DEVICES={gpus} {shlex.join(cmd)}", flush=True)
        if self.args.dry_run:
            return None
        return subprocess.Popen(cmd, env=env, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)

    def run(self, name, cmd, gpus):
        if self.is_done(name):
            print(f"[skip] {name} (already done)")
            return
        proc = self._popen(name, cmd, gpus)
        if proc is not None and proc.wait() != 0:
            sys.exit(f"[fail] {name}: see {os.path.join(self.log_dir, name + '.log')}")
        self.mark_done(name)

    def run_parallel(self, jobs, gpus):
        """Run ``jobs = [(name, cmd), ...]`` with at most ``--jobs_per_gpu`` jobs per GPU at a time."""
        pending = []
        for name, cmd in jobs:
            if self.is_done(name):
                print(f"[skip] {name} (already done)")
            else:
                pending.append((name, cmd))
        free, running = [g for _ in range(self.args.jobs_per_gpu) for g in gpus], {}
        while pending or running:
            while pending and free:
                name, cmd = pending.pop(0)
                gpu = free.pop(0)
                running[name] = (self._popen(name, cmd, gpu), gpu)
            for name, (proc, gpu) in list(running.items()):
                code = 0 if proc is None else proc.poll()
                if code is None:
                    continue
                if code != 0:
                    for other, _ in running.values():
                        if other is not None and other.poll() is None:
                            other.terminate()
                    sys.exit(f"[fail] {name}: see {os.path.join(self.log_dir, name + '.log')}")
                self.mark_done(name)
                del running[name]
                free.append(gpu)
            if running:
                time.sleep(10)


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model = args.model or cfg["model"]
    train_file = args.train_file or cfg["train_file"]
    online = cfg.get("mode", "cached") == "online"
    gpus = [g.strip() for g in args.gpus.split(",")]

    config = AutoConfig.from_pretrained(model, trust_remote_code=cfg.get("trust_remote_code", False))
    selection = resolve_layer_config(cfg["layer_config"], getattr(config, get_arch(config).num_layers_attr))
    curriculum = cfg["curriculum"]
    phases = build_phases(selection, alpha=curriculum.get("alpha", 0.035), beta=curriculum.get("beta", 0.0))

    work = os.path.abspath(args.work_dir)
    layer_dir, stacked_dir = os.path.join(work, "layers"), os.path.join(work, "stacked")
    common = {"model": model, "layer_config": cfg["layer_config"], "use_flash_attn": cfg.get("use_flash_attn", False),
              "trust_remote_code": cfg.get("trust_remote_code", False)}
    run_name = cfg.get("run_name", os.path.basename(work))

    print(f"Teacher layers kept: {selection}")
    print(f"Curriculum: exponential schedule, alpha = {curriculum.get('alpha', 0.035)}"
          f" -> phase epochs {[phase.epochs for phase in phases]}")
    for phase in phases:
        print(f"Phase {phase.index}: train layers {phase.active_layers} for {phase.epochs} epochs"
              f"{' (inputs from student)' if phase.from_student else ''}")

    runner = Runner(args)
    for phase in phases:
        if phase.index < args.start_phase or args.post_train_only:
            continue
        p = phase.index

        # 1. features
        if not online:
            feature_dir = os.path.join(work, "features", "student" if phase.from_student else "teacher")
            extract = {**common, **cfg.get("extract", {}), "train_file": train_file, "out_dir": feature_dir,
                       "student": stacked_dir if phase.from_student else None,
                       "active_layers": ",".join(str(i) for i in phase.active_layers)}
            if not runner.is_done(f"phase{p}_extract") and os.path.isdir(feature_dir) and not args.dry_run:
                shutil.rmtree(feature_dir)  # stale features of a previous phase / interrupted run
            runner.run(f"phase{p}_extract", [sys.executable, os.path.join(SCRIPTS, "extract_features.py"),
                                             *to_cli(extract)], ",".join(gpus))

        # 2. layer-wise training of the active layers
        train_opts = {**common, **cfg.get("train", {}), "out_dir": layer_dir,
                      "student": stacked_dir if phase.from_student else None,
                      "log_wandb": args.log_wandb, "project": args.project, "run_name": f"{run_name}_phase{p}"}
        if online:
            script = "train_layer_online.py"
            train_opts.update(train_file=train_file, train_epoch=phase.epochs, block_size=cfg.get("block_size"))
        else:
            script = "train_layer.py"
            train_opts.update(feature_dir=feature_dir, train_epoch=phase.epochs, block_size=cfg.get("block_size"),
                              quant=cfg.get("extract", {}).get("quant", 8))
        jobs = [(f"phase{p}_layer{i}", [os.path.join(SCRIPTS, script), *to_cli({**train_opts, "layer_idx": i})])
                for i in phase.active_layers]
        if online:  # small models: one layer at a time, data-parallel over all GPUs (as in the paper experiments)
            for name, cmd in jobs:
                runner.run(name, [*accelerate_launcher(len(gpus)), *cmd], ",".join(gpus))
        else:       # one process (and GPU) per layer, layers in parallel
            runner.run_parallel([(name, [sys.executable, *cmd]) for name, cmd in jobs], gpus)
        if args.cleanup_features and not online and not args.dry_run and os.path.isdir(feature_dir):
            shutil.rmtree(feature_dir)

        # 3. stack
        stack = {**common, "layer_dir": layer_dir, "save_dir": stacked_dir, "block_size": cfg.get("block_size")}
        runner.run(f"phase{p}_stack", [sys.executable, os.path.join(SCRIPTS, "stack_layers.py"), *to_cli(stack)],
                   gpus[0])

    print(f"Compressed model: {stacked_dir}")

    if args.post_train or args.post_train_only:
        post = dict(cfg.get("post_train") or {})
        post_opts = {"model": stacked_dir, "train_file": post.pop("train_file", train_file), "out_dir": work,
                     "run_name": "post_train", "use_flash_attn": common["use_flash_attn"],
                     "trust_remote_code": common["trust_remote_code"], "log_wandb": args.log_wandb, **post}
        runner.run("post_train", [*accelerate_launcher(len(gpus)), os.path.join(SCRIPTS, "post_train.py"),
                                  *to_cli(post_opts)], ",".join(gpus))
        print(f"Post-trained model: {os.path.join(work, 'post_train')}")


if __name__ == "__main__":
    main()
