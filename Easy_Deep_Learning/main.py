"""Easy Deep Learning CLI entrypoint."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"
os.environ["KMP_USE_SHM"] = "0"

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from Easy_Deep_Learning.core.logging_utils import configure_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser with train/test subcommands."""
    parser = argparse.ArgumentParser(description="Easy Deep Learning")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train a model and save artifacts")
    train.add_argument("--data", type=Path, required=True, help="Training CSV path")
    train.add_argument("--target-column", type=str, required=True, help="Target column name")
    train.add_argument("--task-type", choices=["classification", "regression"], required=True)
    train.add_argument("--model-type", choices=["auto", "dnn", "xgboost", "rf", "svm", "knn", "lr", "gbm"], default="dnn")
    train.add_argument("--config", type=Path, default=Path("config/model_config.yaml"))
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--model-params", type=str, default="{}", help="JSON string of model hyperparameters")

    test = subparsers.add_parser("test", help="Evaluate a saved run on new data")
    test.add_argument("--from-run", type=str, required=True, help="Run ID under runs/")
    test.add_argument("--data", type=Path, required=True, help="Test CSV path")
    test.add_argument("--target-column", type=str, default=None, help="Optional override target column")

    img_train = subparsers.add_parser("image-train", help="Train CNN on image dataset")
    img_train.add_argument("--dataset", choices=["MNIST", "FashionMNIST", "CIFAR10", "SVHN", "EMNIST"], required=True)
    img_train.add_argument("--epochs", type=int, default=5)
    img_train.add_argument("--lr", type=float, default=1e-3)
    img_train.add_argument("--batch-size", type=int, default=64)
    img_train.add_argument("--seed", type=int, default=42)
    img_train.add_argument("--data-dir", type=Path, default=Path("/tmp/easy_dl"))
    img_train.add_argument("--model-arch", choices=["cnn", "resnet18"], default="cnn")

    img_test = subparsers.add_parser("image-test", help="Evaluate saved CNN run")
    img_test.add_argument("--from-run", type=str, required=True)

    txt_train = subparsers.add_parser("text-train", help="Train RNN on text dataset")
    txt_train.add_argument("--dataset", choices=["AG_NEWS_SAMPLE", "SST2_SAMPLE", "TREC_SAMPLE"], default="AG_NEWS_SAMPLE")
    txt_train.add_argument("--data", type=Path, default=None, help="Optional CSV path for text dataset")
    txt_train.add_argument("--text-column", type=str, default="text")
    txt_train.add_argument("--label-column", type=str, default="label")
    txt_train.add_argument("--max-vocab", type=int, default=5000)
    txt_train.add_argument("--max-len", type=int, default=100)
    txt_train.add_argument("--epochs", type=int, default=3)
    txt_train.add_argument("--lr", type=float, default=1e-3)
    txt_train.add_argument("--batch-size", type=int, default=64)
    txt_train.add_argument("--seed", type=int, default=42)
    txt_train.add_argument("--data-dir", type=Path, default=Path("/tmp/easy_dl"))
    txt_train.add_argument("--stopwords", action="store_true")
    txt_train.add_argument("--ngram", type=int, default=1)
    txt_train.add_argument("--bpe", action="store_true")
    txt_train.add_argument("--bpe-vocab-size", type=int, default=200)
    txt_train.add_argument("--model-arch", choices=["gru", "lstm", "textcnn", "transformer"], default="gru")

    txt_test = subparsers.add_parser("text-test", help="Evaluate saved RNN run")
    txt_test.add_argument("--from-run", type=str, required=True)
    txt_test.add_argument("--data", type=Path, default=None)

    automl = subparsers.add_parser("automl", help="Train multiple models and build a leaderboard")
    automl.add_argument("--data", type=Path, required=True, help="Training CSV path")
    automl.add_argument("--target-column", type=str, required=True, help="Target column name")
    automl.add_argument("--task-type", choices=["classification", "regression"], required=True)
    automl.add_argument("--config", type=Path, default=Path("config/model_config.yaml"))
    automl.add_argument("--seed", type=int, default=42)
    automl.add_argument("--max-models", type=int, default=6)

    tune = subparsers.add_parser("tune", help="Auto tune hyperparameters for a model")
    tune.add_argument("--data", type=Path, required=True, help="Training CSV path")
    tune.add_argument("--target-column", type=str, required=True, help="Target column name")
    tune.add_argument("--task-type", choices=["classification", "regression"], required=True)
    tune.add_argument("--model-type", choices=["xgboost", "rf", "svm", "knn", "lr", "gbm"], required=True)
    tune.add_argument("--config", type=Path, default=Path("config/model_config.yaml"))
    tune.add_argument("--seed", type=int, default=42)
    tune.add_argument("--max-trials", type=int, default=10)

    agent = subparsers.add_parser("agent", help="Run tool-using agent on a dataset")
    agent.add_argument("--data", type=Path, required=True, help="CSV path")
    agent.add_argument("--target-column", type=str, required=True)
    agent.add_argument("--task-type", choices=["classification", "regression"], required=True)

    rag = subparsers.add_parser("rag", help="Run lightweight RAG on a text corpus")
    rag.add_argument("--query", type=str, required=True)
    rag.add_argument("--docs", type=Path, required=True, help="Text file containing docs (one per line)")
    rag.add_argument("--top-k", type=int, default=3)
    rag.add_argument("--chunk-size", type=int, default=400)
    rag.add_argument("--overlap", type=int, default=80)

    compare = subparsers.add_parser("compare", help="Compare multiple runs and generate a report")
    compare.add_argument("--run-ids", type=str, required=True, help="Comma-separated run IDs")

    cv = subparsers.add_parser("cv", help="Run cross-validation report")
    cv.add_argument("--data", type=Path, required=True, help="Training CSV path")
    cv.add_argument("--target-column", type=str, required=True, help="Target column name")
    cv.add_argument("--task-type", choices=["classification", "regression"], required=True)
    cv.add_argument("--model-type", choices=["dnn", "xgboost", "rf", "svm", "knn", "lr", "gbm"], required=True)
    cv.add_argument("--seed", type=int, default=42)
    cv.add_argument("--folds", type=int, default=5)
    cv.add_argument("--model-params", type=str, default="{}")

    ft_llm = subparsers.add_parser("finetune-llm", help="Fine-tune a causal LLM with LoRA")
    ft_llm.add_argument("--data", type=Path, required=True, help="JSONL/CSV path")
    ft_llm.add_argument("--prompt-column", type=str, default="prompt")
    ft_llm.add_argument("--completion-column", type=str, default="completion")
    ft_llm.add_argument("--model-name", type=str, default="distilgpt2")
    ft_llm.add_argument("--epochs", type=int, default=1)
    ft_llm.add_argument("--lr", type=float, default=2e-5)
    ft_llm.add_argument("--batch-size", type=int, default=2)
    ft_llm.add_argument("--seed", type=int, default=42)
    ft_llm.add_argument("--max-length", type=int, default=256)
    ft_llm.add_argument("--lora-r", type=int, default=8)
    ft_llm.add_argument("--lora-alpha", type=int, default=16)
    ft_llm.add_argument("--lora-dropout", type=float, default=0.05)
    ft_llm.add_argument("--eval-size", type=float, default=0.1)

    ft_img = subparsers.add_parser("finetune-image", help="Fine-tune a pretrained image model on a folder dataset")
    ft_img.add_argument("--data-dir", type=Path, required=True, help="Folder with class subfolders")
    ft_img.add_argument("--model-arch", type=str, default="resnet18")
    ft_img.add_argument("--epochs", type=int, default=5)
    ft_img.add_argument("--lr", type=float, default=1e-4)
    ft_img.add_argument("--batch-size", type=int, default=16)
    ft_img.add_argument("--seed", type=int, default=42)
    ft_img.add_argument("--pretrained", action="store_true")
    ft_img.add_argument("--freeze-backbone", action="store_true")
    ft_img.add_argument("--val-split", type=float, default=0.2)

    ft_text = subparsers.add_parser("finetune-text", help="Fine-tune a text transformer on a CSV dataset")
    ft_text.add_argument("--data", type=Path, required=True, help="CSV path")
    ft_text.add_argument("--text-column", type=str, default="text")
    ft_text.add_argument("--label-column", type=str, default="label")
    ft_text.add_argument("--model-name", type=str, default="bert-base-uncased")
    ft_text.add_argument("--epochs", type=int, default=2)
    ft_text.add_argument("--lr", type=float, default=2e-5)
    ft_text.add_argument("--batch-size", type=int, default=8)
    ft_text.add_argument("--seed", type=int, default=42)

    reg_list = subparsers.add_parser("registry-list", help="List model registry entries and tags")
    reg_list.add_argument("--runs-dir", type=Path, default=Path("runs"))

    reg_resolve = subparsers.add_parser("registry-resolve", help="Resolve a registry tag to run_id")
    reg_resolve.add_argument("--tag", type=str, required=True)
    reg_resolve.add_argument("--runs-dir", type=Path, default=Path("runs"))

    reg_promote = subparsers.add_parser("registry-promote", help="Promote a run to production tag")
    reg_promote.add_argument("--run-id", type=str, required=True)
    reg_promote.add_argument("--run-type", type=str, default=None)
    reg_promote.add_argument("--task-type", type=str, default=None)
    reg_promote.add_argument("--model-type", type=str, default=None)
    reg_promote.add_argument("--runs-dir", type=Path, default=Path("runs"))

    return parser


def main() -> None:
    """Execute CLI command."""
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)

    if args.command == "train":
        from Easy_Deep_Learning.core.workflows import train_and_save
        try:
            model_params = json.loads(args.model_params)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid --model-params JSON: {exc}") from exc

        result = train_and_save(
            data_path=args.data,
            config_path=args.config,
            target_column=args.target_column,
            task_type=args.task_type,
            model_type=args.model_type,
            seed=args.seed,
            model_params=model_params,
        )
        payload = {
            "project": "Easy Deep Learning",
            "mode": "train",
            "run_id": result.run_id,
            "run_path": str(result.run_path.resolve()),
            "metrics": result.metrics,
        }
        standard_meta = result.run_path / "run_metadata.standard.json"
        if standard_meta.exists():
            payload["run_metadata_standard"] = str(standard_meta.resolve())
        print(json.dumps(payload, indent=2))
        return

    if args.command == "test":
        from Easy_Deep_Learning.core.workflows import test_from_run
        payload = test_from_run(
            run_id=args.from_run,
            test_data_path=args.data,
            target_column=args.target_column,
            save_artifacts=True,
        )
        print(json.dumps(payload, indent=2))
        return

    if args.command == "image-train":
        from Easy_Deep_Learning.core.torch_workflows import train_cnn_image

        result = train_cnn_image(
            dataset_name=args.dataset,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            seed=args.seed,
            data_dir=args.data_dir,
            model_arch=args.model_arch,
        )
        payload = {
            "project": "Easy Deep Learning",
            "mode": "image-train",
            "run_id": result.run_id,
            "run_path": str(result.run_path.resolve()),
            "metrics": result.metrics,
        }
        print(json.dumps(payload, indent=2))
        return

    if args.command == "image-test":
        from Easy_Deep_Learning.core.torch_workflows import test_cnn_image

        payload = test_cnn_image(args.from_run)
        print(json.dumps(payload, indent=2))
        return

    if args.command == "text-train":
        from Easy_Deep_Learning.core.torch_workflows import train_rnn_text

        result = train_rnn_text(
            dataset_name=args.dataset,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            seed=args.seed,
            data_dir=args.data_dir,
            data_path=args.data,
            text_column=args.text_column,
            label_column=args.label_column,
            max_vocab=args.max_vocab,
            max_len=args.max_len,
            stopwords=args.stopwords,
            ngram=args.ngram,
            bpe=args.bpe,
            bpe_vocab_size=args.bpe_vocab_size,
            model_arch=args.model_arch,
        )
        payload = {
            "project": "Easy Deep Learning",
            "mode": "text-train",
            "run_id": result.run_id,
            "run_path": str(result.run_path.resolve()),
            "metrics": result.metrics,
        }
        print(json.dumps(payload, indent=2))
        return

    if args.command == "text-test":
        from Easy_Deep_Learning.core.torch_workflows import test_rnn_text

        payload = test_rnn_text(args.from_run, data_path=args.data)
        print(json.dumps(payload, indent=2))
        return

    if args.command == "automl":
        from Easy_Deep_Learning.core.workflows import run_leaderboard

        payload = run_leaderboard(
            data_path=args.data,
            config_path=args.config,
            target_column=args.target_column,
            task_type=args.task_type,
            seed=args.seed,
            max_models=args.max_models,
        )
        print(json.dumps(payload, indent=2))
        return

    if args.command == "tune":
        from Easy_Deep_Learning.core.workflows import auto_tune_and_train

        result = auto_tune_and_train(
            data_path=args.data,
            config_path=args.config,
            target_column=args.target_column,
            task_type=args.task_type,
            model_type=args.model_type,
            seed=args.seed,
            max_trials=args.max_trials,
        )
        payload = {
            "project": "Easy Deep Learning",
            "mode": "tune",
            "run_id": result.run_id,
            "run_path": str(result.run_path.resolve()),
            "metrics": result.metrics,
        }
        print(json.dumps(payload, indent=2))
        return

    if args.command == "agent":
        from Easy_Deep_Learning.agents.tool_agent import AgentInput, ToolUsingAgent, make_default_tools

        agent = ToolUsingAgent()
        result = agent.run(
            AgentInput(
                dataset_path=args.data,
                target_column=args.target_column,
                task_type=args.task_type,
            ),
            tools=make_default_tools(),
        )
        print(json.dumps({
            "tool_calls": [call.__dict__ for call in result.tool_calls],
            "tool_results": [res.__dict__ for res in result.tool_results],
            "summary": result.final_summary,
        }, indent=2))
        return

    if args.command == "rag":
        from Easy_Deep_Learning.core.rag import run_rag

        docs = [line.strip() for line in args.docs.read_text(encoding="utf-8").splitlines() if line.strip()]
        result = run_rag(
            query=args.query,
            docs=docs,
            top_k=args.top_k,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
        print(json.dumps({
            "query": result.query,
            "answer": result.answer,
            "contexts": result.contexts,
            "scores": result.scores,
            "eval": result.eval,
        }, indent=2))
        return

    if args.command == "compare":
        from Easy_Deep_Learning.core.compare import generate_compare_report

        run_ids = [rid.strip() for rid in args.run_ids.split(",") if rid.strip()]
        payload = generate_compare_report(run_ids)
        print(json.dumps(payload, indent=2))
        return

    if args.command == "cv":
        from Easy_Deep_Learning.core.workflows import cross_validate_and_report
        try:
            model_params = json.loads(args.model_params)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid --model-params JSON: {exc}") from exc

        payload = cross_validate_and_report(
            data_path=args.data,
            target_column=args.target_column,
            task_type=args.task_type,
            model_type=args.model_type,
            seed=args.seed,
            folds=args.folds,
            model_params=model_params,
        )
        print(json.dumps(payload, indent=2))
        return

    if args.command == "finetune-image":
        from Easy_Deep_Learning.core.finetune import finetune_image_folder

        result = finetune_image_folder(
            data_dir=args.data_dir,
            model_arch=args.model_arch,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            seed=args.seed,
            use_pretrained=bool(args.pretrained),
            freeze_backbone=bool(args.freeze_backbone),
            val_split=float(args.val_split),
        )
        payload = {
            "project": "Easy Deep Learning",
            "mode": "finetune-image",
            "run_id": result.run_id,
            "run_path": str(result.run_path.resolve()),
            "metrics": result.metrics,
        }
        print(json.dumps(payload, indent=2))
        return

    if args.command == "finetune-text":
        from Easy_Deep_Learning.core.finetune import finetune_text_transformer

        result = finetune_text_transformer(
            data_path=args.data,
            text_column=args.text_column,
            label_column=args.label_column,
            model_name=args.model_name,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        payload = {
            "project": "Easy Deep Learning",
            "mode": "finetune-text",
            "run_id": result.run_id,
            "run_path": str(result.run_path.resolve()),
            "metrics": result.metrics,
        }
        print(json.dumps(payload, indent=2))
        return

    if args.command == "finetune-llm":
        from Easy_Deep_Learning.core.llm_finetune import finetune_llm_lora

        result = finetune_llm_lora(
            data_path=args.data,
            model_name=args.model_name,
            prompt_column=args.prompt_column,
            completion_column=args.completion_column,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            seed=args.seed,
            max_length=args.max_length,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            eval_size=args.eval_size,
        )
        payload = {
            "project": "Easy Deep Learning",
            "mode": "finetune-llm",
            "run_id": result.run_id,
            "run_path": str(result.run_path.resolve()),
            "metrics": result.metrics,
        }
        print(json.dumps(payload, indent=2))
        return

    if args.command == "registry-list":
        from Easy_Deep_Learning.core.model_registry_layer import ModelRegistry

        payload = ModelRegistry(runs_dir=args.runs_dir).list_registry()
        print(json.dumps(payload, indent=2))
        return

    if args.command == "registry-resolve":
        from Easy_Deep_Learning.core.model_registry_layer import ModelRegistry

        run_id = ModelRegistry(runs_dir=args.runs_dir).resolve_tag(args.tag)
        print(json.dumps({"tag": args.tag, "run_id": run_id}, indent=2))
        return

    if args.command == "registry-promote":
        from Easy_Deep_Learning.core.model_registry_layer import ModelRegistry

        run_type = args.run_type
        task_type = args.task_type
        model_type = args.model_type
        run_path = args.runs_dir / args.run_id
        meta_path = run_path / "run_metadata.standard.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            run_type = run_type or meta.get("run_type")
            task_type = task_type or meta.get("task_type")
            model_type = model_type or meta.get("model_type")
        if not run_type or not task_type:
            raise SystemExit("run_type and task_type are required if run metadata is unavailable.")

        tag = ModelRegistry(runs_dir=args.runs_dir).promote_to_production(
            run_id=args.run_id,
            run_type=run_type,
            task_type=task_type,
            model_type=model_type,
        )
        print(json.dumps({"run_id": args.run_id, "production_tag": tag}, indent=2))
        return

    parser.error("Unknown command")


if __name__ == "__main__":
    main()
