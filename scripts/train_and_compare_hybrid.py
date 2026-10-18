"""
Train BiLSTM + TCN Hybrid and Compare with Current Model.
Enforces rule: If accuracy / performance is less than present model, DO NOT update best_model.pt.
"""

import sys
import json
import shutil
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.data.preprocessing import prepare_dataset
from src.data.dataset import get_dataloaders
from src.data.imbalance import get_loss_function
from src.models.bilstm_attention import build_model
from src.training.trainer import SepsisTrainer
from src.training.metrics import compute_all_metrics, find_optimal_threshold


def main():
    print("=" * 70)
    print("  EXPERIMENT: BiLSTM + TCN Hybrid vs. Current Best Model")
    print("=" * 70)

    # 1. Prepare data (same seed & split)
    print("\n[1/4] Loading and preparing dataset (Hospital A)...")
    data = prepare_dataset(hospital="A")
    loaders = get_dataloaders(data, batch_size=config.TrainConfig.batch_size, use_weighted_sampler=True)

    # 2. Evaluate CURRENT baseline model on test set
    current_model_path = config.MODELS_DIR / "best_model_bilstm_backup.pt"
    if not current_model_path.exists():
        current_model_path = config.MODELS_DIR / "best_model.pt"

    print(f"\n[2/4] Evaluating CURRENT baseline model from: {current_model_path}")
    current_ckpt = torch.load(str(current_model_path), map_location="cpu", weights_only=False)
    m_type = current_ckpt.get("model_type", "bilstm_attention")
    m_cfg = current_ckpt.get("model_config", {})
    current_model = build_model(m_type, **m_cfg)
    current_model.load_state_dict(current_ckpt["model_state_dict"])
    current_model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_model.to(device)

    all_probs_curr, all_labels_curr = [], []
    with torch.no_grad():
        for bx, by in loaders["test"]:
            bx = bx.to(device)
            p = torch.sigmoid(current_model(bx).squeeze(-1))
            all_probs_curr.append(p.cpu().numpy())
            all_labels_curr.append(by.numpy())

    y_true_curr = np.concatenate(all_labels_curr)
    y_prob_curr = np.concatenate(all_probs_curr)

    curr_metrics_def = compute_all_metrics(y_true_curr, y_prob_curr, threshold=0.5)
    opt_th_curr, opt_f1_curr = find_optimal_threshold(y_true_curr, y_prob_curr, metric="f1")
    curr_metrics_opt = compute_all_metrics(y_true_curr, y_prob_curr, threshold=opt_th_curr)

    print(f"  Current Model: {m_type}")
    print(f"  Current AUROC:                {curr_metrics_def['auroc']:.4f}")
    print(f"  Current AUPRC:                {curr_metrics_def['auprc']:.4f}")
    print(f"  Current Accuracy (def 0.5):   {curr_metrics_def['accuracy']:.4f} ({curr_metrics_def['accuracy']*100:.1f}%)")
    print(f"  Current Optimal Threshold:    {opt_th_curr:.4f}")
    print(f"  Current Accuracy (optimal):   {curr_metrics_opt['accuracy']:.4f} ({curr_metrics_opt['accuracy']*100:.1f}%)")
    print(f"  Current F1 (optimal):         {opt_f1_curr:.4f}")

    # 3. Build and train Hybrid BiLSTM + TCN
    trial_dir = config.MODELS_DIR / "hybrid_trial"
    trial_dir.mkdir(parents=True, exist_ok=True)

    print("\n[3/4] Training NEW Hybrid BiLSTM + TCN model...")
    hybrid_model = build_model("bilstm_tcn")
    criterion = get_loss_function(loss_type="focal")
    optimizer = torch.optim.Adam(
        hybrid_model.parameters(),
        lr=config.TrainConfig.learning_rate,
        weight_decay=config.TrainConfig.weight_decay
    )

    trainer = SepsisTrainer(
        model=hybrid_model,
        criterion=criterion,
        optimizer=optimizer,
        checkpoint_dir=str(trial_dir)
    )

    history = trainer.train(
        loaders["train"],
        loaders["val"],
        max_epochs=config.TrainConfig.max_epochs
    )

    # Load best checkpoint from trial
    best_hybrid_ckpt_path = trial_dir / "best_model.pt"
    if best_hybrid_ckpt_path.exists():
        trainer.load_checkpoint(str(best_hybrid_ckpt_path))

    # Evaluate Hybrid Model on Test Set
    print("\n[4/4] Evaluating Hybrid BiLSTM + TCN model on test set...")
    all_probs_hyb, all_labels_hyb = [], []
    with torch.no_grad():
        for bx, by in loaders["test"]:
            bx = bx.to(device)
            p = torch.sigmoid(trainer.model(bx).squeeze(-1))
            all_probs_hyb.append(p.cpu().numpy())
            all_labels_hyb.append(by.numpy())

    y_true_hyb = np.concatenate(all_labels_hyb)
    y_prob_hyb = np.concatenate(all_probs_hyb)

    hyb_metrics_def = compute_all_metrics(y_true_hyb, y_prob_hyb, threshold=0.5)
    opt_th_hyb, opt_f1_hyb = find_optimal_threshold(y_true_hyb, y_prob_hyb, metric="f1")
    hyb_metrics_opt = compute_all_metrics(y_true_hyb, y_prob_hyb, threshold=opt_th_hyb)

    print("\n" + "=" * 70)
    print("  HEAD-TO-HEAD COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Metric':<28} {'Current (BiLSTM+Attn)':<22} {'Hybrid (BiLSTM+TCN)':<22}")
    print("-" * 70)
    print(f"{'AUROC':<28} {curr_metrics_def['auroc']:<22.4f} {hyb_metrics_def['auroc']:<22.4f}")
    print(f"{'AUPRC':<28} {curr_metrics_def['auprc']:<22.4f} {hyb_metrics_def['auprc']:<22.4f}")
    print(f"{'Accuracy (default 0.5)':<28} {curr_metrics_def['accuracy']*100:<21.2f}% {hyb_metrics_def['accuracy']*100:<21.2f}%")
    print(f"{'Accuracy (optimal)':<28} {curr_metrics_opt['accuracy']*100:<21.2f}% {hyb_metrics_opt['accuracy']*100:<21.2f}%")
    print(f"{'Optimal F1':<28} {opt_f1_curr:<22.4f} {opt_f1_hyb:<22.4f}")
    print(f"{'Sensitivity (Recall)':<28} {curr_metrics_opt['recall']:<22.4f} {hyb_metrics_opt['recall']:<22.4f}")
    print(f"{'Specificity':<28} {curr_metrics_opt['specificity']:<22.4f} {hyb_metrics_opt['specificity']:<22.4f}")
    print("=" * 70)

    # Decision logic requested by user:
    # "if the accuracy less then present then do not update it"
    # We compare both accuracy (at optimal and default) and AUROC
    is_better = (
        (hyb_metrics_opt['accuracy'] > curr_metrics_opt['accuracy'] or 
         (hyb_metrics_opt['accuracy'] == curr_metrics_opt['accuracy'] and hyb_metrics_def['auroc'] > curr_metrics_def['auroc']))
    )

    decision_record = {
        "current_model": {
            "type": m_type,
            "auroc": float(curr_metrics_def['auroc']),
            "auprc": float(curr_metrics_def['auprc']),
            "accuracy_default": float(curr_metrics_def['accuracy']),
            "accuracy_optimal": float(curr_metrics_opt['accuracy']),
            "f1_optimal": float(opt_f1_curr)
        },
        "hybrid_model": {
            "type": "bilstm_tcn",
            "auroc": float(hyb_metrics_def['auroc']),
            "auprc": float(hyb_metrics_def['auprc']),
            "accuracy_default": float(hyb_metrics_def['accuracy']),
            "accuracy_optimal": float(hyb_metrics_opt['accuracy']),
            "f1_optimal": float(opt_f1_hyb)
        },
        "updated": bool(is_better)
    }

    with open(config.RESULTS_DIR / "hybrid_comparison_results.json", "w") as f:
        json.dump(decision_record, f, indent=2)

    if is_better:
        print("\n>>> DECISION: Hybrid BiLSTM+TCN accuracy is GREATER than current model.")
        print(">>> UPDATING best_model.pt with Hybrid checkpoint.")
        shutil.copy(best_hybrid_ckpt_path, config.MODELS_DIR / "best_model.pt")
        print("[OK] best_model.pt updated successfully.")
    else:
        print("\n>>> DECISION: Hybrid BiLSTM+TCN accuracy is LESS THAN OR EQUAL to current model.")
        print(">>> DO NOT UPDATE rule applied. Original best_model.pt remains UNCHANGED.")
        print("[OK] Original model preserved.")


if __name__ == "__main__":
    main()
