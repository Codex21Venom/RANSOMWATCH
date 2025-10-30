# model_trainer.py
import os
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, confusion_matrix, classification_report, roc_curve
from sklearn.calibration import calibration_curve
from lightgbm import LGBMClassifier, early_stopping
from catboost import CatBoostClassifier, Pool
import xgboost as xgb
from model_utils import evaluate_classifier, save_model_artifact
import argparse

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class ModelTrainer:
    def __init__(self, random_state=42, out_dir="models"):
        self.random_state = random_state
        self.out_dir = out_dir
        self.plot_dir = os.path.join(out_dir, "plots")
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

    def _init_models(self, scale_pos_weight: Optional[float] = None):
        # LightGBM
        lgb_model = LGBMClassifier(
            n_estimators=1000,
            max_depth=-1,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=4,
            verbose=-1
        )

        # CatBoost
        cb_model = CatBoostClassifier(
            iterations=1000,
            depth=6,
            learning_rate=0.05,
            verbose=0,
            random_seed=self.random_state
        )
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=1000,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            eval_metric="logloss",
            n_jobs=4,
            early_stopping_rounds=50
        )

        return {"lightgbm": lgb_model, "catboost": cb_model, "xgboost": xgb_model}

    def create_comparison_plots(self, results: Dict, X_test: pd.DataFrame, y_test: pd.Series):
        """Create comprehensive comparison plots for all models"""
        
        # 1. ROC Curve Comparison
        plt.figure(figsize=(15, 12))
        
        plt.subplot(2, 3, 1)
        for name, result in results.items():
            model = result["model_obj"]
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                plt.plot(fpr, tpr, label=f'{name.upper()} (AUC = {roc_auc:.4f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. Precision-Recall Curve Comparison
        plt.subplot(2, 3, 2)
        for name, result in results.items():
            model = result["model_obj"]
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                pr_auc = auc(recall, precision)
                plt.plot(recall, precision, label=f'{name.upper()} (AUC = {pr_auc:.4f})', linewidth=2)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 3. Metrics Comparison Bar Chart
        plt.subplot(2, 3, 3)
        metrics_to_plot = ['roc_auc', 'pr_auc', 'f1', 'accuracy', 'precision', 'recall']
        metric_names = ['ROC-AUC', 'PR-AUC', 'F1-Score', 'Accuracy', 'Precision', 'Recall']
        
        x = np.arange(len(metrics_to_plot))
        width = 0.25
        models = list(results.keys())
        
        for i, model in enumerate(models):
            metrics = results[model]["metrics"]
            values = [metrics.get(metric, 0) for metric in metrics_to_plot]
            plt.bar(x + i*width, values, width, label=model.upper(), alpha=0.8)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Model Metrics Comparison')
        plt.xticks(x + width, metric_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')

        # 4. Confusion Matrices
        for i, (name, result) in enumerate(results.items()):
            plt.subplot(2, 3, 4 + i)
            cm = result["metrics"].get('confusion_matrix', [[0, 0], [0, 0]])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Predicted 0', 'Predicted 1'],
                       yticklabels=['Actual 0', 'Actual 1'])
            plt.title(f'{name.upper()} Confusion Matrix')
            plt.tight_layout()

        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'model_comparison_comprehensive.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def create_individual_confusion_matrices(self, results: Dict):
        """Create individual confusion matrix plots for each model"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, (name, result) in enumerate(results.items()):
            cm = result["metrics"].get('confusion_matrix', [[0, 0], [0, 0]])
            sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', ax=axes[idx],
                       xticklabels=['Predicted 0', 'Predicted 1'],
                       yticklabels=['Actual 0', 'Actual 1'],
                       cbar_kws={'shrink': 0.8})
            axes[idx].set_title(f'{name.upper()} - Confusion Matrix', fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('Predicted Label', fontsize=12)
            axes[idx].set_ylabel('True Label', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'confusion_matrices_individual.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def create_calibration_plots(self, results: Dict, X_test: pd.DataFrame, y_test: pd.Series):
        """Create calibration plots for model probability assessment"""
        plt.figure(figsize=(10, 8))
        
        for name, result in results.items():
            model = result["model_obj"]
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_pred_proba, n_bins=10)
                plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f'{name.upper()}', linewidth=2)
        
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Plots (Reliability Curves)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plot_dir, 'calibration_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def create_feature_importance_plot(self, best_model, feature_names, best_model_name):
        """Create feature importance plot for the best model"""
        plt.figure(figsize=(12, 8))
        
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Plot top 20 features
            top_n = min(20, len(feature_names))
            plt.barh(range(top_n), importances[indices[:top_n]][::-1])
            plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]][::-1])
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Feature Importances - {best_model_name.upper()}')
            plt.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f'feature_importance_{best_model_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def create_performance_radar_chart(self, results: Dict):
        """Create radar chart for model performance comparison"""
        metrics = ['roc_auc', 'pr_auc', 'f1', 'precision', 'recall', 'accuracy']
        metric_labels = ['ROC-AUC', 'PR-AUC', 'F1-Score', 'Precision', 'Recall', 'Accuracy']
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for name, result in results.items():
            values = [result["metrics"].get(metric, 0) for metric in metrics]
            values += values[:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=name.upper())
            ax.fill(angles, values, alpha=0.1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Radar Chart', size=14, fontweight='bold')
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        plt.savefig(os.path.join(self.plot_dir, 'performance_radar_chart.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def create_training_summary_plot(self, results: Dict):
        """Create a comprehensive summary plot"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Main metrics comparison
        main_metrics = ['roc_auc', 'pr_auc', 'f1', 'accuracy']
        metric_names = ['ROC-AUC', 'PR-AUC', 'F1-Score', 'Accuracy']
        
        models = list(results.keys())
        x = np.arange(len(main_metrics))
        
        for i, model in enumerate(models):
            metrics = results[model]["metrics"]
            values = [metrics.get(metric, 0) for metric in main_metrics]
            axes[0, 0].bar(x + i*0.2, values, 0.2, label=model.upper(), alpha=0.8)
        
        axes[0, 0].set_xticks(x + 0.2)
        axes[0, 0].set_xticklabels(metric_names)
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Key Metrics Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Precision-Recall comparison
        precision_recall_data = []
        for model in models:
            metrics = results[model]["metrics"]
            precision_recall_data.append([metrics.get('precision', 0), metrics.get('recall', 0)])
        
        precision_recall_df = pd.DataFrame(precision_recall_data, 
                                         index=[m.upper() for m in models],
                                         columns=['Precision', 'Recall'])
        precision_recall_df.plot(kind='bar', ax=axes[0, 1], alpha=0.8)
        axes[0, 1].set_title('Precision vs Recall')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Model ranking
        weighted_scores = []
        for model in models:
            metrics = results[model]["metrics"]
            score = (metrics.get('roc_auc', 0) * 0.4 + 
                    metrics.get('pr_auc', 0) * 0.4 + 
                    metrics.get('f1', 0) * 0.2)
            weighted_scores.append(score)
        
        ranking_df = pd.DataFrame({
            'Model': [m.upper() for m in models],
            'Weighted Score': weighted_scores
        }).sort_values('Weighted Score', ascending=True)
        
        axes[1, 0].barh(ranking_df['Model'], ranking_df['Weighted Score'], color='lightcoral', alpha=0.8)
        axes[1, 0].set_xlabel('Weighted Score (ROC-AUC 40% + PR-AUC 40% + F1 20%)')
        axes[1, 0].set_title('Model Ranking by Weighted Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Error analysis
        error_rates = []
        for model in models:
            cm = results[model]["metrics"].get('confusion_matrix', [[0, 0], [0, 0]])
            total = np.sum(cm)
            errors = total - np.trace(cm)
            error_rates.append(errors / total)
        
        axes[1, 1].bar([m.upper() for m in models], error_rates, color='orange', alpha=0.8)
        axes[1, 1].set_ylabel('Error Rate')
        axes[1, 1].set_title('Model Error Rates')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'training_summary_dashboard.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def compare_and_train(self, X: pd.DataFrame, y: pd.Series, test_size=0.2, val_size=0.15) -> Dict:
        # Split: train+val / test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )
        # train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_size, stratify=y_trainval, random_state=self.random_state
        )

        # scale_pos_weight heuristic
        pos = int(sum(y_train == 1))
        neg = int(sum(y_train == 0))
        scale_pos_weight = (neg / pos) if pos > 0 else 1.0

        logging.info(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
        logging.info(f"Scale pos weight (neg/pos) -> {scale_pos_weight:.3f}")
        logging.info("")

        results = {}
        model_metrics = {}

        # Initialize all models
        models = self._init_models(scale_pos_weight=scale_pos_weight)
        
        # Phase 1: Train all models and collect metrics
        logging.info("🚀 TRAINING ALL MODELS")
        logging.info("=" * 60)
        
        for name, model in models.items():
            logging.info(f"📊 Training {name.upper()} ...")
            
            if name == "catboost":
                pool_train = Pool(X_train, y_train)
                pool_val = Pool(X_val, y_val)
                model.fit(pool_train, eval_set=pool_val, early_stopping_rounds=50, use_best_model=True)
            elif name == "lightgbm":
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[early_stopping(50, verbose=False)])
            else:  # XGBoost
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            metrics = evaluate_classifier(model, X_test, y_test)
            results[name] = {"metrics": metrics, "model_obj": model}
            model_metrics[name] = metrics
            
            logging.info(f"✅ {name.upper()} COMPLETED")
            logging.info(f"   ROC-AUC:    {metrics.get('roc_auc', 0):.6f}")
            logging.info(f"   PR-AUC:     {metrics.get('pr_auc', 0):.6f}")
            logging.info(f"   F1-Score:   {metrics.get('f1', 0):.6f}")
            logging.info(f"   Accuracy:   {metrics.get('accuracy', 0):.6f}")
            logging.info("-" * 50)

        # Phase 2: Create comprehensive plots
        logging.info("\n📈 CREATING COMPREHENSIVE VISUALIZATIONS")
        logging.info("=" * 60)
        
        self.create_comparison_plots(results, X_test, y_test)
        self.create_individual_confusion_matrices(results)
        self.create_calibration_plots(results, X_test, y_test)
        self.create_performance_radar_chart(results)
        self.create_training_summary_plot(results)
        
        logging.info("✅ All visualizations created and saved to plots/ directory")

        # Phase 3: Display model comparison
        logging.info("\n" + "=" * 70)
        logging.info("📊 MODEL PERFORMANCE COMPARISON")
        logging.info("=" * 70)
        
        comparison_data = []
        for name, metrics in model_metrics.items():
            comparison_data.append({
                'Model': name.upper(),
                'ROC-AUC': f"{metrics.get('roc_auc', 0):.6f}",
                'PR-AUC': f"{metrics.get('pr_auc', 0):.6f}",
                'F1-Score': f"{metrics.get('f1', 0):.6f}",
                'Accuracy': f"{metrics.get('accuracy', 0):.6f}",
                'Precision': f"{metrics.get('precision', 0):.6f}",
                'Recall': f"{metrics.get('recall', 0):.6f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        logging.info("\n" + comparison_df.to_string(index=False))
        
        # Phase 4: Select best model
        logging.info("\n" + "=" * 70)
        logging.info("🏆 SELECTING BEST MODEL")
        logging.info("=" * 70)
        
        def score_key(entry):
            m = entry["metrics"]
            roc_auc = m.get("roc_auc", 0) or 0.0
            pr_auc = m.get("pr_auc", 0) or 0.0
            f1 = m.get("f1", 0) or 0.0
            return (roc_auc * 0.4 + pr_auc * 0.4 + f1 * 0.2)

        best_name = max(results.keys(), key=lambda k: score_key(results[k]))
        best_entry = results[best_name]
        best_model = best_entry["model_obj"]
        best_metrics = best_entry["metrics"]

        logging.info(f"📊 Selection Criteria: Weighted Score (ROC-AUC 40% + PR-AUC 40% + F1 20%)")
        logging.info("")
        
        for name, result in results.items():
            m = result["metrics"]
            roc_auc = m.get("roc_auc", 0) or 0.0
            pr_auc = m.get("pr_auc", 0) or 0.0
            f1 = m.get("f1", 0) or 0.0
            weighted_score = roc_auc * 0.4 + pr_auc * 0.4 + f1 * 0.2
            marker = " 🏆" if name == best_name else ""
            logging.info(f"   {name.upper():<12} | Weighted Score: {weighted_score:.6f}{marker}")

        logging.info(f"\n🎯 SELECTED BEST MODEL: {best_name.upper()}")

        # Create feature importance plot for best model
        self.create_feature_importance_plot(best_model, X.columns.tolist(), best_name)

        # Phase 5: Save the best model
        logging.info("\n" + "=" * 70)
        logging.info("💾 SAVING BEST MODEL")
        logging.info("=" * 70)
        
        model_path, meta_path = save_model_artifact(
            best_model, X.columns.tolist(), out_dir=self.out_dir, name_prefix=f"best_{best_name}"
        )

        summary = {
            "best_model_name": best_name,
            "best_model_path": model_path,
            "best_model_metrics": best_metrics,
            "all_results": {k: v["metrics"] for k, v in results.items()},
            "model_comparison": comparison_data
        }

        summary_path = os.path.join(self.out_dir, "comparison_summary.json")
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=2)

        logging.info(f"✅ Best model saved to: {model_path}")
        logging.info(f"✅ Comparison summary saved to: {summary_path}")
        logging.info(f"✅ All plots saved to: {self.plot_dir}")
        
        # Final best model details
        logging.info("\n" + "=" * 70)
        logging.info("🎉 BEST MODEL DETAILS")
        logging.info("=" * 70)
        logging.info(f"Model:          {best_name.upper()}")
        logging.info(f"ROC-AUC:        {best_metrics.get('roc_auc', 0):.6f}")
        logging.info(f"PR-AUC:         {best_metrics.get('pr_auc', 0):.6f}")
        logging.info(f"F1-Score:       {best_metrics.get('f1', 0):.6f}")
        logging.info(f"Accuracy:       {best_metrics.get('accuracy', 0):.6f}")
        logging.info(f"Precision:      {best_metrics.get('precision', 0):.6f}")
        logging.info(f"Recall:         {best_metrics.get('recall', 0):.6f}")
        
        cm = best_metrics.get('confusion_matrix', [[0, 0], [0, 0]])
        logging.info(f"Confusion Matrix:")
        logging.info(f"                Predicted 0   Predicted 1")
        logging.info(f"  Actual 0:      {cm[0][0]:>8}      {cm[0][1]:>8}")
        logging.info(f"  Actual 1:      {cm[1][0]:>8}      {cm[1][1]:>8}")

        return summary


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=None, help="Optional single CSV containing features+Label")
    parser.add_argument("--benign", default="C:/Users/AgentxVenom/Documents/Soham Goswami/New_Ransomwatch/data/benign.csv", help="Path to benign.csv")
    parser.add_argument("--ransom", default="C:/Users/AgentxVenom/Documents/Soham Goswami/New_Ransomwatch/data/ransom.csv", help="Path to ransom.csv")
    parser.add_argument("--out-dir", default="models", help="Output models directory")
    args = parser.parse_args()

    trainer = ModelTrainer(out_dir=args.out_dir)

    if args.data_path:
        df = pd.read_csv(args.data_path)
        if "Label" not in df.columns:
            raise SystemExit("Provided data must include 'Label' column.")
        X = df.drop(columns=["Label"]).select_dtypes(include=["number"])
        y = df["Label"]
        summary = trainer.compare_and_train(X, y)
    else:
        try:
            from data_preprocessing import load_and_preprocess_data
            X_train, X_test, y_train, y_test, pipeline, feature_names = load_and_preprocess_data(
                benign_path=args.benign, ransom_path=args.ransom)
            X = pd.concat([X_train, X_test], ignore_index=True)
            y = pd.concat([y_train, y_test], ignore_index=True)
            summary = trainer.compare_and_train(X, y)
        except Exception as e:
            raise SystemExit(f"Failed to load data via preprocessing: {e}")

    # Enhanced final output
    print("\n" + "=" * 80)
    print("🎯 TRAINING COMPLETED - FINAL SUMMARY")
    print("=" * 80)
    
    print("\n📊 ALL MODEL PERFORMANCES:")
    print("-" * 50)
    all_results = summary['all_results']
    for model_name, metrics in all_results.items():
        print(f"   {model_name.upper():<12} | ROC-AUC: {metrics.get('roc_auc', 0):.6f} | "
              f"PR-AUC: {metrics.get('pr_auc', 0):.6f} | F1: {metrics.get('f1', 0):.6f}")
    
    print("\n🏆 BEST MODEL SELECTED:")
    print("-" * 50)
    print(f"   Model:    {summary['best_model_name'].upper()}")
    print(f"   ROC-AUC:  {summary['best_model_metrics'].get('roc_auc', 0):.6f}")
    print(f"   PR-AUC:   {summary['best_model_metrics'].get('pr_auc', 0):.6f}")
    print(f"   F1-Score: {summary['best_model_metrics'].get('f1', 0):.6f}")
    print(f"   Accuracy: {summary['best_model_metrics'].get('accuracy', 0):.6f}")
    
    print(f"\n💾 Model saved to: {summary['best_model_path']}")
    print(f"📄 Summary saved to: {os.path.join(args.out_dir, 'comparison_summary.json')}")
    print(f"📊 Plots saved to: {os.path.join(args.out_dir, 'plots')}")
    print("=" * 80)


if __name__ == "__main__":
    main_cli()