import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import Dict, List, Optional
from .interfaces import IVisualizer

class MatplotlibVisualizer(IVisualizer):
    """
    Visualization service for generating ML model performance plots.
    Follows Single Responsibility Principle and integrates with existing pipeline.
    """
    
    def __init__(self, output_dir: str = "visualizations"):
        """
        Initialize visualizer with output directory.
        
        Args:
            output_dir: Directory to save generated plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set seaborn style for modern aesthetics
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
    
    def generate_performance_comparison(self, results: pd.DataFrame) -> str:
        """
        Generate bar chart comparing all metrics across models.
        
        Args:
            results: DataFrame with models as index and metrics as columns
            
        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC', 'FNR']
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#e67e22']
        
        for idx, (metric, color) in enumerate(zip(metrics, colors)):
            ax = axes[idx // 3, idx % 3]
            
            if metric in results.columns:
                results[metric].plot(kind='bar', ax=ax, color=color, alpha=0.8)
                ax.set_title(metric.replace('_', ' '), fontweight='bold')
                ax.set_ylabel('Score')
                ax.set_xlabel('')
                ax.set_ylim(0, 1)
                ax.grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.3f', padding=3)
                
                # Rotate x-axis labels
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        output_path = self.output_dir / "performance_comparison.png"
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Performance comparison saved to: {output_path}")
        return str(output_path)
    
    def generate_confusion_matrices(self, model_predictions: Dict[str, dict]) -> str:
        """
        Generate grid of confusion matrices for all models.
        
        Args:
            model_predictions: Dict with model names as keys and 
                             {'y_true', 'y_pred'} as values
                             
        Returns:
            Path to saved plot
        """
        n_models = len(model_predictions)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')
        
        for ax, (model_name, pred_data) in zip(axes, model_predictions.items()):
            cm = confusion_matrix(pred_data['y_true'], pred_data['y_pred'])
            
            # Calculate percentages
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            
            # Create annotations with both count and percentage
            annot = np.array([[f'{count}\n({pct:.1f}%)' 
                             for count, pct in zip(row_counts, row_pcts)]
                            for row_counts, row_pcts in zip(cm, cm_percent)])
            
            sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', 
                       cbar=True, ax=ax, square=True,
                       xticklabels=['No Claim', 'Claim'],
                       yticklabels=['No Claim', 'Claim'])
            
            ax.set_title(model_name, fontweight='bold', pad=10)
            ax.set_ylabel('Actual', fontweight='bold')
            ax.set_xlabel('Predicted', fontweight='bold')
        
        plt.tight_layout()
        output_path = self.output_dir / "confusion_matrices.png"
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Confusion matrices saved to: {output_path}")
        return str(output_path)
    
    def generate_roc_curves(self, model_predictions: Dict[str, dict]) -> str:
        """
        Generate ROC curves for all models on the same plot.
        
        Args:
            model_predictions: Dict with model names as keys and 
                             {'y_true', 'y_pred_proba'} as values
                             
        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for (model_name, pred_data), color in zip(model_predictions.items(), colors):
            if 'y_pred_proba' in pred_data:
                fpr, tpr, _ = roc_curve(pred_data['y_true'], pred_data['y_pred_proba'])
                roc_auc = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, color=color, lw=2, 
                       label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontweight='bold')
        ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", frameon=True, shadow=True)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "roc_curves.png"
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        print(f"✓ ROC curves saved to: {output_path}")
        return str(output_path)
    
    def generate_feature_importance(self, model, feature_names: List[str], 
                                   model_name: str, top_n: int = 20) -> Optional[str]:
        """
        Generate feature importance plot for tree-based models.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            model_name: Name of the model for title
            top_n: Number of top features to display
            
        Returns:
            Path to saved plot or None if model doesn't support feature importance
        """
        if not hasattr(model, 'feature_importances_'):
            print(f"⊗ {model_name} does not support feature importance")
            return None
        
        importance = model.feature_importances_
        indices = np.argsort(importance)[-top_n:]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.barh(range(len(indices)), importance[indices], color='#3498db', alpha=0.8)
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance Score', fontweight='bold')
        ax.set_title(f'Top {top_n} Feature Importances - {model_name}', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        safe_name = model_name.lower().replace(' ', '_')
        output_path = self.output_dir / f"feature_importance_{safe_name}.png"
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Feature importance saved to: {output_path}")
        return str(output_path)
    
    def generate_class_distribution(self, y_before: pd.Series, y_after: pd.Series) -> str:
        """
        Generate before/after comparison of class distribution.
        
        Args:
            y_before: Target variable before oversampling
            y_after: Target variable after oversampling
            
        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Class Distribution: Before vs After Oversampling', 
                    fontsize=14, fontweight='bold')
        
        # Before oversampling
        before_counts = y_before.value_counts()
        axes[0].bar(['No Claim', 'Claim'], before_counts.values, 
                   color=['#3498db', '#e74c3c'], alpha=0.8)
        axes[0].set_title('Before Oversampling', fontweight='bold')
        axes[0].set_ylabel('Count', fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add count labels
        for i, v in enumerate(before_counts.values):
            axes[0].text(i, v + max(before_counts.values) * 0.02, str(v), 
                        ha='center', fontweight='bold')
        
        # After oversampling
        after_counts = y_after.value_counts()
        axes[1].bar(['No Claim', 'Claim'], after_counts.values, 
                   color=['#3498db', '#e74c3c'], alpha=0.8)
        axes[1].set_title('After Oversampling', fontweight='bold')
        axes[1].set_ylabel('Count', fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add count labels
        for i, v in enumerate(after_counts.values):
            axes[1].text(i, v + max(after_counts.values) * 0.02, str(v), 
                        ha='center', fontweight='bold')
        
        plt.tight_layout()
        output_path = self.output_dir / "class_distribution.png"
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Class distribution saved to: {output_path}")
        return str(output_path)
    
    def generate_all_plots(self, results: pd.DataFrame, models_dict: Dict,
                          X_test, y_test, y_train_before, y_train_after,
                          feature_names: List[str]) -> Dict[str, str]:
        """
        Generate all visualization plots in one call.
        
        Args:
            results: Performance metrics DataFrame
            models_dict: Dictionary of trained models
            X_test: Test features
            y_test: Test labels
            y_train_before: Training labels before oversampling
            y_train_after: Training labels after oversampling
            feature_names: List of feature names
            
        Returns:
            Dictionary mapping plot type to file path
        """
        print("\n" + "="*50)
        print("GENERATING VISUALIZATIONS")
        print("="*50)
        
        output_paths = {}
        
        # 1. Performance comparison
        output_paths['performance'] = self.generate_performance_comparison(results)
        
        # 2. Prepare predictions for confusion matrices and ROC curves
        model_predictions = {}
        for name, model in models_dict.items():
            y_pred = model.predict(X_test)
            pred_data = {
                'y_true': y_test,
                'y_pred': y_pred
            }
            
            # Get probability predictions if available
            if hasattr(model.model, 'predict_proba'):
                y_pred_proba = model.model.predict_proba(X_test)[:, 1]
                pred_data['y_pred_proba'] = y_pred_proba
            
            model_predictions[name] = pred_data
        
        # 3. Confusion matrices
        output_paths['confusion_matrices'] = self.generate_confusion_matrices(model_predictions)
        
        # 4. ROC curves (only if probability predictions available)
        if any('y_pred_proba' in pred for pred in model_predictions.values()):
            output_paths['roc_curves'] = self.generate_roc_curves(model_predictions)
        
        # 5. Feature importance for tree-based models
        for name, model in models_dict.items():
            path = self.generate_feature_importance(model.model, feature_names, name)
            if path:
                output_paths[f'feature_importance_{name}'] = path
        
        # 6. Class distribution
        output_paths['class_distribution'] = self.generate_class_distribution(
            y_train_before, y_train_after
        )
        
        print("="*50)
        print(f"✓ All visualizations saved to: {self.output_dir}/")
        print("="*50)
        
        return output_paths