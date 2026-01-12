from abc import ABC, abstractmethod
import pandas as pd
from typing import Tuple, Dict, List, Optional

class IDataLoader(ABC):
    @abstractmethod
    def load(self) -> pd.DataFrame:
        pass

class IPreprocessor(ABC):
    @abstractmethod
    def process(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        pass

class IModel(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass
    
    @abstractmethod
    def predict(self, X_test):
        pass

class IVisualizer(ABC):
    """
    Interface for visualization services.
    Implements Strategy Pattern for different visualization backends.
    """
    
    @abstractmethod
    def generate_performance_comparison(self, results: pd.DataFrame) -> str:
        """
        Generate bar chart comparing model metrics.
        
        Args:
            results: DataFrame with models as index and metrics as columns
            
        Returns:
            Path to saved visualization
        """
        pass
    
    @abstractmethod
    def generate_confusion_matrices(self, model_predictions: Dict[str, dict]) -> str:
        """
        Generate confusion matrices for all models.
        
        Args:
            model_predictions: Dict with model names as keys and 
                             {'y_true', 'y_pred'} as values
                             
        Returns:
            Path to saved visualization
        """
        pass
    
    @abstractmethod
    def generate_roc_curves(self, model_predictions: Dict[str, dict]) -> str:
        """
        Generate ROC curves for all models.
        
        Args:
            model_predictions: Dict with model names as keys and 
                             {'y_true', 'y_pred_proba'} as values
                             
        Returns:
            Path to saved visualization
        """
        pass
    
    @abstractmethod
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
            Path to saved visualization or None if not supported
        """
        pass
    
    @abstractmethod
    def generate_class_distribution(self, y_before: pd.Series, y_after: pd.Series) -> str:
        """
        Generate before/after comparison of class distribution.
        
        Args:
            y_before: Target variable before oversampling
            y_after: Target variable after oversampling
            
        Returns:
            Path to saved visualization
        """
        pass
    
    @abstractmethod
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
        pass