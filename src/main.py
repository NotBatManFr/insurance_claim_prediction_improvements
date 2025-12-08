import sys
import os

# Ensure src is in pythonpath
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.config import AppConfig
from src.data_loader import CSVLoader
from src.preprocessor import InsurancePreprocessor
from src.models import SklearnModelAdapter
from src.pipeline import PipelineOrchestrator

def main():
    try:
        # 1. Configuration
        config = AppConfig()

        # 2. Dependencies
        loader = CSVLoader(config.data_path)
        preprocessor = InsurancePreprocessor()
        
        # 3. Pipeline
        pipeline = PipelineOrchestrator(config, loader, preprocessor)

        # 4. Add Models
        pipeline.add_model("Logistic Regression", SklearnModelAdapter(LogisticRegression(max_iter=1000)))
        pipeline.add_model("Decision Tree", SklearnModelAdapter(DecisionTreeClassifier(random_state=config.random_state)))
        pipeline.add_model("Random Forest", SklearnModelAdapter(RandomForestClassifier(random_state=config.random_state)))

        # 5. Run
        results = pipeline.run()
        print("\n" + "="*50)
        print("FINAL MODEL PERFORMANCE")
        print("="*50)
        print(results)
        print("="*50)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Tip: Ensure 'train_data.csv' is inside the 'data/' folder.")

if __name__ == "__main__":
    main()
