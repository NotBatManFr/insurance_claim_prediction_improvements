import sys
import os
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.config import AppConfig
from src.data_loader import CSVLoader
from src.preprocessor import InsurancePreprocessor
from src.models import SklearnModelAdapter
from src.pipeline import PipelineOrchestrator
from src.visualizer import MatplotlibVisualizer

def main():
    """
    Main entry point with optional visualization support.
    
    Usage:
        python src/main.py                    # Run with visualizations
        python src/main.py --no-viz           # Run without visualizations
        python src/main.py --viz-dir plots/   # Custom visualization directory
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Insurance Claim Prediction Pipeline')
    parser.add_argument('--no-viz', action='store_true', 
                       help='Disable visualization generation')
    parser.add_argument('--viz-dir', type=str, default='visualizations',
                       help='Directory for saving visualizations (default: visualizations/)')
    args = parser.parse_args()
    
    try:
        # 1. Configuration
        config = AppConfig()

        # 2. Dependencies
        loader = CSVLoader(config.data_path)
        preprocessor = InsurancePreprocessor()
        
        # 3. Visualizer (optional)
        visualizer = None if args.no_viz else MatplotlibVisualizer(output_dir=args.viz_dir)
        
        # 4. Pipeline
        pipeline = PipelineOrchestrator(config, loader, preprocessor, visualizer)

        # 5. Add Models
        pipeline.add_model("Logistic Regression", 
                          SklearnModelAdapter(LogisticRegression(max_iter=1000)))
        pipeline.add_model("Decision Tree", 
                          SklearnModelAdapter(DecisionTreeClassifier(random_state=config.random_state)))
        pipeline.add_model("Random Forest", 
                          SklearnModelAdapter(RandomForestClassifier(random_state=config.random_state)))

        # 6. Run
        enable_viz = not args.no_viz
        results = pipeline.run(enable_visualization=enable_viz)
        
        print("\n" + "="*50)
        print("FINAL MODEL PERFORMANCE")
        print("="*50)
        print(results)
        print("="*50)
        
        if enable_viz:
            print(f"\n📊 Visualizations saved to: {args.viz_dir}/")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Tip: Ensure 'train_data.csv' is inside the 'data/' folder.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()