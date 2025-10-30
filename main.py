# main.py
from model_trainer import ModelTrainer
from data_preprocessing import load_and_preprocess_data

if __name__ == "__main__":
    # Convenience: run full preprocessing then comparison trainer
    X_train, X_test, y_train, y_test, pipeline, feature_names = load_and_preprocess_data(
    benign_path=r"C:\Users\AgentxVenom\Documents\Soham Goswami\New_Ransomwatch\data\benign.csv",
    ransom_path=r"C:\Users\AgentxVenom\Documents\Soham Goswami\New_Ransomwatch\data\ransom.csv"
)
    X = __import__("pandas").concat([X_train, X_test], ignore_index=True)
    y = __import__("pandas").concat([y_train, y_test], ignore_index=True)
    trainer = ModelTrainer()
    summary = trainer.compare_and_train(X, y)
    print("Done. Summary:", summary)
