
import os
import pandas as pd
import numpy as np
import torch
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

@dataclass
class Config:
    model_name: str = "ProsusAI/finbert"
    num_labels: int = 3
    num_train_epochs: int = 8
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    learning_rate: float = 1e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    output_dir: str = "trained_financial_model"
    dataset_path: str = "data/all-data.csv"
    max_length: int = 256
    eval_steps: int = 100
    save_steps: int = 200
    logging_steps: int = 50
    early_stopping_patience: int = 3


class FinancialSentimentPredictor:
    def __init__(self, config=None):
        self.cfg = config if config else Config()
        self.model = None
        self.tokenizer = None
        self.labels = ["negative", "neutral", "positive"]
        self.label_map = {"negative": 0, "neutral": 1, "positive": 2}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}

    #Load Model
    def load_model(self):
        if not os.path.exists(self.cfg.output_dir):
            print(f"Error: Model folder '{self.cfg.output_dir}' not found.")
            return False
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.output_dir, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.cfg.output_dir, local_files_only=True, num_labels=self.cfg.num_labels
            )
            self.model.eval()
            print("Model loaded successfully.")
            return True
        except Exception as e:
            print(f"Model load failed: {e}")
            return False

    #  Predict Sentiment 
    def predict_sentiment(self, text: str):
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Run load_model() first.")

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=self.cfg.max_length
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]

        idx = int(np.argmax(probs))
        sentiment = self.reverse_label_map[idx]
        confidence = float(probs[idx])

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "scores": {
                "negative": float(probs[0]),
                "neutral": float(probs[1]),
                "positive": float(probs[2])
            }
        }

    #Load Dataset 
    def load_dataset(self):
        if not os.path.exists(self.cfg.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.cfg.dataset_path}")

        df = pd.read_csv(self.cfg.dataset_path, encoding="ISO-8859-1")
        if len(df.columns) == 2:
            df.columns = ["sentiment", "sentence"]

        sentiment_map = {"positive": 2, "neutral": 1, "negative": 0}
        df["label"] = df["sentiment"].str.lower().map(sentiment_map)
        df.dropna(subset=["label", "sentence"], inplace=True)

        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
        print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")
        return train_df, val_df

    #Tokenization 
    def tokenize_function(self, examples):
        return self.tokenizer(
            examples["sentence"], truncation=True, padding="max_length", max_length=self.cfg.max_length
        )

    def preprocess_datasets(self, train_df, val_df):
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)

        tokenized_train = train_dataset.map(self.tokenize_function, batched=True)
        tokenized_val = val_dataset.map(self.tokenize_function, batched=True)

        tokenized_train = tokenized_train.remove_columns(
            [c for c in train_dataset.column_names if c not in ["input_ids", "attention_mask", "label"]]
        )
        tokenized_val = tokenized_val.remove_columns(
            [c for c in val_dataset.column_names if c not in ["input_ids", "attention_mask", "label"]]
        )

        tokenized_train.set_format("torch")
        tokenized_val.set_format("torch")
        print("Dataset preprocessing complete.")
        return tokenized_train, tokenized_val

    #Evaluation Metrics 
    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    # Model Training
    def train(self):
        print("Starting model training...")
        train_df, val_df = self.load_dataset()

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.cfg.model_name, num_labels=self.cfg.num_labels
        )

        tokenized_train, tokenized_val = self.preprocess_datasets(train_df, val_df)

        training_args = TrainingArguments(
            output_dir=self.cfg.output_dir,
            num_train_epochs=self.cfg.num_train_epochs,
            per_device_train_batch_size=self.cfg.per_device_train_batch_size,
            per_device_eval_batch_size=self.cfg.per_device_eval_batch_size,
            learning_rate=self.cfg.learning_rate,
            warmup_steps=self.cfg.warmup_steps,
            weight_decay=self.cfg.weight_decay,
            logging_dir=os.path.join(self.cfg.output_dir, "logs"),
            logging_steps=self.cfg.logging_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=self.cfg.eval_steps,
            save_steps=self.cfg.save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            report_to="none",
        )

        early_stop = EarlyStoppingCallback(early_stopping_patience=self.cfg.early_stopping_patience)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[early_stop],
        )

        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(self.cfg.output_dir)
        print("Training completed and model saved.")


if __name__ == "__main__":
    cfg = Config()
    predictor = FinancialSentimentPredictor(cfg)
    predictor.train()

    # Quick sentiment test
    if predictor.load_model():
        text = "The company reported record profits this quarter."
        result = predictor.predict_sentiment(text)
        print(f"\nText: {text}")
        print(f"Predicted Sentiment: {result['sentiment'].upper()} (Confidence: {result['confidence']:.2f})")
