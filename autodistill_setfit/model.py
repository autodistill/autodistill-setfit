from autodistill.text_classification import TextClassificationTargetModel
from datasets import load_dataset
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset

class SetFitTrainer(TextClassificationTargetModel):
    def __init__(self, model_name=None):
        if model_name:
            self.model = SetFitModel.from_pretrained(model_name)
            self.classes = self.model.labels
        else:
            self.model = None
            self.classes = []

    def predict(self, input: str) -> str:
        preds = self.model.predict([input])

        return preds[0]

    def train(
        self,
        dataset_file,
        setfit_model_id="sentence-transformers/all-mpnet-base-v2",
        output_dir="output",
        epochs=1,
    ):
        dataset = load_dataset("json", data_files=dataset_file, split="train")
        dataset = dataset.rename_column("content", "text")
        dataset = dataset.rename_column("classification", "label")
        
        dataset = dataset.train_test_split(test_size=0.3)
        
        train_dataset = sample_dataset(dataset["train"], label_column="label", num_samples=8)
        eval_dataset = dataset["test"]
        test_dataset = dataset["test"]

        labels = list(set(dataset["train"]["label"]) | set(dataset["test"]["label"]))

        self.classes = labels

        model = SetFitModel.from_pretrained(
            setfit_model_id
        )

        args = TrainingArguments(
            batch_size=16,
            num_epochs=epochs,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )

        args.eval_strategy = args.evaluation_strategy

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            metric="accuracy",
            column_mapping={"text": "text", "label": "label"}  # Map dataset columns to text/label expected by trainer
        )

        trainer.train()
        metrics = trainer.evaluate()

        print(metrics)

        model.save_pretrained(output_dir)
