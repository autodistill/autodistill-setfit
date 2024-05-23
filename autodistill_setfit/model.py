from autodistill.text_classification import TextClassificationTargetModel
from datasets import load_dataset
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset


class SetFitModel(TextClassificationTargetModel):
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
        setfit_model_id="sentence-transformers/paraphrase-mpnet-base-v2",
        output="output",
        epochs=5,
    ):
        dataset = load_dataset("json", data_files=dataset_file, split="train")
        dataset = dataset.rename_column("content", "text")
        # do train test split
        dataset = sample_dataset(dataset, 0.8, 0.1, 0.1)
        train_dataset = dataset["train"]
        eval_dataset = dataset["eval"]
        test_dataset = dataset["test"]

        labels = dataset["train"].unique("label")
        label_map = {label: i for i, label in enumerate(labels)}
        self.classes = labels

        model = SetFitModel.from_pretrained(
            setfit_model_id, labels=[label for label in label_map.keys()]
        )

        args = TrainingArguments(
            batch_size=16,
            num_epochs=epochs,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            metric="accuracy",
            column_mapping={"sentence": "content", "label": "classification"},
        )

        trainer.train()
        metrics = trainer.evaluate(test_dataset)

        print(metrics)

        model.save_pretrained(output)
