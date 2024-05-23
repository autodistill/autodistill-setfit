<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png"
      >
    </a>
  </p>
</div>

# Autodistill SetFit Module

This repository contains the code supporting the SetFit target model trainer for use with [Autodistill](https://github.com/autodistill/autodistill).

SetFit is a framework for fine-tuning Sentence Transformer models with a few examples of each class on which you want to train. SetFit is developed by [Hugging Face](https://github.com/huggingface/setfit).

## Installation

To use the SetFit target model, you will need to install the following dependency:

```bash
pip3 install autodistill-setfit
```

## Quickstart

The SetFit module takes in `.jsonl` files and trains a text classification model.

Each record in the JSONL file should have an entry called `text` that contains the text to be classified. The `label` entry should contain the ground truth label for the text. This format is returned by Autodistill base text classification models like the GPTClassifier.

Here is an example entry of a record used to train a research paper subject classifier:

```json
{"title": "CC-GPX: Extracting High-Quality Annotated Geospatial Data from Common Crawl", "content": "arXiv:2405.11039v1 Announce Type: new \nAbstract: The Common Crawl (CC) corpus....", "classification": "natural language processing"}
```

```python
from autodistill_setfit import SetFitModel

target_model = SetFitModel()

# train a model
target_model.train("./data.jsonl", output="model", epochs=5)

target_model = SetFitModel("model")

# run inference on the new model
pred = target_model.predict("Geospatial data.")

print(pred)
# geospatial
```

## License

This project is licensed under an [MIT license](LICENSE).

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!
