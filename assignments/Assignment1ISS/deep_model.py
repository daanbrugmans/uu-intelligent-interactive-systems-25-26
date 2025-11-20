from abc import ABC
from pathlib import Path

import transformers
import datasets
import peft


class DeepEmotionClassifier(ABC):
    def __init__(self):
        # self.preprocessor = transformers.ViTImageProcessor.from_pretrained(
        #     "google/vit-base-patch16-224-in21k"
        # )
        # self.model = transformers.ViTForImageClassification.from_pretrained(
        #     "dima806/facial_emotions_image_detection"
        # )
        super.__init__()

        self.processor
        self.model


class BaseViT(DeepEmotionClassifier):
    def __init__(self):
        self.processor = transformers.ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224"
        )
        self.model = transformers.ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224", num_labels=7, ignore_mismatched_sizes=True
        )


class FinetunedViT(DeepEmotionClassifier):
    def __init__(self):
        project_root = Path(__file__).parents[0]
        path_to_finetuned_model = str(Path(project_root, "finetuned_vit"))

        self.processor = transformers.ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224"
        )
        self.model = peft.PeftModel.from_pretrained(
            transformers.ViTForImageClassification.from_pretrained(
                "google/vit-base-patch16-224",
                num_labels=7,
                ignore_mismatched_sizes=True,
            ),
            path_to_finetuned_model,
        )


class Dima806ViT(DeepEmotionClassifier):
    def __init__(self):
        self.processor = transformers.ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )
        self.model = transformers.ViTForImageClassification.from_pretrained(
            "dima806/facial_emotions_image_detection"
        )


def train_vit_for_emotion_classification(
    vit_model, train: datasets.Dataset, val: datasets.Dataset
) -> None:
    def prepare_dataset_for_training(dataset: datasets.Dataset) -> datasets.Dataset:
        input_features = vit_model.processor(dataset["image"], return_tensors="pt")
        input_features["labels"] = dataset["label"]

        return input_features

    lora_config = peft.LoraConfig(
        r=8,
        lora_alpha=16,  # Usually twice the rank `r`
        lora_dropout=0.1,
        bias="none",
        inference_mode=False,
        target_modules=["query", "key", "value"],
    )

    training_args = transformers.TrainingArguments(
        output_dir=None,
        eval_strategy="epoch",
        num_train_epochs=5,
        learning_rate=3e-6,
        weight_decay=0.02,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        save_total_limit=1,
        fp16=True,
        seed=313131,
    )

    lora_vit = peft.get_peft_model(vit_model.model, lora_config)
    preprocessed_train = train.map(prepare_dataset_for_training, batched=True)
    preprocessed_val = val.map(prepare_dataset_for_training, batched=True)

    trainer = transformers.Trainer(
        lora_vit,
        args=training_args,
        train_dataset=preprocessed_train,
        eval_dataset=preprocessed_val,
        tokenizer=vit_model.tokenizer,
    )

    trainer.train()

    project_root = Path(__file__).parents[0]
    path_to_finetuned_model = str(Path(project_root, "finetuned_vit"))

    lora_vit.save_pretrained(path_to_finetuned_model)


if __name__ == "__main__":
    project_root = Path(__file__).parents[0]
    dataset_with_splits = datasets.load_from_disk(
        Path(project_root, "emotion_classifier_datasets", "daan")
    )

    train_vit_for_emotion_classification(
        BaseViT(), dataset_with_splits["train"], dataset_with_splits["val"]
    )
