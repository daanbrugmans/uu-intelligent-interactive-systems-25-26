
import os
from abc import ABC
from pathlib import Path

import pandas as pd
import torch
import transformers
import datasets
import peft
from tqdm import tqdm
from sklearn.metrics import accuracy_score

import runtime


class DeepEmotionClassifier(ABC):
    def __init__(self):
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
            path_to_finetuned_model
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
        input_features["pixel_values"] = torch.squeeze(input_features["pixel_values"])
        input_features["labels"] = dataset["label"]

        return input_features

    lora_config = peft.LoraConfig(
        r=8,
        lora_alpha=16,  # Usually twice the rank `r`
        lora_dropout=0.1,
        bias="none",
        inference_mode=False,
        target_modules=["query", "key", "value", "classifier"],
    )

    training_args = transformers.TrainingArguments(
        output_dir=None,
        eval_strategy="epoch",
        num_train_epochs=5,
        learning_rate=3e-4,
        weight_decay=0.02,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        save_total_limit=1,
        fp16=True,
        seed=313131,
    )

    lora_vit = peft.get_peft_model(vit_model.model, lora_config)
    preprocessed_train = train.map(prepare_dataset_for_training, batched=False)
    preprocessed_val = val.map(prepare_dataset_for_training, batched=False)

    trainer = transformers.Trainer(
        lora_vit,
        args=training_args,
        train_dataset=preprocessed_train,
        eval_dataset=preprocessed_val
    )

    trainer.train()

    project_root = Path(__file__).parents[0]
    path_to_finetuned_model = str(Path(project_root, "finetuned_vit"))

    lora_vit.save_pretrained(path_to_finetuned_model)
    

def dima806_label_to_fer_label(dima806_label: int) -> int:
    if dima806_label == 0:
        return 5 # Sad
    elif dima806_label == 1:
        return 1 # Disgust
    elif dima806_label == 2:
        return 0 # Anger
    elif dima806_label == 3:
        return 4 # Neutral
    elif dima806_label == 4:
        return 2 # Fear
    elif dima806_label == 5:
        return 6 # Surprise
    elif dima806_label == 6:
        return 3 # Happy
    
def get_predictions(vit_model, dataset: datasets.Dataset, use_dima806_labels: bool = False) -> list:   
    predictions = []
    
    for row in tqdm(dataset):        
        pixel_values = vit_model.processor(row["image"], return_tensors="pt").to(device)
        model_output = vit_model.model(**pixel_values)
        emissions = model_output.logits
        prediction = emissions.argmax(-1).item()
        
        if use_dima806_labels:
            prediction = dima806_label_to_fer_label(prediction)
        
        predictions.append(prediction)
        
    return predictions


if __name__ == "__main__":
    project_root = Path(__file__).parents[0]
    dataset_with_splits = datasets.load_from_disk(
        Path(project_root, "emotion_classifier_datasets", "daan")
    )

    train_vit_for_emotion_classification(
        BaseViT(), dataset_with_splits["train"], dataset_with_splits["val"]
    )
