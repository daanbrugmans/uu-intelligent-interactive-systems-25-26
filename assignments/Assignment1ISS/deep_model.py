import transformers


class DeepEmotionDetector:
    def __init__(self):
        self.preprocessor = transformers.ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )
        self.model = transformers.ViTForImageClassification.from_pretrained(
            "dima806/facial_emotions_image_detection"
        )
