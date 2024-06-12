from transformers import AutoImageProcessor, ViTForImageClassification, CLIPProcessor, CLIPModel, DeiTForImageClassificationWithTeacher, Dinov2Model
import config as c


def load_model (model_name):

    if 'clip' in model_name:
        model = CLIPModel.from_pretrained(model_name)
        processor_clip = CLIPProcessor.from_pretrained(model_name)
        model = model.to(c.device)
        model = model.vision_model
        def processor(images, return_tensors="pt"):
            return processor_clip(text=["a photo of a cat", "a photo of a dog"], images=images, return_tensors=return_tensors, padding=True)
    elif 'dinov2' in model_name:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = Dinov2Model.from_pretrained(model_name)
        model = model.to(c.device)
    elif 'dino' in model_name:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = ViTForImageClassification.from_pretrained(model_name)
        model = model.to(c.device)
    elif 'vit' in model_name:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = ViTForImageClassification.from_pretrained(model_name)
        model = model.to(c.device)
    elif 'deit' in model_name:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = DeiTForImageClassificationWithTeacher.from_pretrained(model_name)
        model = model.to(c.device)

    return model, processor

