"""
Test Research Model on a single image.
"""

import sys
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn


class EfficientNetDetector(nn.Module):
    """EfficientNet-B4 based AI detector."""

    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b4(weights=None)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2),
        )
        self.temperature = nn.Parameter(torch.ones(1))


def test_image(image_path: str):
    """Test a single image with the research model."""
    print("\n" + "=" * 60)
    print("  RESEARCH MODEL (EfficientNet-B4) DETECTION")
    print("=" * 60)

    # Load image
    print(f"\n  Loading image: {image_path}")
    img = Image.open(image_path).convert("RGB")

    # Transform
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    img_tensor = transform(img).unsqueeze(0)

    # Load model
    print("  Loading trained model...")
    model = EfficientNetDetector()
    checkpoint = torch.load(
        "models/research/research_model_final.pth",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.temperature.data = torch.tensor([checkpoint["temperature"]])
    model.eval()

    # Predict
    print("  Running inference...")
    with torch.no_grad():
        logits = model.backbone(img_tensor)
        scaled = logits / model.temperature
        probs = torch.softmax(scaled, dim=1)
        ai_prob = probs[0, 1].item()
        real_prob = probs[0, 0].item()

    # Results
    print("\n" + "-" * 60)
    print("  RESULTS")
    print("-" * 60)
    print(f"  AI-Generated Probability:  {ai_prob:.1%}")
    print(f"  Real Image Probability:    {real_prob:.1%}")
    print(f"  Calibration Temperature:   {checkpoint['temperature']:.3f}")
    print()

    if ai_prob > 0.7:
        verdict = "HIGH LIKELIHOOD - AI-GENERATED"
    elif ai_prob > 0.4:
        verdict = "MODERATE LIKELIHOOD - Possibly AI"
    else:
        verdict = "LOW LIKELIHOOD - Probably Real"

    print(f"  VERDICT: {verdict}")
    print("=" * 60 + "\n")

    return ai_prob


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_research_model.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    test_image(image_path)
