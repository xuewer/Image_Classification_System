from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from model import build_model
from run_utils import latest_checkpoint


st.set_page_config(
    page_title="Image Classification Demo",
    page_icon="🍎",
    layout="centered",
)

st.title("Image Classification System")
st.caption("PyTorch + ResNet18 / ResNet50 + Data Augmentation + Error Analysis")

PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINT = latest_checkpoint(PROJECT_DIR / "outputs")


@st.cache_resource
def load_model(checkpoint_path: str, checkpoint_version: tuple[int, int]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    classes = checkpoint["classes"]
    image_size = checkpoint.get("image_size", 224)
    model_name = checkpoint.get("model_name", "resnet18")

    model = build_model(
        len(classes),
        pretrained=False,
        model_name=model_name,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    normalize = transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    )
    transforms_for_tta = [
        transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    ]

    return model, classes, transforms_for_tta, device


def predict_with_tta(model, image, transforms_for_tta, device):
    probabilities = []
    with torch.no_grad():
        for transform in transforms_for_tta:
            inputs = transform(image).unsqueeze(0).to(device)
            probabilities.append(F.softmax(model(inputs), dim=1)[0])
    return torch.stack(probabilities).mean(dim=0)


uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
)

if CHECKPOINT is None:
    st.warning(
        f"Model checkpoint not found under {PROJECT_DIR / 'outputs'}\n\n"
        "Train the model first, for example: "
        "`python train.py --data data/split --epochs 5 --device cpu`"
    )
else:
    st.caption(f"Checkpoint: `{CHECKPOINT}`")

if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("Unable to read this image. Please upload a valid JPG, PNG or WEBP file.")
        st.stop()
    st.image(image, caption="Input image", use_container_width=True)

    if CHECKPOINT is None:
        st.stop()

    try:
        checkpoint_stat = CHECKPOINT.stat()
        model, classes, transforms_for_tta, device = load_model(
            str(CHECKPOINT), (checkpoint_stat.st_mtime_ns, checkpoint_stat.st_size)
        )
        probabilities = predict_with_tta(
            model,
            image,
            transforms_for_tta,
            device,
        )

        top_k = min(5, len(classes))
        values, indices = torch.topk(probabilities, top_k)

        st.subheader("Prediction")
        st.success(f"Predicted class: {classes[indices[0].item()]}")

        for value, index in zip(values.cpu().tolist(), indices.cpu().tolist()):
            st.write(f"**{classes[index]}** - {value:.2%}")
            st.progress(float(value))

    except Exception as error:
        st.error(str(error))
else:
    st.info("Upload an image to start prediction.")
