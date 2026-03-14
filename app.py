import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="WasteAI — CNN vs CLIP",
    page_icon="♻️",
    layout="wide"
)

st.title("♻️ Waste Classification System")
st.write("Comparison of **CNN (Supervised)** vs **CLIP (Zero-Shot)**")

# -------------------------------------------------
# Waste Classes
# -------------------------------------------------

classes = ["plastic", "paper", "glass", "metal"]

# -------------------------------------------------
# Image Transform for CNN
# -------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -------------------------------------------------
# Load CNN Model
# -------------------------------------------------

@st.cache_resource
def load_cnn_model():

    model = models.resnet18(pretrained=False)

    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))

    model.load_state_dict(torch.load("models/cnn_model.pth", map_location="cpu"))

    model.eval()

    return model


cnn_model = load_cnn_model()

# -------------------------------------------------
# CNN Prediction
# -------------------------------------------------

def predict_cnn(image):

    img = transform(image).unsqueeze(0)

    with torch.no_grad():

        outputs = cnn_model(img)

        probs = F.softmax(outputs, dim=1)

        index = probs.argmax().item()

        confidence = probs[0][index].item()

    return classes[index], confidence


# -------------------------------------------------
# Load CLIP Model
# -------------------------------------------------

@st.cache_resource
def load_clip():

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    return model, processor


clip_model, processor = load_clip()

# -------------------------------------------------
# CLIP Labels
# -------------------------------------------------

labels = [
    "plastic waste",
    "paper waste",
    "glass waste",
    "metal waste"
]

# -------------------------------------------------
# CLIP Prediction
# -------------------------------------------------

def predict_clip(image):

    inputs = processor(
        text=labels,
        images=image,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():

        outputs = clip_model(**inputs)

        probs = outputs.logits_per_image.softmax(dim=1)

        index = probs.argmax().item()

        confidence = probs[0][index].item()

    return labels[index], confidence


# -------------------------------------------------
# Upload Image
# -------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Waste Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------------------------
# Run Predictions
# -------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Input Image", width=300)

    with st.spinner("Running Models..."):

        cnn_pred, cnn_conf = predict_cnn(image)

        clip_pred, clip_conf = predict_clip(image)

    col1, col2 = st.columns(2)

    # CNN Result
    with col1:

        st.subheader("CNN Prediction")

        st.success(f"Class: {cnn_pred}")

        st.write(f"Confidence: {cnn_conf:.2f}")

    # CLIP Result
    with col2:

        st.subheader("CLIP Prediction")

        st.info(f"Class: {clip_pred}")

        st.write(f"Confidence: {clip_conf:.2f}")

    # Agreement Check

    if cnn_pred in clip_pred:

        st.success("✅ Both models agree")

    else:

        st.warning("⚠ Models disagree – manual inspection recommended")