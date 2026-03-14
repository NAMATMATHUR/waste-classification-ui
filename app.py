import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# ------------------------------
# Page settings
# ------------------------------

st.set_page_config(page_title="Waste Classification AI", layout="wide")

st.title("♻ Smart Waste Classification System")
st.subheader("CNN vs CLIP Model Comparison")

# ------------------------------
# Waste Classes
# ------------------------------

classes = ["plastic", "paper", "glass", "metal"]

clip_labels = [
    "a photo of plastic waste",
    "a photo of paper waste",
    "a photo of glass waste",
    "a photo of metal waste"
]

# ------------------------------
# Image preprocessing for CNN
# ------------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ------------------------------
# Load CNN model
# ------------------------------

@st.cache_resource
def load_cnn():

    model = models.resnet18(weights="DEFAULT")

    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))

    # If you have trained weights later
    # model.load_state_dict(torch.load("cnn_model.pth", map_location="cpu"))

    model.eval()

    return model


cnn_model = load_cnn()

# ------------------------------
# Load CLIP model
# ------------------------------

@st.cache_resource
def load_clip():

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    return clip_model, processor


clip_model, processor = load_clip()

# ------------------------------
# CNN Prediction
# ------------------------------

def predict_cnn(image):

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = cnn_model(img)

    probs = torch.nn.functional.softmax(outputs, dim=1)

    conf, pred = torch.max(probs,1)

    return classes[pred.item()], conf.item()*100


# ------------------------------
# CLIP Prediction
# ------------------------------

def predict_clip(image):

    inputs = processor(
        text=clip_labels,
        images=image,
        return_tensors="pt",
        padding=True
    )

    outputs = clip_model(**inputs)

    probs = outputs.logits_per_image.softmax(dim=1)

    conf, pred = torch.max(probs,1)

    return classes[pred.item()], conf.item()*100


# ------------------------------
# Upload Image
# ------------------------------

uploaded_file = st.file_uploader(
    "Upload Waste Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width=300)

    if st.button("Run Prediction"):

        with st.spinner("Running AI models..."):

            cnn_label, cnn_conf = predict_cnn(image)

            clip_label, clip_conf = predict_clip(image)

        col1, col2 = st.columns(2)

        with col1:
            st.success("CNN Prediction")
            st.write(f"Class: **{cnn_label}**")
            st.progress(int(cnn_conf))
            st.write(f"Confidence: {cnn_conf:.2f}%")

        with col2:
            st.info("CLIP Prediction")
            st.write(f"Class: **{clip_label}**")
            st.progress(int(clip_conf))
            st.write(f"Confidence: {clip_conf:.2f}%")

# ------------------------------
# Footer
# ------------------------------

st.markdown("---")

st.write("Developed for Waste Management AI Project")