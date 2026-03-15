import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# -------------------------
# Page Settings
# -------------------------

st.set_page_config(page_title="Waste Classification AI", layout="wide")

st.title("♻ Smart Waste Classification System")
st.subheader("CNN vs CLIP Waste Detection")

# -------------------------
# Waste Classes
# -------------------------

classes = [
    "plastic","plastic","plastic","plastic",
    "paper","paper","paper",
    "cardboard","cardboard",
    "glass","glass","glass",
    "metal","metal","metal"
]

clip_labels = [
    "a photo of plastic waste",
    "a plastic bottle",
    "plastic packaging",
    "plastic container",

    "paper waste",
    "newspaper",
    "office paper",

    "cardboard waste",
    "cardboard box",

    "glass waste",
    "glass bottle",
    "broken glass",

    "metal waste",
    "aluminum can",
    "steel can"
]

# -------------------------
# Image Transform
# -------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# -------------------------
# Load CNN Model
# -------------------------

@st.cache_resource
def load_cnn():

    model = models.resnet18(weights="DEFAULT")

    model.fc = torch.nn.Linear(model.fc.in_features, 4)

    # Load your trained model later
    # model.load_state_dict(torch.load("cnn_model.pth", map_location="cpu"))

    model.eval()

    return model

cnn_model = load_cnn()

# -------------------------
# Load CLIP Model
# -------------------------

@st.cache_resource
def load_clip():

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    return model, processor

clip_model, processor = load_clip()

# -------------------------
# CNN Prediction
# -------------------------

def predict_cnn(image):

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = cnn_model(img)

    probs = torch.nn.functional.softmax(outputs, dim=1)

    conf, pred = torch.max(probs,1)

    class_names = ["plastic","paper","glass","metal"]

    return class_names[pred.item()], conf.item()*100

# -------------------------
# CLIP Prediction
# -------------------------

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

    predicted_class = classes[pred.item()]

    return predicted_class, conf.item()*100

# -------------------------
# Input Section
# -------------------------

st.header("Upload or Capture Waste Image")

uploaded_file = st.file_uploader(
    "Upload Waste Image",
    type=["jpg","jpeg","png"]
)

camera_photo = st.camera_input("Or take a photo")

image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

elif camera_photo is not None:
    image = Image.open(camera_photo).convert("RGB")

# -------------------------
# Prediction
# -------------------------

if image is not None:

    st.image(image, caption="Input Image", width=300)

    if st.button("Run Prediction"):

        with st.spinner("Analyzing waste..."):

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

# -------------------------
# Footer
# -------------------------

st.markdown("---")
st.write("Developed for AI Waste Management Project")