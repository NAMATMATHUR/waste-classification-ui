import streamlit as st
from PIL import Image

st.title("♻ Waste Classification System")
st.subheader("CNN vs CLIP Comparison")

st.write("Upload an image of waste to classify.")

uploaded_file = st.file_uploader("Upload Waste Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict using CNN"):
        st.success("CNN Prediction: Plastic (example)")

    if st.button("Predict using CLIP"):
        st.success("CLIP Prediction: Plastic (example)")

st.subheader("Try Sample Images")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Plastic Bottle"):
        st.image("samples/plastic.jpg")
        st.write("Prediction: Plastic")

with col2:
    if st.button("Paper"):
        st.image("samples/paper.jpg")
        st.write("Prediction: Paper")

with col3:
    if st.button("Glass"):
        st.image("samples/glass.jpg")
        st.write("Prediction: Glass")