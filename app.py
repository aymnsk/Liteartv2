import streamlit as st
from PIL import Image
from fst import apply_style_transfer

st.set_page_config(page_title="LiteArt Studio", layout="centered")
st.title("🎨 LiteArt Studio - Neural Style Transfer")

style_options = {
    "Mosaic 🎭": "mosaic",
    "Candy 🍬": "candy",
    "Rain Princess 🌧️👸": "rain_princess",
    "Udnie 🌀": "udnie",
}

uploaded_file = st.file_uploader("📤 Upload an image to stylize", type=["jpg", "jpeg", "png"])
style_choice = st.selectbox("🎨 Choose a style", list(style_options.keys()))

if uploaded_file:
    content_img = Image.open(uploaded_file).convert("RGB")
    st.image(content_img, caption="🖼️ Original Image", use_container_width=True)

    if st.button("✨ Apply Style"):
        with st.spinner("Applying style... please wait ⏳"):
            try:
                output_img = apply_style_transfer(content_img, style_options[style_choice])
                st.image(output_img, caption=f"🎨 Stylized with {style_choice}", use_container_width=True)
            except Exception as e:
                st.error(f"An error occurred: {e}")
