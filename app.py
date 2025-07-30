import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
import io
import json
import os
from huggingface_hub import login
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")
if HUGGINGFACE_TOKEN:
    login(token=HUGGINGFACE_TOKEN)
else:
    st.warning("⚠️ Hugging Face token not set. Set it via environment variable `HF_TOKEN`.")

st.title("🎨 AI Image Generator")
prompt = st.text_input("Enter your image prompt:")
steps = st.slider("Steps", 10, 100, 30)
guidance = st.slider("Guidance Scale", 1.0, 20.0, 7.5)
generate = st.button("Generate Image")

# Model options
default_models = {
    "Stable Diffusion v1-5 (default)": "runwayml/stable-diffusion-v1-5",
}

local_model_dir = "models"
if os.path.exists(local_model_dir):
    for file in os.listdir(local_model_dir):
        if file.endswith(".safetensors") or file.endswith(".ckpt"):
            default_models[file] = os.path.join(local_model_dir, file)

selected_model = st.selectbox("Select a base model:", list(default_models.keys()))

# Choose device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model only if changed
if 'pipe' not in st.session_state or st.session_state.get('last_model') != selected_model:
    model_path = default_models[selected_model]
    try:
        if model_path.endswith(('.safetensors', '.ckpt')):
            st.session_state.pipe = StableDiffusionPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        else:
            st.session_state.pipe = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        st.session_state.pipe.to(device)
        st.session_state.last_model = selected_model
    except Exception as e:
        st.error(f"🚨 Failed to load model: {e}")

if generate and prompt:
    if "pipe" not in st.session_state:
        st.error("🚨 The model failed to load. Please check the model path or Hugging Face token.")
    else:
        with st.spinner("⏳ Generating..."):
            pipe = st.session_state.pipe
            image = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance).images[0]

            col1, col2 = st.columns([3, 1])
            with col1:
                st.image(image, caption="Generated Image", use_container_width=True)
            with col2:
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                st.download_button("Download Image", buf.getvalue(), file_name="generated.png", mime="image/png")

                attributes = {
                    "prompt": prompt,
                    "model": selected_model,
                    "steps": steps,
                    "guidance_scale": guidance
                }
                attr_buf = io.StringIO()
                json.dump(attributes, attr_buf)
                st.download_button("Download Attributes", attr_buf.getvalue(), file_name="attributes.json", mime="application/json")