import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="centered"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main { padding-top: 2rem; }
    .title-block { text-align: center; margin-bottom: 2.5rem; }
    .title-block h1 { font-size: 2.2rem; font-weight: 500; letter-spacing: -0.5px; color: #0f0f0f; margin-bottom: 0.3rem; }
    .title-block p { font-size: 1rem; color: #666; font-weight: 300; }
    .result-card { background: #0f0f0f; border-radius: 16px; padding: 2rem; text-align: center; margin: 1.5rem 0; }
    .result-digit { font-family: 'DM Mono', monospace; font-size: 5rem; font-weight: 500; color: #ffffff; line-height: 1; }
    .result-conf { font-size: 1rem; color: #888; margin-top: 0.5rem; font-weight: 300; }
    .tip-box { background: #f7f7f5; border-left: 3px solid #0f0f0f; border-radius: 0 8px 8px 0; padding: 0.8rem 1rem; margin-top: 1.5rem; font-size: 0.85rem; color: #555; }
    .section-label { font-size: 0.75rem; font-weight: 500; letter-spacing: 1px; text-transform: uppercase; color: #999; margin-bottom: 0.5rem; }
    .stButton > button { width: 100%; background: #0f0f0f; color: white; border: none; border-radius: 10px; padding: 0.75rem; font-size: 1rem; font-weight: 500; }
    div[data-testid="stFileUploader"] { border: 1.5px dashed #ccc; border-radius: 12px; padding: 1rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model_path = "mnist_cnn.keras"

    if not os.path.exists(model_path):
        with st.spinner("No saved model found — training a fresh one (takes ~2 min)..."):
            (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
            x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]

            m = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(10, activation="softmax")
            ])
            m.compile(optimizer="adam",
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
            m.fit(x_train, y_train, epochs=5, batch_size=128,
                  validation_split=0.1, verbose=0)
            m.save(model_path)

    return tf.keras.models.load_model(model_path)


def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("L")
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img).astype("float32")
    if arr.mean() > 127:
        arr = 255.0 - arr
    arr = arr / 255.0
    return arr.reshape(1, 28, 28, 1)


st.markdown("""
<div class="title-block">
    <h1>🔢 MNIST Digit Classifier</h1>
    <p>Upload a photo of a handwritten digit — the model will classify it</p>
</div>
""", unsafe_allow_html=True)

model = load_model()

st.markdown('<div class="section-label">Upload image</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("", type=["png", "jpg", "jpeg", "webp"],
                             label_visibility="collapsed")

if uploaded:
    image = Image.open(uploaded)
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="section-label">Your image</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)

        arr_preview = np.array(image.convert("L").resize((28, 28), Image.LANCZOS))
        if arr_preview.mean() > 127:
            arr_preview = 255 - arr_preview
        st.markdown('<div class="section-label" style="margin-top:1rem">28×28 model input</div>',
                    unsafe_allow_html=True)
        st.image(arr_preview, width=112, clamp=True)

    with col2:
        st.markdown('<div class="section-label">Prediction</div>', unsafe_allow_html=True)

        inp = preprocess(image)
        probs = model.predict(inp, verbose=0)[0]
        predicted = int(np.argmax(probs))
        confidence = float(probs[predicted]) * 100

        st.markdown(f"""
        <div class="result-card">
            <div class="result-digit">{predicted}</div>
            <div class="result-conf">{confidence:.1f}% confidence</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">All probabilities</div>', unsafe_allow_html=True)
        for i, p in enumerate(probs):
            cols = st.columns([0.4, 4, 1])
            cols[0].markdown(f'<span style="font-size:0.8rem;color:#444;">{i}</span>',
                             unsafe_allow_html=True)
            bar_color = "#0f0f0f" if i == predicted else "#e0e0e0"
            cols[1].markdown(
                f'<div style="background:{bar_color};height:8px;border-radius:99px;'
                f'width:{p*100:.1f}%;margin-top:6px;"></div>',
                unsafe_allow_html=True)
            cols[2].markdown(
                f'<span style="font-size:0.8rem;color:#444;">{p*100:.1f}%</span>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="tip-box">
        <strong>Low confidence?</strong> Try writing the digit larger, with high contrast
        (dark pen on white paper), good lighting, and crop tightly around the digit.
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 0; color: #aaa;">
        Upload a <strong>.jpg</strong> or <strong>.png</strong> photo of a handwritten digit above
    </div>
    """, unsafe_allow_html=True)
