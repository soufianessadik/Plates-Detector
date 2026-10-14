import io
import time

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps
from ultralytics import YOLO

# ------------------------------------------------------------------ setup
st.set_page_config(page_title="Plate Detector", page_icon="🔍", layout="wide")

ACCENT = "#3B82F6"
MODEL_PATH = "best.pt"

st.markdown(
    f"""
    <style>
    #MainMenu, footer {{visibility: hidden;}}
    .block-container {{padding-top: 2rem; max-width: 1200px;}}
    .hero {{
        padding: 1.6rem 1.8rem; border-radius: 16px; margin-bottom: 1.4rem;
        background: linear-gradient(135deg, rgba(59,130,246,.20), rgba(59,130,246,.03));
        border: 1px solid rgba(59,130,246,.35);
    }}
    .hero h1 {{margin: 0; font-size: 2.1rem; font-weight: 800;}}
    .hero p {{margin: .45rem 0 0; color: #9CA3AF; font-size: 1.02rem;}}
    .badge {{
        display: inline-block; margin-top: .8rem; padding: .25rem .7rem; border-radius: 999px;
        background: rgba(59,130,246,.15); color: {ACCENT}; font-size: .8rem; font-weight: 600;
        border: 1px solid rgba(59,130,246,.4);
    }}
    .stat {{
        background: #111827; border: 1px solid #1F2937; border-radius: 12px;
        padding: .9rem 1.1rem; text-align: center;
    }}
    .stat .v {{font-size: 1.7rem; font-weight: 800; color: {ACCENT}; line-height: 1.2;}}
    .stat .l {{color: #9CA3AF; font-size: .85rem; margin-top: .2rem;}}
    .step {{
        background: #111827; border: 1px solid #1F2937; border-radius: 12px;
        padding: 1.1rem 1.2rem; height: 100%;
    }}
    .step .n {{
        display: inline-flex; width: 28px; height: 28px; border-radius: 50%;
        align-items: center; justify-content: center; font-weight: 700;
        background: {ACCENT}; color: white; margin-bottom: .6rem;
    }}
    .step .t {{font-weight: 700; margin-bottom: .25rem;}}
    .step .d {{color: #9CA3AF; font-size: .9rem;}}
    .plate-meta {{display: flex; justify-content: space-between; align-items: center; margin-top: .3rem;}}
    .plate-meta .name {{font-weight: 700;}}
    .plate-meta .conf {{
        padding: .15rem .6rem; border-radius: 999px; font-weight: 700; font-size: .85rem;
        background: rgba(59,130,246,.15); color: {ACCENT};
    }}
    .section {{font-size: 1.25rem; font-weight: 700; margin: 1.4rem 0 .6rem;}}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


def get_font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_boxes(img, boxes, confs):
    out = img.copy()
    d = ImageDraw.Draw(out)
    lw = max(2, round(min(img.size) / 200))
    font = get_font(max(14, round(min(img.size) / 30)))
    for i, ((x1, y1, x2, y2), c) in enumerate(zip(boxes, confs), 1):
        d.rectangle([x1, y1, x2, y2], outline=ACCENT, width=lw)
        label = f" #{i}  {c:.0%} "
        tb = d.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = y1 - th - 3 * lw if y1 - th - 3 * lw > 0 else y2 + lw
        d.rectangle([x1, ty, x1 + tw + 2 * lw, ty + th + 3 * lw], fill=ACCENT)
        d.text((x1 + lw, ty + lw), label, fill="white", font=font)
    return out


def crop_plate(img, box, pad=0.08, width=360):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1, y1 = max(0, x1 - pad * w), max(0, y1 - pad * h)
    x2, y2 = min(img.width, x2 + pad * w), min(img.height, y2 + pad * h)
    c = img.crop((int(x1), int(y1), int(x2), int(y2)))
    scale = width / max(1, c.width)
    return c.resize((width, max(1, int(c.height * scale))), Image.LANCZOS)


def stat(value, label):
    return f"<div class='stat'><div class='v'>{value}</div><div class='l'>{label}</div></div>"


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### Settings")
    conf = st.slider(
        "Confidence threshold", 0.05, 0.95, 0.25, 0.05,
        help="Only detections above this confidence are shown. "
             "Lower it to see more plates, raise it to keep only the surest ones.",
    )
    st.markdown("### Model performance")
    st.caption("Measured on 398 real-world test photos the model never saw during training.")
    c1, c2 = st.columns(2)
    c1.metric("Precision", "92.2%")
    c2.metric("Recall", "89.2%")
    c1.metric("mAP50", "91.0%")
    c2.metric("mAP50-95", "56.4%")
    st.markdown("### About")
    st.caption(
        "YOLO26-s detector trained on about 9,300 images of license plates "
        "from many countries, in parking, street and close-up views."
    )

# ------------------------------------------------------------------ header
st.markdown(
    """
    <div class="hero">
        <h1>🔍 Plate Detector</h1>
        <p>Upload a photo of a vehicle and the model finds every license plate in it.</p>
        <span class="badge">YOLO26 · Real-time detection</span>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded = st.file_uploader("Upload a car photo", type=["jpg", "jpeg", "png", "webp"])

if uploaded is None:
    cols = st.columns(3)
    steps = [
        ("Upload", "Choose a photo of one or several vehicles (JPG, PNG or WEBP)."),
        ("Detect", "The model locates every license plate in a fraction of a second."),
        ("Review", "See each plate boxed on the photo, then cropped and enlarged below."),
    ]
    for col, (i, (t, d)) in zip(cols, enumerate(steps, 1)):
        col.markdown(
            f"<div class='step'><div class='n'>{i}</div><div class='t'>{t}</div>"
            f"<div class='d'>{d}</div></div>",
            unsafe_allow_html=True,
        )
    st.stop()

# ------------------------------------------------------------------ detection
img = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")

with st.spinner("Detecting plates..."):
    model = load_model()
    t0 = time.perf_counter()
    r = model.predict(img, conf=conf, imgsz=640, verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000

boxes = r.boxes.xyxy.cpu().numpy()
confs = r.boxes.conf.cpu().numpy()
order = np.argsort(-confs)
boxes, confs = boxes[order], confs[order]
result = draw_boxes(img, boxes, confs)

s1, s2, s3 = st.columns(3)
s1.markdown(stat(len(boxes), "plates detected"), unsafe_allow_html=True)
s2.markdown(stat(f"{confs.max():.0%}" if len(confs) else "–", "highest confidence"),
            unsafe_allow_html=True)
s3.markdown(stat(f"{ms:.0f} ms", "processing time"), unsafe_allow_html=True)

st.markdown("<div class='section'>Result</div>", unsafe_allow_html=True)
left, right = st.columns(2)
with left:
    st.caption("Original")
    st.image(img, use_container_width=True)
with right:
    st.caption("Detected plates")
    st.image(result, use_container_width=True)
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    st.download_button("Download result", buf.getvalue(),
                       file_name="plates_detected.png", mime="image/png")

st.markdown("<div class='section'>Detected plates</div>", unsafe_allow_html=True)
if len(boxes) == 0:
    st.info("No plate found. Try lowering the confidence threshold in the sidebar.")
else:
    cols = st.columns(3)
    for i, (b, c) in enumerate(zip(boxes, confs), 1):
        with cols[(i - 1) % 3]:
            with st.container(border=True):
                st.image(crop_plate(img, b), use_container_width=True)
                st.markdown(
                    f"<div class='plate-meta'><span class='name'>Plate #{i}</span>"
                    f"<span class='conf'>{c:.0%}</span></div>",
                    unsafe_allow_html=True,
                )
