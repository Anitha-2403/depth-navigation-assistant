import streamlit as st
import torch
import numpy as np
import cv2

from PIL import Image
from transformers import pipeline


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Depth Navigation Assistant",
    page_icon="🧭",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #777;
        font-size: 17px;
        margin-bottom: 25px;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 10px;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #f5f5f5;
        margin-top: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🧭 AI Depth Navigation Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Single Image Depth Estimation using Depth Anything V2'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Controls")

st.sidebar.write(
    """
    This application estimates relative depth from
    a single RGB image and performs basic navigation
    analysis.
    """
)

input_method = st.sidebar.radio(
    "Select Input Method",
    [
        "📁 Upload Image",
        "📷 Camera"
    ]
)


# ============================================================
# LOAD DEPTH ANYTHING V2
# ============================================================

@st.cache_resource
def load_depth_model():

    device = 0 if torch.cuda.is_available() else -1

    model = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=device
    )

    return model


# ============================================================
# MODEL LOADING
# ============================================================

with st.spinner("Loading Depth Anything V2 model..."):

    depth_model = load_depth_model()


# ============================================================
# INPUT IMAGE
# ============================================================

image = None


if input_method == "📁 Upload Image":

    uploaded_file = st.file_uploader(
        "Upload a road, outdoor, or indoor image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")


elif input_method == "📷 Camera":

    camera_image = st.camera_input(
        "Take a picture using your camera"
    )

    if camera_image is not None:

        image = Image.open(
            camera_image
        ).convert("RGB")


# ============================================================
# DEPTH PROCESSING FUNCTIONS
# ============================================================

def normalize_depth(depth):

    depth = depth.astype(np.float32)

    minimum = np.min(depth)
    maximum = np.max(depth)

    normalized = (
        depth - minimum
    ) / (
        maximum - minimum + 1e-8
    )

    return normalized


def create_depth_visualization(depth):

    normalized = normalize_depth(depth)

    depth_uint8 = (
        normalized * 255
    ).astype(np.uint8)

    colored_depth = cv2.applyColorMap(
        depth_uint8,
        cv2.COLORMAP_INFERNO
    )

    colored_depth = cv2.cvtColor(
        colored_depth,
        cv2.COLOR_BGR2RGB
    )

    return colored_depth


def create_depth_overlay(image, depth):

    normalized = normalize_depth(depth)

    depth_uint8 = (
        normalized * 255
    ).astype(np.uint8)

    colored_depth = cv2.applyColorMap(
        depth_uint8,
        cv2.COLORMAP_INFERNO
    )

    colored_depth = cv2.cvtColor(
        colored_depth,
        cv2.COLOR_BGR2RGB
    )

    original = np.array(image)

    original = cv2.resize(
        original,
        (
            colored_depth.shape[1],
            colored_depth.shape[0]
        )
    )

    overlay = cv2.addWeighted(
        original,
        0.55,
        colored_depth,
        0.45,
        0
    )

    return overlay


def analyze_regions(depth):

    height, width = depth.shape

    # Divide image into three vertical regions

    left = depth[
        :,
        :width // 3
    ]

    center = depth[
        :,
        width // 3:2 * width // 3
    ]

    right = depth[
        :,
        2 * width // 3:
    ]

    left_value = np.mean(left)

    center_value = np.mean(center)

    right_value = np.mean(right)

    return (
        left_value,
        center_value,
        right_value
    )


def classify_depth(depth):

    normalized = normalize_depth(depth)

    # Depth Anything produces relative depth.
    # We use percentile thresholds rather than
    # interpreting values as meters.

    near_threshold = np.percentile(
        normalized,
        30
    )

    far_threshold = np.percentile(
        normalized,
        70
    )

    near_ratio = np.mean(
        normalized <= near_threshold
    )

    far_ratio = np.mean(
        normalized >= far_threshold
    )

    medium_ratio = (
        1.0
        - near_ratio
        - far_ratio
    )

    return (
        near_ratio,
        medium_ratio,
        far_ratio
    )


def get_navigation_result(
    depth,
    left,
    center,
    right
):

    normalized = normalize_depth(depth)

    # Lower values are treated as relatively
    # closer regions for this visualization.

    near_threshold = np.percentile(
        normalized,
        30
    )

    height, width = normalized.shape

    # Focus more strongly on the lower-middle
    # area because it is more relevant to a
    # ground-level navigation path.

    lower_start = int(
        height * 0.45
    )

    center_start = int(
        width * 0.33
    )

    center_end = int(
        width * 0.67
    )

    navigation_area = normalized[
        lower_start:,
        center_start:center_end
    ]

    center_near_ratio = np.mean(
        navigation_area <= near_threshold
    )

    # Navigation heuristic

    if center_near_ratio >= 0.40:

        message = "⚠️ OBSTACLE AHEAD"
        status = "warning"

    else:

        # Compare relative region depth

        region_values = {
            "left": left,
            "center": center,
            "right": right
        }

        closest_region = min(
            region_values,
            key=region_values.get
        )

        if closest_region == "center":

            message = "🟢 PATH CLEAR"
            status = "success"

        elif closest_region == "left":

            message = "➡️ CONSIDER MOVING RIGHT"
            status = "info"

        else:

            message = "⬅️ CONSIDER MOVING LEFT"
            status = "info"

    return (
        message,
        status,
        center_near_ratio
    )


def draw_regions(image, depth):

    original = np.array(image).copy()

    height, width, _ = original.shape

    # Resize depth to image dimensions

    depth_resized = cv2.resize(
        depth,
        (
            width,
            height
        )
    )

    normalized = normalize_depth(
        depth_resized
    )

    near_threshold = np.percentile(
        normalized,
        30
    )

    # Region boundaries

    x1 = width // 3
    x2 = 2 * width // 3

    # Draw vertical region boundaries

    cv2.line(
        original,
        (x1, 0),
        (x1, height),
        (255, 255, 255),
        2
    )

    cv2.line(
        original,
        (x2, 0),
        (x2, height),
        (255, 255, 255),
        2
    )

    # Labels

    cv2.putText(
        original,
        "LEFT",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        original,
        "CENTER",
        (x1 + 20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        original,
        "RIGHT",
        (x2 + 20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # Highlight near pixels in the lower part
    # of the scene

    lower_start = int(
        height * 0.45
    )

    near_mask = (
        normalized[lower_start:] <= near_threshold
    )

    overlay = original.copy()

    lower_overlay = overlay[
        lower_start:
    ]

    lower_overlay[
        near_mask
    ] = [255, 80, 80]

    overlay[
        lower_start:
    ] = lower_overlay

    result = cv2.addWeighted(
        original,
        0.70,
        overlay,
        0.30,
        0
    )

    return result


# ============================================================
# PROCESS IMAGE
# ============================================================

if image is not None:

    st.divider()

    st.subheader("📷 Input Image")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Original RGB Image",
            use_container_width=True
        )

    with st.spinner(
        "Estimating depth using Depth Anything V2..."
    ):

        # Run model

        result = depth_model(
            image
        )

        depth_image = result["depth"]

        # Convert PIL depth image to numpy

        depth = np.array(
            depth_image
        ).astype(
            np.float32
        )

    with col2:

        depth_visualization = (
            create_depth_visualization(
                depth
            )
        )

        st.image(
            depth_visualization,
            caption="Estimated Relative Depth",
            use_container_width=True
        )


    # ========================================================
    # DEPTH OVERLAY
    # ========================================================

    st.divider()

    st.subheader(
        "🌈 Depth Overlay"
    )

    overlay = create_depth_overlay(
        image,
        depth
    )

    st.image(
        overlay,
        caption="Original Image + Depth Visualization",
        use_container_width=True
    )


    # ========================================================
    # REGION ANALYSIS
    # ========================================================

    st.divider()

    st.subheader(
        "🧭 Navigation Region Analysis"
    )

    left, center, right = analyze_regions(
        depth
    )

    # Normalize region values for display

    all_values = np.array([
        left,
        center,
        right
    ])

    minimum = np.min(
        all_values
    )

    maximum = np.max(
        all_values
    )

    normalized_regions = (
        all_values - minimum
    ) / (
        maximum - minimum + 1e-8
    )

    left_display = normalized_regions[0]
    center_display = normalized_regions[1]
    right_display = normalized_regions[2]


    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "⬅️ Left Region",
            f"{left_display:.2f}"
        )

    with col2:

        st.metric(
            "⬆️ Center Region",
            f"{center_display:.2f}"
        )

    with col3:

        st.metric(
            "➡️ Right Region",
            f"{right_display:.2f}"
        )


    st.caption(
        "These values represent relative depth within "
        "the current image and are not distances in meters."
    )


    # ========================================================
    # NEAR / MEDIUM / FAR
    # ========================================================

    st.divider()

    st.subheader(
        "📊 Near / Medium / Far Analysis"
    )

    (
        near_ratio,
        medium_ratio,
        far_ratio
    ) = classify_depth(
        depth
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "🔴 Near",
            f"{near_ratio * 100:.1f}%"
        )

    with col2:

        st.metric(
            "🟡 Medium",
            f"{medium_ratio * 100:.1f}%"
        )

    with col3:

        st.metric(
            "🟢 Far",
            f"{far_ratio * 100:.1f}%"
        )


    # ========================================================
    # NAVIGATION RESULT
    # ========================================================

    st.divider()

    st.subheader(
        "🚦 Navigation Assistance"
    )

    (
        message,
        status,
        center_near_ratio
    ) = get_navigation_result(
        depth,
        left,
        center,
        right
    )


    if status == "warning":

        st.error(
            message
        )

    elif status == "success":

        st.success(
            message
        )

    else:

        st.info(
            message
        )


    st.write(
        f"**Center navigation area near ratio:** "
        f"{center_near_ratio * 100:.1f}%"
    )


    # ========================================================
    # REGION VISUALIZATION
    # ========================================================

    st.divider()

    st.subheader(
        "🗺️ Navigation Region Visualization"
    )

    region_image = draw_regions(
        image,
        depth
    )

    st.image(
        region_image,
        caption=(
            "Left / Center / Right Regions "
            "with Near-Region Highlighting"
        ),
        use_container_width=True
    )


    # ========================================================
    # PROJECT INFORMATION
    # ========================================================

    st.divider()

    with st.expander(
        "ℹ️ About This Project"
    ):

        st.write(
            """
            This project estimates relative depth from a
            single RGB image using the pretrained
            Depth Anything V2 model.

            The estimated depth is then analyzed to identify
            relatively near, medium and far regions.

            The image is divided into left, center and right
            regions to provide basic navigation assistance.

            Important:
            Depth Anything V2 provides relative depth in this
            application. The displayed values should not be
            interpreted as physical distance measurements in
            meters.

            This application is intended for educational and
            demonstration purposes and is not an autonomous
            driving or safety-critical navigation system.
            """
        )


else:

    st.info(
        "👆 Please upload an image or capture an image "
        "using the camera to start depth estimation."
    )