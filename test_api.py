import io
import requests
from PIL import Image
import numpy as np

BASE_URL = "http://localhost:5000"


def test_health():
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health check: {response.status_code}")
        print(response.json())
    except Exception as e:
        print(f"Health check failed: {e}")


def _make_dummy_leaf_image() -> bytes:
    """Generates a plausible-ish green leaf-toned image so /predict has
    something realistic to chew on instead of pure random noise."""
    arr = np.random.randint(40, 140, size=(128, 128, 3), dtype=np.uint8)
    arr[:, :, 1] = np.clip(arr[:, :, 1].astype(int) + 60, 0, 255)  # boost green channel
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


def test_predict(image_path: str = None):
    image_bytes = open(image_path, "rb").read() if image_path else _make_dummy_leaf_image()
    files = {"image": ("leaf.jpg", image_bytes, "image/jpeg")}
    try:
        response = requests.post(f"{BASE_URL}/predict", files=files)
        print(f"\nPredict check: {response.status_code}")
        result = response.json()
        print(result)

        if response.status_code == 200:
            assert "confidence" in result, "Missing confidence field"
            assert 0 <= result["confidence"] <= 100, (
                f"Confidence out of range: {result['confidence']} "
                "(if this fails, check for a double-softmax bug)"
            )
            print("Sanity checks passed.")
    except Exception as e:
        print(f"Predict check failed: {e}")


if __name__ == "__main__":
    test_health()
    # Pass a real labeled leaf image path for a meaningful confidence check,
    # e.g. test_predict("samples/tomato_early_blight_01.jpg")
    test_predict()