import cv2
import numpy as np
from deepface import DeepFace
from skimage.feature import local_binary_pattern
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
import re
import easyocr
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager
from scipy.spatial.distance import cosine
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn


# ---------------------------------------------------------
# CONFIGURATION & CONSTANTS
# ---------------------------------------------------------
@dataclass
class VerificationConfig:
    laplacian_threshold: float = 5.0
    lbp_threshold: float = 0.150
    face_padding_ratio: float = 0.20
    min_confidence: float = 0.0
    similarity_threshold: float = 0.6
    strict: bool = False
    detector_backend: str = 'retinaface'
    model_name: str = 'ArcFace'
    distance_metric: str = 'cosine'
    max_image_size: Tuple[int, int] = (1920, 1080)

    def validate(self):
        if not (0 < self.laplacian_threshold <= 100):
            raise ValueError("laplacian_threshold must be between 0 and 100")
        if not (0 <= self.lbp_threshold <= 1):
            raise ValueError("lbp_threshold must be between 0 and 1")
        if not (0 < self.similarity_threshold <= 1):
            raise ValueError("similarity_threshold must be between 0 and 1")


def setup_logging(log_level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    if not logger.handlers:
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler('verification_pipeline.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = setup_logging()

# Initialize EasyOCR reader globally
ocr_reader = easyocr.Reader(['en', 'hi'], gpu=False)

# Labels to ignore during name extraction
IGNORE_KEYWORDS = [
    "SURNAME", "GIVEN", "NAME", "NAMES", "THIA", "GAYA", "NAM",
    "उपनाम", "दिया", "गया", "नाम", "PASSPORT", "NATIONALITY",
    "DATE", "BIRTH", "SEX", "INDIAN", "P<"
]


# ---------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------
def validate_image_path(image_path: str) -> Path:
    path = Path(image_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {image_path}")
    if path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}:
        raise ValueError(f"Unsupported image format: {path.suffix}")

    return path


def load_image(image_path: str, color_space: str = 'BGR') -> np.ndarray:
    validate_image_path(image_path)

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        raise ValueError(f"Invalid image dimensions: {w}x{h}")

    if color_space == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def resize_image_if_needed(img: np.ndarray, max_size: Tuple[int, int]) -> np.ndarray:
    h, w = img.shape[:2]
    max_w, max_h = max_size

    if w > max_w or h > max_h:
        scale = min(max_w / w, max_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        logger.warning(f"Resizing image from {w}x{h} to {new_w}x{new_h}")
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img


def save_image(image: np.ndarray, output_path: str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(output_path), image)
    if not success:
        raise IOError(f"Failed to save image to: {output_path}")

    logger.info(f"Image saved to: {output_path}")
    return output_path


@contextmanager
def temporary_file(path: str):
    try:
        yield path
    finally:
        if os.path.exists(path):
            try:
                os.remove(path)
                logger.debug(f"Temporary file cleaned up: {path}")
            except OSError as e:
                logger.warning(f"Failed to delete temporary file {path}: {e}")


# ---------------------------------------------------------
# FACE MATCHING & LIVENESS FUNCTIONS
# ---------------------------------------------------------
def extract_id_face(
        id_card_path: str,
        save_path: str = "extracted_face.jpg",
        config: Optional[VerificationConfig] = None
) -> str:
    if config is None:
        config = VerificationConfig()

    logger.info(f"Extracting face from ID card: {id_card_path}")

    img = load_image(id_card_path)
    img = resize_image_if_needed(img, config.max_image_size)

    rotations = [
        (None, "0°"),
        (cv2.ROTATE_90_CLOCKWISE, "90° Clockwise"),
        (cv2.ROTATE_180, "180°"),
        (cv2.ROTATE_90_COUNTERCLOCKWISE, "270° Clockwise")
    ]

    for rot_code, rot_label in rotations:
        try:
            candidate_img = img.copy() if rot_code is None else cv2.rotate(img, rot_code)

            with temporary_file("temp_rotated_id.jpg") as temp_path:
                cv2.imwrite(temp_path, candidate_img)

                face_objs = DeepFace.extract_faces(
                    img_path=temp_path,
                    detector_backend=config.detector_backend,
                    enforce_detection=False
                )

                if len(face_objs) > 0 and face_objs[0]['confidence'] > config.min_confidence:
                    facial_area = face_objs[0]['facial_area']
                    x, y, w, h = (
                        facial_area['x'],
                        facial_area['y'],
                        facial_area['w'],
                        facial_area['h']
                    )

                    pad_x = int(w * config.face_padding_ratio)
                    pad_y = int(h * config.face_padding_ratio)

                    img_h, img_w = candidate_img.shape[:2]
                    y1 = max(0, y - pad_y)
                    y2 = min(img_h, y + h + pad_y)
                    x1 = max(0, x - pad_x)
                    x2 = min(img_w, x + w + pad_x)

                    cropped_face = candidate_img[y1:y2, x1:x2]

                    if cropped_face.size == 0:
                        logger.warning(f"Cropped face is empty at rotation {rot_label}")
                        continue

                    save_image(cropped_face, save_path)
                    logger.info(f"Face extracted successfully (Orientation: {rot_label})")

                    return save_path

        except Exception as e:
            logger.debug(f"Face detection failed at rotation {rot_label}: {e}")
            continue

    raise ValueError("Failed to detect face on ID card across all 4 orientations")


def check_liveness(
        image_path: str,
        config: Optional[VerificationConfig] = None
) -> Tuple[bool, str, Dict[str, float]]:
    if config is None:
        config = VerificationConfig()

    logger.info(f"Running liveness check on: {image_path}")

    img = load_image(image_path)
    img = resize_image_if_needed(img, config.max_image_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    metrics = {}

    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    metrics['blur_score'] = float(laplacian_var)

    if laplacian_var < config.laplacian_threshold:
        msg = f"Spoof detected: Image too blurry (Score: {laplacian_var:.2f})"
        logger.warning(msg)
        if config.strict:
            return False, msg, metrics

    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)

    uniformity = float(np.sum(hist ** 2))
    metrics['texture_uniformity'] = uniformity

    if uniformity > config.lbp_threshold:
        msg = f"Spoof detected: Screen/Print texture (Score: {uniformity:.4f})"
        logger.warning(msg)
        if config.strict:
            return False, msg, metrics

    logger.info("Liveness check passed")
    return True, "Liveness verification passed", metrics


def verify_faces(
        id_face_path: str,
        selfie_path: str,
        config: Optional[VerificationConfig] = None
) -> Dict[str, Any]:
    if config is None:
        config = VerificationConfig()

    logger.info("Starting facial verification...")

    try:
        img1 = cv2.imread(id_face_path)
        if img1 is None:
            raise ValueError(f"Could not read extracted face at '{id_face_path}'")

        img1_resized = cv2.resize(img1, (112, 112))
        cv2.imwrite(id_face_path, img1_resized)

        emb1_objs = DeepFace.represent(
            img_path=id_face_path,
            model_name=config.model_name,
            detector_backend='skip',
            enforce_detection=False
        )

        emb2_objs = DeepFace.represent(
            img_path=selfie_path,
            model_name=config.model_name,
            detector_backend='skip',
            enforce_detection=False
        )

        vec1 = emb1_objs[0]["embedding"]
        vec2 = emb2_objs[0]["embedding"]

        dist = float(cosine(vec1, vec2))
        similarity_score = round((1.0 - dist) * 100, 2)
        verified = dist < 0.68

        return {
            "verified": verified,
            "similarity_score": similarity_score,
            "threshold": 0.68,
            "distance": round(dist, 4)
        }

    except Exception as e:
        logger.error(f"Facial verification failed: {e}")
        raise ValueError(f"Verification error: {e}")


# ---------------------------------------------------------
# MRZ & VIZ FUNCTIONS
# ---------------------------------------------------------
def sanitize_mrz_name(text: str) -> str:
    if not text:
        return ""
    char_map = {
        '1': 'I',
        '0': 'O',
        '8': 'B',
        '5': 'S',
        '2': 'Z'
    }
    for digit, letter in char_map.items():
        text = text.replace(digit, letter)
    return text


def clean_mrz_string(text: str) -> str:
    if not text:
        return ""
    return text.replace(" ", "").upper()


def extract_names_from_mrz(mrz_line_1: str):
    clean_mrz = clean_mrz_string(mrz_line_1)

    if clean_mrz.startswith('P<') or clean_mrz.startswith('P>'):
        name_section = clean_mrz[5:]
    elif clean_mrz.startswith('P'):
        name_section = clean_mrz[4:]
    else:
        name_section = clean_mrz

    name_section = name_section.rstrip('<')

    if '<<' in name_section:
        surname_part, given_name_part = name_section.split('<<', 1)
    else:
        surname_part = name_section
        given_name_part = ""

    surname = sanitize_mrz_name(surname_part.replace('<', ' ').strip())
    given_names = sanitize_mrz_name(given_name_part.replace('<', ' ').strip())

    return surname, given_names


def format_viz_for_mrz(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r'[^A-Za-z0-9]', '', text)
    return cleaned.upper()


def extract_mrz_line_1(ocr_results) -> str:
    mrz_line_1_pattern = re.compile(r'^P[A-Z0-9<]{3,}[<]+')

    for item in ocr_results:
        if len(item) == 3:
            _, text, _ = item
        else:
            continue

        clean_text = text.replace(" ", "").upper()

        if mrz_line_1_pattern.match(clean_text):
            return sanitize_mrz_name(clean_text)

    return ""


def extract_viz_data(ocr_results) -> dict:
    viz_data = {
        "passport_num": None,
        "surname": None,
        "given_names": None,
        "nationality": None,
        "sex": None,
        "date_of_birth": None,
        "date_of_issue": None,
        "date_of_expiry": None,
        "place_of_birth": None,
        "place_of_issue": None
    }

    raw_lines = [text.strip() for _, text, conf in ocr_results if text.strip()]
    full_text = " ".join(raw_lines)

    # 1. Extract Passport Number
    passport_match = re.search(r'\b[A-Z][0-9]{7}\b', full_text)
    if passport_match:
        viz_data["passport_num"] = passport_match.group(0)

    # 2. Extract Dates
    dates = re.findall(r'\b\d{2}[\/\.-]\d{2}[\/\.-]\d{4}\b', full_text)
    if len(dates) >= 3:
        viz_data["date_of_birth"] = dates[0]
        viz_data["date_of_issue"] = dates[1]
        viz_data["date_of_expiry"] = dates[2]
    elif len(dates) == 2:
        viz_data["date_of_birth"] = dates[0]
        viz_data["date_of_expiry"] = dates[1]

    # 3. Extract Sex
    sex_match = re.search(r'\b(SEX|GENDER|लिंग)\b[:\s]*([M|F|X])\b', full_text, re.IGNORECASE)
    if sex_match:
        viz_data["sex"] = sex_match.group(2).upper()
    else:
        if re.search(r'\bM\b', full_text):
            viz_data["sex"] = "M"
        elif re.search(r'\bF\b', full_text):
            viz_data["sex"] = "F"

    # 4. Extract Nationality
    if "INDIAN" in full_text.upper():
        viz_data["nationality"] = "INDIAN"

    # 5. Robust Name Extraction with IGNORE_KEYWORDS filter
    for i, line in enumerate(raw_lines):
        line_upper = line.upper()

        # --- SURNAME ---
        if any(keyword in line_upper for keyword in ["SURNAME", "उपनाम"]):
            clean_same_line = re.sub(r'^(SURNAME|उपनाम)[:\s]*', '', line_upper).strip()
            if clean_same_line and not any(k in clean_same_line for k in ["SURNAME", "उपनाम"]):
                viz_data["surname"] = clean_same_line
            else:
                for offset in [1, 2]:
                    if i + offset < len(raw_lines):
                        candidate = raw_lines[i + offset].strip().upper()
                        if candidate and not any(k in candidate for k in IGNORE_KEYWORDS):
                            viz_data["surname"] = candidate
                            break

        # --- GIVEN NAMES ---
        if any(keyword in line_upper for keyword in ["GIVEN NAME", "GIVEN NAMES", "THIA GAYA NAM", "दिया गया नाम"]):
            clean_same_line = re.sub(r'^(GIVEN NAME|GIVEN NAMES|THIA GAYA NAM|दिया गया नाम)[:\s]*', '', line_upper).strip()
            if clean_same_line and not any(k in clean_same_line for k in ["GIVEN", "NAME", "NAM"]):
                viz_data["given_names"] = clean_same_line
            else:
                for offset in [1, 2]:
                    if i + offset < len(raw_lines):
                        candidate = raw_lines[i + offset].strip().upper()
                        if candidate and not any(k in candidate for k in IGNORE_KEYWORDS):
                            viz_data["given_names"] = candidate
                            break

    return viz_data


def verify_mrz_with_name(mrz_line_1: str, viz_surname: str, viz_given_names: str) -> dict:
    mrz_surname, mrz_given_names = extract_names_from_mrz(mrz_line_1)

    formatted_viz_surname = format_viz_for_mrz(viz_surname)
    formatted_viz_given_names = format_viz_for_mrz(viz_given_names)

    surname_match = (format_viz_for_mrz(mrz_surname) == formatted_viz_surname)
    given_name_match = (format_viz_for_mrz(mrz_given_names) == formatted_viz_given_names)

    full_match = surname_match and given_name_match

    return {
        "verified": full_match,
        "surname_match": surname_match,
        "given_name_match": given_name_match,
        "extracted_mrz": {
            "surname": mrz_surname,
            "given_names": mrz_given_names
        },
        "formatted_viz": {
            "surname": formatted_viz_surname,
            "given_names": formatted_viz_given_names
        }
    }


def process_mrz_viz_validation(id_image_path: str) -> dict:
    ocr_results = ocr_reader.readtext(id_image_path)
    extracted_mrz_line_1 = extract_mrz_line_1(ocr_results)
    extracted_viz = extract_viz_data(ocr_results)

    # Parse fallback candidates directly from MRZ if available
    mrz_surname_fallback, mrz_given_fallback = "", ""
    if extracted_mrz_line_1:
        mrz_surname_fallback, mrz_given_fallback = extract_names_from_mrz(extracted_mrz_line_1)

    # Fetch extracted VIZ values
    viz_surname = extracted_viz.get("surname")
    viz_given_names = extracted_viz.get("given_names")

    # Use MRZ fallbacks if VIZ values are empty or captured label keywords
    if not viz_surname or viz_surname in ["SURNAME", "उपनाम"]:
        viz_surname = mrz_surname_fallback

    if not viz_given_names or viz_given_names in ["GIVEN NAME", "GIVEN NAMES", "NAME"]:
        viz_given_names = mrz_given_fallback

    if extracted_mrz_line_1:
        mrz_viz_result = verify_mrz_with_name(extracted_mrz_line_1, viz_surname, viz_given_names)
        return {
            "mrz_found": True,
            "mrz_line_1": extracted_mrz_line_1,
            "mrz_viz_verification": mrz_viz_result
        }
    else:
        return {
            "mrz_found": False,
            "error": "Could not locate MRZ Line 1 in image",
            "extracted_viz": extracted_viz
        }


# ---------------------------------------------------------
# FASTAPI APPLICATION
# ---------------------------------------------------------
app = FastAPI(title="Identity & MRZ Verification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite default port
        "http://localhost:3000",  # Create-React-App default port
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/verify")
async def verify_identity(
        id_card: UploadFile = File(...),
        selfie: UploadFile = File(...)
):
    config = VerificationConfig()

    id_card_path = f"temp_{id_card.filename}"
    selfie_path = f"temp_{selfie.filename}"
    extracted_face_path = f"temp_extracted_{id_card.filename}"

    try:
        with open(id_card_path, "wb") as f:
            f.write(await id_card.read())

        with open(selfie_path, "wb") as f:
            f.write(await selfie.read())

        # 1. Liveness check on Selfie
        is_live, liveness_msg, liveness_metrics = check_liveness(selfie_path, config)
        if not is_live:
            raise HTTPException(status_code=400, detail=f"Liveness check failed: {liveness_msg}")

        # 2. Extract Face & Face Matching
        extracted_face_file = extract_id_face(id_card_path, save_path=extracted_face_path, config=config)
        face_matching_result = verify_faces(extracted_face_file, selfie_path, config)

        # 3. MRZ & VIZ Validation
        mrz_viz_result = process_mrz_viz_validation(id_card_path)

        return {
            "status": "SUCCESS",
            "face_matching_score": {
                "similarity_score": f"{face_matching_result['similarity_score']}%",
                "verified": face_matching_result["verified"],
                "distance": face_matching_result["distance"],
                "threshold": face_matching_result["threshold"]
            },
            "mrz_viz_matching": mrz_viz_result
        }

    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        for path in [id_card_path, selfie_path, extracted_face_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


if __name__ == "__main__":
    uvicorn.run(app)
