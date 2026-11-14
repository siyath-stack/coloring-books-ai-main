#!/usr/bin/env python3
"""
Adult Coloring Book Generator for Amazon KDP
Generates intricate line art designs and compiles into print-ready PDFs.

Version 2.1.0 - Enhanced Quality Assurance
- Automatic quality validation on every generated image
- Pre-flight dependency checks
- Post-generation verification
- Comprehensive error handling
- Prevents recurrence of known issues
"""

import os
import json
import re
import time
import random
import hashlib
import requests
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Tuple, Any
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Quality Assurance Constants
MIN_PRINT_WIDTH = 2550  # 8.5" at 300 DPI
MIN_PRINT_HEIGHT = 3300  # 11" at 300 DPI
REQUIRED_DPI = 300
MAX_ACCEPTABLE_COLORS = 2  # For line art (black & white only)

# Image Quality Constants
MIN_FILE_SIZE_BYTES = 10_000  # 10KB - smaller indicates generation issue
MAX_FILE_SIZE_BYTES = 10_000_000  # 10MB - larger may cause upload issues
RATE_LIMIT_DELAY_SECONDS = 2  # Delay between API calls to avoid rate limiting
MAX_PAGE_GENERATION_ATTEMPTS = 3
PRINT_SAFE_MARGIN_FRACTION = 0.02

# Prompt Variations (shared between generators)
PROMPT_VARIATIONS = [
    "",
    ", with a slightly different composition",
    ", with one simple supporting detail",
    ", with a clear centered composition",
]

COLORING_PAGE_STYLE = (
    "simple clean cartoon line art, like a classic children's coloring page; one large, "
    "friendly, recognizable subject centered on the page; thick smooth black ink outlines "
    "with consistent line weight; closed outlines around broad white areas left empty for "
    "coloring; plain pure-white paper with only a few simple supporting outlines"
)
COLORING_PAGE_NEGATIVE_PROMPT = (
    "color, gray, grayscale, shading, shadows, gradients, texture, hatching, stippling, "
    "photograph, photorealistic, 3d render, sketch, rough lines, thin scratchy lines, "
    "tiny details, intricate patterns, cluttered or filled background, black background, "
    "large filled areas, text, "
    "letters, words, captions, watermark, signature, logo"
)


def build_coloring_page_prompt(prompt: str) -> str:
    """Apply the shared clean, simple reference style to any theme description."""
    subject = prompt.split(",", maxsplit=1)[0]
    simplified_prompt = re.sub(
        r"\b(?:highly\s+)?(?:intricate|ornate|elaborate|complex|detailed|decorative)\b",
        "",
        subject,
        flags=re.IGNORECASE,
    )
    simplified_prompt = " ".join(simplified_prompt.split()).strip()
    return (
        f"Printable black-and-white coloring page. Subject: {simplified_prompt}. "
        f"Style: {COLORING_PAGE_STYLE}. Keep the described subject count; do not add "
        "unrelated characters or objects. Leave every shape white inside for coloring. "
        "No filled backgrounds, no solid black shadows, no gray, no shading, no texture, "
        "no tiny details, no text, and no border."
    )


# API Keys (set via environment variable)
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
HF_TOKEN = os.environ.get("HF_TOKEN")

# KDP Print Dimensions (in pixels at 300 DPI)
KDP_SIZES = {
    "8.5x11": (2550, 3300),      # Standard US Letter
    "8x10": (2400, 3000),         # Popular coloring book size
    "6x9": (1800, 2700),          # Trade paperback
    "8.25x8.25": (2475, 2475),    # Square format
}


# ==================== QUALITY ASSURANCE FUNCTIONS ====================

def validate_dependencies() -> Dict[str, bool]:
    """Check if all required dependencies are installed.

    Returns:
        Dict with dependency name and availability status
    """
    deps = {
        'opencv': False,
        'numpy': False,
        'PIL': False,
        'reportlab': False,
        'requests': True  # Already imported above
    }

    try:
        import cv2
        deps['opencv'] = True
        logger.debug(f"✓ OpenCV {cv2.__version__} available")
    except ImportError:
        logger.warning("✗ OpenCV not available - line art processing will be disabled")

    try:
        import numpy
        deps['numpy'] = True
        logger.debug(f"✓ NumPy {numpy.__version__} available")
    except ImportError:
        logger.warning("✗ NumPy not available - line art processing will be disabled")

    try:
        from PIL import Image
        deps['PIL'] = True
        logger.debug("✓ Pillow available")
    except ImportError:
        logger.warning("✗ Pillow not available - PDF generation may fail")

    try:
        from reportlab.pdfgen import canvas
        deps['reportlab'] = True
        logger.debug("✓ ReportLab available")
    except ImportError:
        logger.warning("✗ ReportLab not available - PDF generation will be disabled")

    return deps


def validate_image_quality(image_path: Path,
                          min_width: int = MIN_PRINT_WIDTH,
                          min_height: int = MIN_PRINT_HEIGHT,
                          check_binary: bool = True,
                          required_dpi: int = REQUIRED_DPI) -> Dict[str, Any]:
    """Validate that a generated image meets print quality standards.

    Args:
        image_path: Path to the image file
        min_width: Minimum acceptable width in pixels
        min_height: Minimum acceptable height in pixels
        check_binary: Whether to check for pure black & white
        required_dpi: Required DPI for print quality

    Returns:
        Dict with validation results and any issues found
    """
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return {
            'valid': False,
            'error': 'PIL/NumPy not available for validation',
            'warnings': []
        }

    results = {
        'valid': True,
        'warnings': [],
        'issues': [],
        'metrics': {}
    }

    try:
        img = Image.open(image_path)
        width, height = img.size

        # Check dimensions
        results['metrics']['width'] = width
        results['metrics']['height'] = height
        results['metrics']['physical_width'] = width / required_dpi
        results['metrics']['physical_height'] = height / required_dpi

        if width < min_width or height < min_height:
            results['valid'] = False
            results['issues'].append(
                f"CRITICAL: Image too small ({width}x{height}), needs {min_width}x{min_height} for print"
            )

        # Check DPI metadata
        dpi = img.info.get('dpi', (72, 72))
        results['metrics']['dpi'] = dpi

        if dpi[0] < required_dpi or dpi[1] < required_dpi:
            results['warnings'].append(
                f"DPI metadata is {dpi}, should be {required_dpi} for print"
            )

        # Check color mode
        results['metrics']['mode'] = img.mode

        # Check binary (pure black & white) if requested
        if check_binary:
            arr = np.array(img)
            unique_colors = len(np.unique(arr))
            results['metrics']['unique_colors'] = unique_colors

            if unique_colors > MAX_ACCEPTABLE_COLORS:
                results['warnings'].append(
                    f"Image has {unique_colors} colors, expected {MAX_ACCEPTABLE_COLORS} for line art"
                )
            ink_ratio = float(np.mean(arr < 128))
            results['metrics']['ink_ratio'] = ink_ratio
            if ink_ratio < 0.001:
                results['valid'] = False
                results['issues'].append("CRITICAL: Image is nearly blank")
            elif ink_ratio > 0.30:
                results['valid'] = False
                results['issues'].append("CRITICAL: Image has too much filled black area for coloring")

        # Check file size (should be reasonable, not too small or huge)
        file_size = image_path.stat().st_size
        results['metrics']['file_size_kb'] = file_size / 1024

        if file_size < MIN_FILE_SIZE_BYTES:
            results['warnings'].append(
                f"File size very small ({file_size/1024:.1f}KB), may indicate generation issue"
            )
        elif file_size > MAX_FILE_SIZE_BYTES:
            results['warnings'].append(
                f"File size very large ({file_size/1024/1024:.1f}MB), may cause upload issues"
            )

        # Log validation summary
        if results['valid'] and not results['warnings']:
            logger.debug(f"✓ Quality validation passed: {image_path.name}")
        elif results['warnings']:
            logger.warning(f"⚠ Quality warnings for {image_path.name}: {', '.join(results['warnings'])}")

    except Exception as e:
        results['valid'] = False
        results['error'] = f"Validation failed: {str(e)}"
        logger.error(f"Validation error for {image_path}: {e}")

    return results


def check_system_resources() -> Dict[str, Any]:
    """Check available system resources before generation.

    Returns:
        Dict with resource availability information
    """
    import shutil

    resources = {
        'disk_space_available': True,
        'memory_available': True,
        'warnings': []
    }

    # Check disk space in output directory
    try:
        stat = shutil.disk_usage(Path.cwd())
        free_gb = stat.free / (1024**3)
        resources['free_disk_gb'] = free_gb

        if free_gb < 1:
            resources['disk_space_available'] = False
            resources['warnings'].append(f"Low disk space: {free_gb:.2f} GB free")
        elif free_gb < 5:
            resources['warnings'].append(f"Limited disk space: {free_gb:.2f} GB free")
    except Exception as e:
        resources['warnings'].append(f"Could not check disk space: {e}")

    return resources


def preflight_checks(force_lineart: bool = False) -> bool:
    """Run all pre-flight checks before generation.

    Args:
        force_lineart: Whether line art processing is required

    Returns:
        True if all critical checks pass, False otherwise
    """
    logger.info("Running pre-flight checks...")

    all_ok = True

    # Check dependencies
    deps = validate_dependencies()

    if force_lineart and not (deps['opencv'] and deps['numpy']):
        logger.error("CRITICAL: Line art processing requires OpenCV and NumPy")
        logger.error("Install with: pip install opencv-python numpy")
        all_ok = False

    # Check system resources
    resources = check_system_resources()

    if not resources['disk_space_available']:
        logger.error("CRITICAL: Insufficient disk space")
        all_ok = False

    for warning in resources.get('warnings', []):
        logger.warning(warning)

    if all_ok:
        logger.info("✓ All pre-flight checks passed")
    else:
        logger.error("✗ Pre-flight checks failed - cannot proceed")

    return all_ok


def image_pixel_hash(image_path: Path) -> str:
    """Hash normalized pixels so visually identical pages can be detected."""
    from PIL import Image

    with Image.open(image_path) as image:
        normalized = image.convert("L")
        content = f"{normalized.width}x{normalized.height}".encode("ascii")
        return hashlib.sha256(content + normalized.tobytes()).hexdigest()


def print_safe_margin_issue(
    image_path: Path,
    margin_fraction: float = PRINT_SAFE_MARGIN_FRACTION,
) -> Optional[str]:
    """Return a reason when artwork extends into the page's trim-safe margin."""
    from PIL import Image

    with Image.open(image_path) as image:
        grayscale = image.convert("L")
        ink = grayscale.point(lambda pixel: 255 if pixel < 128 else 0)
        bounds = ink.getbbox()
        if bounds is None:
            return "page contains no visible line art"

        left, top, right, bottom = bounds
        margin_x = max(1, int(grayscale.width * margin_fraction))
        margin_y = max(1, int(grayscale.height * margin_fraction))
        if (
            left < margin_x
            or top < margin_y
            or grayscale.width - right < margin_x
            or grayscale.height - bottom < margin_y
        ):
            return (
                f"artwork is inside the {margin_fraction:.0%} trim-safe margin "
                f"(ink bounds: {bounds})"
            )

    return None


# ==================== END QUALITY ASSURANCE FUNCTIONS ====================

# ==================== THEME LOADING ====================

def load_themes(themes_file: Path = None) -> Dict[str, Dict]:
    """Load coloring book themes from JSON file.

    Args:
        themes_file: Path to themes JSON file (default: themes/themes.json)

    Returns:
        Dictionary of themes

    Raises:
        FileNotFoundError: If themes file not found
        json.JSONDecodeError: If themes file is invalid
    """
    if themes_file is None:
        # Default to themes/themes.json relative to this file
        script_dir = Path(__file__).parent
        themes_file = script_dir / "themes" / "themes.json"

    if not themes_file.exists():
        logger.error(f"Themes file not found: {themes_file}")
        raise FileNotFoundError(f"Themes file not found: {themes_file}")

    try:
        with open(themes_file, 'r') as f:
            themes = json.load(f)
        logger.debug(f"Loaded {len(themes)} themes from {themes_file}")
        return themes
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in themes file: {e}")
        raise


# Load themes at module level
try:
    THEMES = load_themes()
except FileNotFoundError:
    logger.warning("Themes file not found, using empty themes dict")
    THEMES = {}
except Exception as e:
    logger.warning(f"Error loading themes: {e}, using empty themes dict")
    THEMES = {}

# ==================== END THEME LOADING ====================

# Legacy theme data replaced with JSON configuration
# THEMES dictionary moved to themes/themes.json for easier maintenance



class ColoringBookGenerator:
    """Generate adult coloring book pages using AI."""

    def __init__(self, output_dir: str = "output", backend: str = "huggingface", force_lineart: bool = False, lineart_method: str = "enhanced"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.api_token = REPLICATE_API_TOKEN
        self.hf_token = HF_TOKEN
        self.backend = backend  # "huggingface", "pollinations", or "replicate"
        self.force_lineart = force_lineart  # Post-process to pure B&W line art
        self.lineart_method = lineart_method  # "enhanced", "standard", or "detailed"

    def generate_image_replicate(self, prompt: str, size: tuple = (2550, 3300)) -> Optional[str]:
        """Generate a coloring page using Replicate API."""

        full_prompt = build_coloring_page_prompt(prompt)

        try:
            # Using SDXL for high quality line art
            response = requests.post(
                "https://api.replicate.com/v1/predictions",
                headers={
                    "Authorization": f"Token {self.api_token}",
                    "Content-Type": "application/json"
                },
                json={
                    "version": "39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",  # SDXL
                    "input": {
                        "prompt": full_prompt,
                        "negative_prompt": COLORING_PAGE_NEGATIVE_PROMPT,
                        "width": min(size[0], 1024),  # SDXL max
                        "height": min(size[1], 1024),
                        "num_inference_steps": 30,
                        "guidance_scale": 7.5,
                    }
                },
                timeout=30
            )

            if response.status_code != 201:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return None

            prediction = response.json()
            prediction_id = prediction["id"]

            # Poll for completion
            for _ in range(60):  # 5 minute timeout
                time.sleep(5)

                status_response = requests.get(
                    f"https://api.replicate.com/v1/predictions/{prediction_id}",
                    headers={"Authorization": f"Token {self.api_token}"},
                    timeout=30
                )

                if status_response.status_code == 200:
                    result = status_response.json()
                    if result["status"] == "succeeded":
                        output = result.get("output")
                        if output:
                            return output[0] if isinstance(output, list) else output
                    elif result["status"] == "failed":
                        logger.error(f"Generation failed: {result.get('error')}")
                        return None

            logger.error("Generation timed out")
            return None

        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return None

    def generate_image_pollinations(self, prompt: str, size: tuple = (1024, 1024)) -> Optional[bytes]:
        """Generate a coloring page using Pollinations.ai (completely free, no API key)."""
        import urllib.parse

        full_prompt = build_coloring_page_prompt(prompt)

        # URL-encode the prompt
        encoded_prompt = urllib.parse.quote(full_prompt)

        # Pollinations.ai API - completely free, no auth needed
        seed = random.randrange(0, 2**32)
        url = (
            f"https://image.pollinations.ai/prompt/{encoded_prompt}"
            f"?width={size[0]}&height={size[1]}&nologo=true&model=flux&seed={seed}"
        )

        try:
            logger.info(f"  Requesting from Pollinations.ai...")
            response = requests.get(url, timeout=120)

            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                if 'image' in content_type or len(response.content) > 10000:
                    logger.info(f"  Success! Got {len(response.content)} bytes")
                    return response.content
                else:
                    logger.warning(f"  Unexpected response: {content_type}")
            else:
                logger.warning(f"  Pollinations error: {response.status_code}")

        except requests.exceptions.Timeout:
            logger.warning("  Pollinations timed out, retrying...")
            try:
                response = requests.get(url, timeout=180)
                if response.status_code == 200:
                    return response.content
            except Exception as e:
                logger.warning(f"  Retry failed: {e}")
        except Exception as e:
            logger.warning(f"  Pollinations error: {e}")

        return None

    def generate_image_huggingface(self, prompt: str, size: tuple = (1024, 1024)) -> Optional[bytes]:
        """Generate a coloring page using HuggingFace free inference API."""

        full_prompt = build_coloring_page_prompt(prompt)

        # Try the new router API with different models
        models = [
            "black-forest-labs/FLUX.1-schnell",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "stabilityai/stable-diffusion-2-1",
        ]

        for model in models:
            try:
                logger.info(f"  Trying model: {model}")

                # Use new router.huggingface.co endpoint
                response = requests.post(
                    f"https://router.huggingface.co/hf-inference/models/{model}",
                    headers={
                        "Authorization": f"Bearer {self.hf_token}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "inputs": full_prompt,
                        "parameters": {
                            "negative_prompt": COLORING_PAGE_NEGATIVE_PROMPT,
                            "num_inference_steps": 25,
                            "guidance_scale": 7.5,
                        }
                    },
                    timeout=180
                )

                if response.status_code == 200:
                    # Response is the image bytes directly
                    content_type = response.headers.get('content-type', '')
                    if 'image' in content_type:
                        logger.info(f"  Success with {model}")
                        return response.content
                    else:
                        logger.warning(f"  Unexpected response type: {content_type}")
                elif response.status_code == 503:
                    # Model loading, wait and retry
                    logger.info(f"  Model loading, waiting...")
                    time.sleep(30)
                    # Retry same model
                    response = requests.post(
                        f"https://router.huggingface.co/hf-inference/models/{model}",
                        headers={
                            "Authorization": f"Bearer {self.hf_token}",
                            "Content-Type": "application/json"
                        },
                        json={"inputs": full_prompt},
                        timeout=180
                    )
                    if response.status_code == 200 and 'image' in response.headers.get('content-type', ''):
                        return response.content
                else:
                    logger.warning(f"  Model {model} failed: {response.status_code} - {response.text[:300]}")

            except requests.exceptions.Timeout:
                logger.warning(f"  Model {model} timed out")
            except Exception as e:
                logger.warning(f"  Model {model} error: {e}")

        return None

    def upscale_to_print_quality(self, image_bytes: bytes, target_size: tuple = (2550, 3300)) -> bytes:
        """Upscale image to print quality resolution if needed.

        Args:
            image_bytes: Input image as bytes
            target_size: Target size for print (default: 8.5x11" at 300 DPI)

        Returns:
            Upscaled image as bytes

        Raises:
            RuntimeError: If upscaling fails critically (prevents bad outputs)
        """
        try:
            import cv2
            import numpy as np
            from PIL import Image
            from io import BytesIO

            # Load image
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                logger.error("CRITICAL: Failed to decode image for upscaling")
                raise RuntimeError("Image decode failed - cannot ensure print quality")

            original_size = (img.shape[1], img.shape[0])

            # Check if upscaling needed
            if img.shape[0] < target_size[1] or img.shape[1] < target_size[0]:
                logger.info(f"  Upscaling from {img.shape[1]}x{img.shape[0]} to {target_size[0]}x{target_size[1]} for print quality")
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)

                # QUALITY ASSURANCE: Verify upscaling actually worked
                if img.shape[1] != target_size[0] or img.shape[0] != target_size[1]:
                    logger.error(f"CRITICAL: Upscaling failed - got {img.shape[1]}x{img.shape[0]}, expected {target_size}")
                    raise RuntimeError("Upscaling verification failed")

                # Convert to PIL and save with DPI
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                pil_img.info['dpi'] = (300, 300)

                output = BytesIO()
                pil_img.save(output, format='PNG', dpi=(300, 300))

                # QUALITY ASSURANCE: Verify DPI was set
                output.seek(0)
                verify_img = Image.open(output)
                actual_dpi = verify_img.info.get('dpi', (0, 0))
                if actual_dpi[0] < 300 or actual_dpi[1] < 300:
                    logger.warning(f"DPI metadata issue: got {actual_dpi}, expected (300, 300)")

                output.seek(0)
                return output.getvalue()
            else:
                # Image already large enough, just set DPI metadata
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                output = BytesIO()
                pil_img.save(output, format='PNG', dpi=(300, 300))
                return output.getvalue()

        except ImportError as e:
            logger.error(f"CRITICAL: Missing dependencies for upscaling: {e}")
            logger.error("Cannot guarantee print quality without OpenCV/NumPy/Pillow")
            raise RuntimeError("Missing required dependencies - install opencv-python numpy Pillow")
        except Exception as e:
            logger.error(f"CRITICAL: Upscaling failed: {e}")
            logger.error("This means the image WILL NOT meet print quality standards!")
            raise RuntimeError(f"Upscaling failed critically: {e}")

    def convert_to_coloring_page(self, image_bytes: bytes, method: str = "enhanced",
                                 target_size: tuple = (2550, 3300)) -> bytes:
        """Convert generated artwork to clean black-and-white coloring-page outlines.

        Args:
            image_bytes: Input image as bytes
            method: 'enhanced' (bold lines), 'standard' (medium lines), or 'detailed' (fine lines)
            target_size: Target size for print (default: 8.5x11" at 300 DPI)

        Returns:
            Processed image as bytes

        Raises:
            RuntimeError: If processing fails critically
        """
        try:
            import cv2
            import numpy as np
            from io import BytesIO
            from PIL import Image

            # Load image from bytes
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                logger.error("CRITICAL: Failed to decode image for line art conversion")
                raise RuntimeError("Image decode failed during line art conversion")

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if method == "enhanced":
                blur_sigma, canny_thresholds, thickness = 2.0, (70, 160), 1
            elif method == "standard":
                blur_sigma, canny_thresholds, thickness = 1.5, (55, 130), 0
            elif method == "detailed":
                blur_sigma, canny_thresholds, thickness = 1.0, (40, 100), 0
            else:
                raise ValueError(f"Unknown line art method: {method}")

            blurred = cv2.GaussianBlur(gray, (0, 0), blur_sigma)
            edges = cv2.Canny(blurred, *canny_thresholds)

            # Remove scattered speckle noise from AI outputs without erasing the main
            # subject outlines. Using a relative threshold keeps genuine detail while
            # stripping tiny disconnected artifacts around the subject.
            if thickness:
                cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                edges = cv2.dilate(edges, cleanup_kernel, iterations=thickness)

            count, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
            if count > 1:
                component_areas = stats[1:, cv2.CC_STAT_AREA]
                largest_component_area = max(int(component_areas.max()), 1)
                min_component_area = max(24, int(largest_component_area * 0.015))
                border_margin = max(1, min(edges.shape[0], edges.shape[1]) // 200)
                for idx in range(1, count):
                    area = stats[idx, cv2.CC_STAT_AREA]
                    x, y, w, h, _ = stats[idx]
                    touching_border = (
                        x <= border_margin or y <= border_margin
                        or x + w >= edges.shape[1] - border_margin
                        or y + h >= edges.shape[0] - border_margin
                    )
                    if area < min_component_area:
                        edges[labels == idx] = 0
                    elif touching_border and area < min_component_area * 10:
                        edges[labels == idx] = 0

            result = cv2.bitwise_not(edges)
            source_height, source_width = result.shape
            # Scale the already-binarized art without blurring its outlines, and
            # center it on the print page instead of stretching its proportions.
            target_width, target_height = target_size
            scale = min(target_width / source_width, target_height / source_height)
            resized_width = max(1, round(source_width * scale))
            resized_height = max(1, round(source_height * scale))
            if (resized_width, resized_height) != (source_width, source_height):
                logger.info(
                    f"  Fitting {source_width}x{source_height} artwork to "
                    f"{target_width}x{target_height} for print quality"
                )
                result = cv2.resize(
                    result,
                    (resized_width, resized_height),
                    interpolation=cv2.INTER_NEAREST
                )
            page = np.full((target_height, target_width), 255, dtype=np.uint8)
            x_offset = (target_width - resized_width) // 2
            y_offset = (target_height - resized_height) // 2
            page[y_offset:y_offset + resized_height, x_offset:x_offset + resized_width] = result
            result = page

            # QUALITY ASSURANCE: Verify pure binary output
            unique_values = len(np.unique(result))
            if unique_values > MAX_ACCEPTABLE_COLORS:
                logger.warning(f"Line art has {unique_values} colors instead of 2 - applying stronger threshold")
                # Apply more aggressive binarization
                _, result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)
                unique_values = len(np.unique(result))
                if unique_values > MAX_ACCEPTABLE_COLORS:
                    logger.error(f"CRITICAL: Could not achieve pure binary output ({unique_values} colors)")

            # Convert to PIL to set DPI metadata
            pil_img = Image.fromarray(result)
            pil_img.info['dpi'] = (300, 300)  # Set 300 DPI for print quality

            # Save to bytes with DPI metadata
            output = BytesIO()
            pil_img.save(output, format='PNG', dpi=(300, 300))

            # QUALITY ASSURANCE: Verify final output
            output.seek(0)
            verify_img = Image.open(output)
            final_size = verify_img.size
            final_dpi = verify_img.info.get('dpi', (0, 0))

            if final_size[0] != target_size[0] or final_size[1] != target_size[1]:
                logger.error(f"CRITICAL: Final image size wrong: {final_size} vs {target_size}")
                raise RuntimeError("Line art conversion produced wrong size output")

            if final_dpi[0] < 300 or final_dpi[1] < 300:
                logger.warning(f"DPI metadata may not be set correctly: {final_dpi}")

            output.seek(0)
            return output.getvalue()

        except ImportError:
            logger.error("CRITICAL: OpenCV not installed - cannot do line art conversion")
            logger.error("Install with: pip install opencv-python numpy")
            raise RuntimeError("Missing dependencies for line art conversion")
        except RuntimeError:
            # Re-raise runtime errors (our own quality checks)
            raise
        except Exception as e:
            logger.error(f"CRITICAL: Line art conversion failed: {e}")
            raise RuntimeError(f"Line art conversion failed: {e}")

    def download_image(self, url: str, filepath: Path) -> bool:
        """Download image from URL."""
        try:
            response = requests.get(url, timeout=60)
            if response.status_code == 200:
                filepath.write_bytes(response.content)
                return True
        except Exception as e:
            logger.error(f"Error downloading image: {e}")
        return False

    def _generate_image_bytes(self, prompt: str, size: tuple = (1536, 1536)) -> Optional[bytes]:
        """Generate image bytes using configured backend.

        Centralized backend selection logic shared between generators.

        Args:
            prompt: Image generation prompt
            size: Image size tuple (width, height)

        Returns:
            Image bytes on success, None on failure
        """
        if self.backend == "pollinations":
            return self.generate_image_pollinations(prompt, size=size)
        elif self.backend == "huggingface":
            return self.generate_image_huggingface(prompt)
        else:  # replicate
            import tempfile
            import uuid
            image_url = self.generate_image_replicate(prompt)
            if image_url:
                # Download to temp file then read bytes
                temp_path = Path(tempfile.gettempdir()) / f"temp_{uuid.uuid4()}.png"
                try:
                    if self.download_image(image_url, temp_path):
                        image_bytes = temp_path.read_bytes()
                        temp_path.unlink()  # Clean up
                        return image_bytes
                except Exception as e:
                    logger.error(f"Error reading temp image: {e}")
                    if temp_path.exists():
                        temp_path.unlink()
            return None

    def _post_process_image(self, image_bytes: bytes) -> bytes:
        """Post-process image (upscaling and/or line art).

        Extracted helper method to reduce code duplication.

        Args:
            image_bytes: Raw image bytes from generation

        Returns:
            Processed image bytes
        """
        if self.force_lineart:
            return self.convert_to_coloring_page(image_bytes, method=self.lineart_method)
        else:
            return self.upscale_to_print_quality(image_bytes)

    def _save_book_metadata(self, book_dir: Path, title: str, theme: str,
                           num_pages: int, generated_pages: list, timestamp: str,
                           failed_pages: Optional[list] = None):
        """Save book metadata to JSON file.

        Extracted helper method for better organization.

        Args:
            book_dir: Book directory path
            title: Book title
            theme: Theme name
            num_pages: Total pages requested
            generated_pages: List of successfully generated pages
            timestamp: Generation timestamp
        """
        metadata = {
            "title": title,
            "theme": theme,
            "pages": num_pages,
            "generated": len(generated_pages),
            "failed_pages": failed_pages or [],
            "timestamp": timestamp,
            "pages_data": generated_pages
        }

        metadata_file = book_dir / "metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2))

    def generate_book(
        self,
        theme: str,
        num_pages: int = 30,
        book_title: str = None,
        resume_from: Optional[Path] = None,
    ) -> Path:
        """Generate a complete coloring book."""

        # PRE-FLIGHT CHECKS: Ensure system is ready
        if not preflight_checks(force_lineart=self.force_lineart):
            raise RuntimeError("Pre-flight checks failed - cannot generate book safely")

        if theme not in THEMES:
            raise ValueError(f"Unknown theme: {theme}. Available: {list(THEMES.keys())}")

        theme_data = THEMES[theme]
        title = book_title or theme_data["name"]
        safe_title = title.replace(" ", "_").replace(":", "")

        if num_pages < 1:
            raise ValueError("Page count must be at least one.")

        if resume_from is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            book_dir = self.output_dir / f"{safe_title}_{timestamp}"
            book_dir.mkdir(parents=True, exist_ok=True)
            existing_pages = []
        else:
            book_dir = Path(resume_from).expanduser()
            metadata_file = book_dir / "metadata.json"
            if not book_dir.is_dir() or not metadata_file.is_file():
                raise FileNotFoundError(
                    f"Cannot resume: '{book_dir}' is not a generated book directory."
                )
            try:
                previous_metadata = json.loads(metadata_file.read_text())
            except (OSError, json.JSONDecodeError) as error:
                raise RuntimeError(f"Cannot resume from invalid metadata: {error}") from error
            if (
                previous_metadata.get("title") != title
                or previous_metadata.get("theme") != theme
                or previous_metadata.get("pages") != num_pages
            ):
                raise ValueError(
                    "Resume settings must match the existing book's title, theme, and page count."
                )
            timestamp = previous_metadata.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
            existing_pages = previous_metadata.get("pages_data", [])
            if not isinstance(existing_pages, list):
                raise RuntimeError("Cannot resume: metadata pages_data must be a list.")

        images_dir = book_dir / "images"
        images_dir.mkdir(exist_ok=True)

        logger.info(f"Generating coloring book: {title}")
        logger.info(f"Theme: {theme}, Pages: {num_pages}")
        logger.info(f"Output: {book_dir}")

        # Generate pages
        prompts = theme_data["prompts"]
        generated = []
        accepted_hashes = set()
        accepted_page_numbers = set()
        for page_data in existing_pages:
            if not isinstance(page_data, dict):
                continue
            page_number = page_data.get("page")
            expected_name = (
                f"page_{page_number:03d}.png"
                if isinstance(page_number, int) and 1 <= page_number <= num_pages
                else None
            )
            if (
                expected_name is None
                or page_data.get("file") != expected_name
                or page_number in accepted_page_numbers
            ):
                continue
            image_path = images_dir / expected_name
            if not image_path.is_file():
                continue
            validation = validate_image_quality(
                image_path,
                check_binary=self.force_lineart,
            )
            if not validation["valid"] or print_safe_margin_issue(image_path):
                continue
            fingerprint = image_pixel_hash(image_path)
            if fingerprint in accepted_hashes:
                continue
            accepted_hashes.add(fingerprint)
            accepted_page_numbers.add(page_number)
            generated.append(page_data)

        def save_progress():
            generated_page_numbers = {page["page"] for page in generated}
            failed_pages = [
                page_number
                for page_number in range(1, num_pages + 1)
                if page_number not in generated_page_numbers
            ]
            self._save_book_metadata(
                book_dir,
                title,
                theme,
                num_pages,
                generated,
                timestamp,
                failed_pages=failed_pages,
            )
            return failed_pages

        save_progress()
        consecutive_provider_failures = 0
        provider_unavailable = False

        for i in range(num_pages):
            if i + 1 in accepted_page_numbers:
                continue
            base_prompt = prompts[i % len(prompts)]
            image_path = images_dir / f"page_{i+1:03d}.png"
            candidate_path = images_dir / f".page_{i+1:03d}_candidate.png"
            accepted = False

            for attempt in range(1, MAX_PAGE_GENERATION_ATTEMPTS + 1):
                prompt = base_prompt + random.choice(PROMPT_VARIATIONS)
                if attempt > 1:
                    prompt += (
                        ", use a different pose and composition, keep the full subject "
                        "inside a generous white margin, and do not draw a frame"
                    )

                logger.info(
                    f"Generating page {i+1}/{num_pages} "
                    f"(attempt {attempt}/{MAX_PAGE_GENERATION_ATTEMPTS})..."
                )
                image_bytes = self._generate_image_bytes(prompt, size=(1536, 1536))
                if not image_bytes:
                    consecutive_provider_failures += 1
                    logger.warning(f"  ✗ No image returned for page {i+1}, attempt {attempt}")
                    if consecutive_provider_failures >= MAX_PAGE_GENERATION_ATTEMPTS:
                        provider_unavailable = True
                        break
                else:
                    consecutive_provider_failures = 0
                    try:
                        candidate_path.write_bytes(self._post_process_image(image_bytes))
                        validation = validate_image_quality(
                            candidate_path,
                            check_binary=self.force_lineart,
                        )
                        if not validation["valid"]:
                            logger.warning(
                                f"  ✗ Page {i+1} failed image validation: "
                                f"{validation.get('issues', [])}"
                            )
                        else:
                            margin_issue = print_safe_margin_issue(candidate_path)
                            if margin_issue:
                                logger.warning(f"  ✗ Page {i+1} rejected: {margin_issue}")
                            else:
                                fingerprint = image_pixel_hash(candidate_path)
                                if fingerprint in accepted_hashes:
                                    logger.warning(
                                        f"  ✗ Page {i+1} rejected: duplicate artwork"
                                    )
                                else:
                                    candidate_path.replace(image_path)
                                    accepted_hashes.add(fingerprint)
                                    generated.append({
                                        "page": i + 1,
                                        "prompt": prompt,
                                        "file": image_path.name,
                                        "quality_metrics": validation.get("metrics", {}),
                                    })
                                    accepted_page_numbers.add(i + 1)
                                    accepted = True
                                    logger.info(
                                        f"  ✓ Saved unique, print-safe art: {image_path.name}"
                                    )
                    except (OSError, RuntimeError, ValueError) as error:
                        logger.warning(f"  ✗ Page {i+1} attempt {attempt} failed: {error}")
                    finally:
                        if candidate_path.exists():
                            candidate_path.unlink()

                if accepted or provider_unavailable:
                    break
                if attempt < MAX_PAGE_GENERATION_ATTEMPTS:
                    time.sleep(RATE_LIMIT_DELAY_SECONDS)

            if not accepted:
                logger.error(
                    f"  ✗ Could not create a unique, print-safe image for page {i+1}"
                )
            if provider_unavailable:
                logger.error("Image provider failed repeatedly; stopping to avoid wasted requests.")
                break
            save_progress()
            if i < num_pages - 1:
                time.sleep(RATE_LIMIT_DELAY_SECONDS)

        # Save metadata using helper method
        failed_pages = save_progress()

        logger.info(f"Generated {len(generated)}/{num_pages} pages")
        logger.info(f"Book saved to: {book_dir}")
        if failed_pages:
            raise RuntimeError(
                f"Book incomplete: generated {len(generated)}/{num_pages} unique, "
                f"print-safe pages. Missing pages: {failed_pages}. "
                f"Partial output and diagnostics were retained at '{book_dir}'."
            )

        return book_dir

    def create_pdf(self, book_dir: Path, size: str = "8.5x11") -> Optional[Path]:
        """Compile images into a print-ready PDF."""
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.pdfgen import canvas
            from reportlab.lib.units import inch
            from PIL import Image
        except ImportError:
            logger.error("Install reportlab and Pillow: pip install reportlab Pillow")
            return None

        images_dir = book_dir / "images"
        metadata_file = book_dir / "metadata.json"
        if not images_dir.exists():
            raise RuntimeError(f"Images directory not found: {images_dir}")
        if not metadata_file.exists():
            raise RuntimeError(f"Book metadata is missing: {metadata_file}")

        try:
            metadata = json.loads(metadata_file.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Could not read book metadata: {error}") from error

        expected_pages = metadata.get("pages")
        generated_pages = metadata.get("generated")
        pages_data = metadata.get("pages_data", [])
        images = sorted(images_dir.glob("page_*.png"))
        if (
            not isinstance(expected_pages, int)
            or expected_pages < 1
            or not isinstance(pages_data, list)
            or any(
                not isinstance(page, dict) or not isinstance(page.get("file"), str)
                for page in pages_data
            )
            or generated_pages != expected_pages
            or len(pages_data) != expected_pages
            or len(images) != expected_pages
        ):
            raise RuntimeError(
                f"Refusing to compile an incomplete book: expected {expected_pages} "
                f"pages, metadata records {generated_pages}, and {len(images)} image files."
            )

        recorded_files = [page.get("file") for page in pages_data]
        if (
            len(set(recorded_files)) != expected_pages
            or set(recorded_files) != {image.name for image in images}
        ):
            raise RuntimeError("Book metadata does not match the generated image files.")

        seen_hashes = {}
        for image_path in images:
            fingerprint = image_pixel_hash(image_path)
            if fingerprint in seen_hashes:
                raise RuntimeError(
                    f"Refusing to compile duplicate artwork on "
                    f"'{seen_hashes[fingerprint]}' and '{image_path.name}'."
                )
            seen_hashes[fingerprint] = image_path.name

        for image_path in images:
            validation = validate_image_quality(
                image_path,
                check_binary=self.force_lineart,
            )
            if not validation["valid"]:
                raise RuntimeError(
                    f"Refusing to compile invalid page '{image_path.name}': "
                    f"{validation.get('issues', [])}"
                )
            margin_issue = print_safe_margin_issue(image_path)
            if margin_issue:
                raise RuntimeError(
                    f"Refusing to compile unsafe page '{image_path.name}': {margin_issue}"
                )

        # Create PDF
        title = metadata.get("title", "Coloring Book")
        pdf_path = book_dir / f"{title.replace(' ', '_')}.pdf"
        temporary_pdf = pdf_path.with_name(f".{pdf_path.stem}.tmp.pdf")

        # Use letter size for KDP
        page_width, page_height = letter

        try:
            c = canvas.Canvas(str(temporary_pdf), pagesize=letter)

            # Add title page
            c.setFont("Helvetica-Bold", 36)
            c.drawCentredString(page_width/2, page_height/2 + 50, title)
            c.setFont("Helvetica", 18)
            c.drawCentredString(page_width/2, page_height/2 - 20, "Adult Coloring Book")
            c.showPage()

            # Add coloring pages
            margin = 0.5 * inch
            content_width = page_width - 2 * margin
            content_height = page_height - 2 * margin

            for img_path in images:
                img = Image.open(img_path)
                img_width, img_height = img.size

                # Calculate scaling to fit page
                scale = min(content_width / img_width, content_height / img_height)
                new_width = img_width * scale
                new_height = img_height * scale

                # Center on page
                x = (page_width - new_width) / 2
                y = (page_height - new_height) / 2

                c.drawImage(str(img_path), x, y, new_width, new_height)
                c.showPage()
            c.save()
            temporary_pdf.replace(pdf_path)
        finally:
            if temporary_pdf.exists():
                temporary_pdf.unlink()
        logger.info(f"PDF created: {pdf_path}")

        return pdf_path


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate Adult Coloring Books for KDP")
    parser.add_argument("--theme", choices=list(THEMES.keys()), default="mandalas",
                       help="Coloring book theme")
    parser.add_argument("--pages", type=int, default=30,
                       help="Number of pages (default: 30)")
    parser.add_argument("--title", type=str, default=None,
                       help="Custom book title")
    parser.add_argument("--output", type=str, default="output",
                       help="Output directory")
    parser.add_argument("--pdf", action="store_true",
                       help="Also create PDF")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume a matching incomplete book directory")
    parser.add_argument("--list-themes", action="store_true",
                       help="List available themes")
    parser.add_argument("--backend", choices=["pollinations", "huggingface", "replicate"], default="pollinations",
                       help="Image generation backend (default: pollinations - free, unlimited)")
    parser.add_argument("--force-lineart", dest="force_lineart", action="store_true",
                       help="Post-process images to clean black-and-white coloring-page art (default)")
    parser.add_argument("--no-lineart", dest="force_lineart", action="store_false",
                       help="Skip black-and-white image post-processing")
    parser.set_defaults(force_lineart=True)
    parser.add_argument("--lineart-method", choices=["enhanced", "standard", "detailed"], default="enhanced",
                       help="Outline weight: enhanced (bold), standard (medium), detailed (fine)")

    args = parser.parse_args()

    if args.list_themes:
        print("\nAvailable Themes:")
        print("-" * 40)
        for key, data in THEMES.items():
            print(f"  {key}: {data['name']}")
            print(f"    Prompts: {len(data['prompts'])}")
        return

    generator = ColoringBookGenerator(
        output_dir=args.output,
        backend=args.backend,
        force_lineart=args.force_lineart,
        lineart_method=args.lineart_method
    )

    book_dir = generator.generate_book(
        theme=args.theme,
        num_pages=args.pages,
        book_title=args.title,
        resume_from=args.resume,
    )

    if args.pdf and book_dir:
        generator.create_pdf(book_dir)


if __name__ == "__main__":
    main()
