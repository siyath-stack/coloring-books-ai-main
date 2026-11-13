import os
import re
import json
import time
import math
import random
import urllib.parse
import requests
import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image, ImageEnhance, ImageDraw, ImageFont

ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN")

COVER_MODEL = os.environ.get("COVER_MODEL", "@cf/bytedance/stable-diffusion-xl-lightning")
INTERIOR_MODEL = os.environ.get("INTERIOR_MODEL", "@cf/leonardo/lucid-origin")

PAGE_WIDTH = 2550
PAGE_HEIGHT = 3300

session = requests.Session()
CLOUDFLARE_QUOTA_EXHAUSTED = False
POLLINATIONS_FALLBACK_ACTIVE = False


class CloudflareQuotaExceeded(RuntimeError):
    """Raised when Cloudflare's daily AI allocation is exhausted."""


class CloudflarePromptRejected(RuntimeError):
    """Raised when Cloudflare's safety filter rejects an image prompt."""


class ImageProviderUnavailable(RuntimeError):
    """Raised when no configured image provider can return an image."""


def cloudflare_quota_error(response):
    """Return a descriptive quota error when Cloudflare reports exhausted usage."""
    try:
        body = response.json()
    except ValueError:
        body = {}

    errors = body.get("errors", []) if isinstance(body, dict) else []
    details = " ".join(
        f"{error.get('code', '')} {error.get('message', '')}"
        for error in errors
        if isinstance(error, dict)
    )
    if any(token in details.lower() for token in ("4006", "daily free allocation", "neurons")):
        return details or response.text[:500]
    return None


def cloudflare_prompt_rejection(response):
    """Return Cloudflare's prompt-safety rejection detail, if present."""
    try:
        body = response.json()
    except ValueError:
        return None

    errors = body.get("errors", []) if isinstance(body, dict) else []
    for error in errors:
        if not isinstance(error, dict):
            continue
        message = str(error.get("message", ""))
        if error.get("code") == 3030 or "input prompt contains nsfw content" in message.lower():
            return f"{error.get('code', '')} {message}".strip()
    return None


def pollinations_free_limit_wait(response):
    """Return Pollinations' requested wait for its per-IP free image limit."""
    try:
        body = response.json()
    except ValueError:
        body = {}

    error = body.get("error", {}) if isinstance(body, dict) else {}
    if response.status_code != 402 or error.get("code") != "FREE_LIMIT":
        return None

    retry_after = response.headers.get("Retry-After", "")
    if retry_after.isdigit():
        return max(1, min(int(retry_after), 300))

    message = str(error.get("message", ""))
    match = re.search(r"retry after\s+(\d+)\s*s", message, re.IGNORECASE)
    if match:
        return max(1, min(int(match.group(1)), 300))

    return 45


def fetch_pollinations_image(prompt, width=1024, height=1024):
    """Generate an image through Pollinations when Cloudflare's quota is exhausted."""
    encoded_prompt = urllib.parse.quote(prompt, safe="")
    url = (
        f"https://image.pollinations.ai/prompt/{encoded_prompt}"
        f"?width={width}&height={height}&nologo=true&model=flux"
    )
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = session.get(url, timeout=180)
        except requests.RequestException as error:
            raise ImageProviderUnavailable(
                f"Pollinations image request failed: {error}"
            ) from error

        wait_seconds = pollinations_free_limit_wait(response)
        if wait_seconds is not None:
            if attempt == max_attempts:
                raise ImageProviderUnavailable(
                    "Pollinations free image limit persisted after "
                    f"{max_attempts} attempts. Retry after the per-IP limit resets."
                )
            wait_seconds += 1
            print(
                "[Warning] Pollinations free image limit reached; "
                f"retrying in {wait_seconds} seconds "
                f"(attempt {attempt + 1}/{max_attempts})."
            )
            time.sleep(wait_seconds)
            continue

        if response.status_code != 200:
            raise ImageProviderUnavailable(
                f"Pollinations rejected the image request (HTTP {response.status_code}): "
                f"{response.text[:300]}"
            )

        try:
            with Image.open(BytesIO(response.content)) as generated:
                generated.verify()
        except (OSError, ValueError) as error:
            raise ImageProviderUnavailable(
                f"Pollinations returned an invalid image: {error}"
            ) from error

        return response.content

    raise ImageProviderUnavailable("Pollinations image request failed without a response.")


def fetch_book_image(prompt, model, model_options=None):
    """Prefer Cloudflare, switching to Pollinations after daily quota exhaustion."""
    global POLLINATIONS_FALLBACK_ACTIVE

    if POLLINATIONS_FALLBACK_ACTIVE or CLOUDFLARE_QUOTA_EXHAUSTED:
        return fetch_pollinations_image(prompt)

    try:
        return fetch_cloudflare_image(prompt, model, model_options=model_options)
    except CloudflareQuotaExceeded:
        POLLINATIONS_FALLBACK_ACTIVE = True
        print(
            "[Warning] Cloudflare's daily AI allocation is exhausted; "
            "switching image generation to Pollinations."
        )
        return fetch_pollinations_image(prompt)
    except CloudflarePromptRejected as error:
        POLLINATIONS_FALLBACK_ACTIVE = True
        print(
            "[Warning] Cloudflare's prompt safety filter rejected this image; "
            f"switching image generation to Pollinations ({error})."
        )
        return fetch_pollinations_image(prompt)


def get_font(size):
    system_font_paths = [
        "Roboto-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF/Roboto-Regular.ttf",
        "arial.ttf"
    ]
    for path in system_font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue

    try:
        url = "https://github.com/googlefonts/roboto/raw/main/src/hinted/Roboto-Regular.ttf"
        r = session.get(url, timeout=5)
        if r.status_code == 200:
            with open("Roboto-Regular.ttf", "wb") as f:
                f.write(r.content)
            return ImageFont.truetype("Roboto-Regular.ttf", size)
    except Exception:
        pass

    return ImageFont.load_default()

def draw_wrapped_centered_text(draw, text, center_x, start_y, font, fill="black", max_width=1800, line_spacing=18):
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        test_line = " ".join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        left, top, right, bottom = bbox
        w = right - left
        if w <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]

    if current_line:
        lines.append(" ".join(current_line))

    y = start_y
    for line in lines:
        draw.text((center_x, y), line, fill=fill, font=font, anchor="mm")
        bbox = draw.textbbox((0, 0), line, font=font)
        left, top, right, bottom = bbox
        line_height = bottom - top
        y += line_height + line_spacing
    return y

# ---------------------------------------------------------------------------
# OPENCV LINE ART CLEANER & MORPHOLOGICAL DILATION
# ---------------------------------------------------------------------------
def clean_and_thicken_line_art(gray_pil_img):
    """
    Convert to grayscale, threshold at 140, then gently thicken black strokes.
    """
    img_np = np.array(gray_pil_img.convert("L"))

    # Thresholding: pixels > 140 become white; all others become black.
    _, binary = cv2.threshold(img_np, 140, 255, cv2.THRESH_BINARY)

    # Invert binary image so lines are white on black background for dilation
    inv_binary = cv2.bitwise_not(binary)

    # Keep outlines printable without magnifying small generation artifacts.
    kernel = np.ones((3, 3), np.uint8)
    thick_inv = cv2.dilate(inv_binary, kernel, iterations=1)

    # Invert back to black lines on white background
    thick_binary = cv2.bitwise_not(thick_inv)

    return Image.fromarray(thick_binary)

def add_print_safe_margin(gray_pil_img):
    pixels = np.asarray(gray_pil_img.convert("L"), dtype=np.uint8)
    height, width = pixels.shape
    resized = Image.fromarray(pixels).resize(
        (int(width * 0.75), int(height * 0.75)),
        Image.Resampling.LANCZOS,
    )
    framed = Image.new("L", (width, height), "white")
    framed.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return framed

def remove_isolated_specks(gray_pil_img):
    pixels = np.asarray(gray_pil_img.convert("L"), dtype=np.uint8).copy()
    ink = pixels <= 140
    speck_area_limit = max(6, int(ink.size * 0.00001))
    component_count, labels, component_stats, _ = cv2.connectedComponentsWithStats(
        ink.astype(np.uint8), connectivity=8
    )
    if component_count <= 1:
        return Image.fromarray(pixels)

    component_areas = component_stats[1:, cv2.CC_STAT_AREA]
    speck_labels = np.flatnonzero(
        (component_areas > 0) & (component_areas <= speck_area_limit)
    ) + 1
    if speck_labels.size:
        pixels[np.isin(labels, speck_labels)] = 255
    return Image.fromarray(pixels)

def line_art_quality_issues(gray_pil_img):
    pixels = np.asarray(gray_pil_img.convert("L"), dtype=np.uint8)
    height, width = pixels.shape
    issues = []

    if min(height, width) < 512:
        issues.append("image resolution is too low")

    ink = pixels <= 140
    ink_ratio = float(np.mean(ink))
    if ink_ratio < 0.002:
        issues.append("image is nearly blank")
    elif ink_ratio > 0.35:
        issues.append("image contains too much dark fill or clutter")

    component_count, _, component_stats, _ = cv2.connectedComponentsWithStats(
        ink.astype(np.uint8), connectivity=8
    )
    if component_count > 1:
        component_areas = component_stats[1:, cv2.CC_STAT_AREA]
        largest_component_ratio = float(
            np.max(component_areas) / ink.size
        )
        if largest_component_ratio > 0.08:
            issues.append("image contains a large solid black region")

        speck_area_limit = max(6, int(ink.size * 0.00001))
        speck_count = int(np.count_nonzero(component_areas <= speck_area_limit))
        speck_count_limit = max(18, int(ink.size * 18 / (768 * 768)))
        if speck_count > speck_count_limit:
            issues.append("image contains excessive isolated specks")

    margin = max(1, int(min(height, width) * 0.01))
    edge_ink = np.concatenate((
        ink[:margin, :].ravel(),
        ink[-margin:, :].ravel(),
        ink[:, :margin].ravel(),
        ink[:, -margin:].ravel(),
    ))
    if float(np.mean(edge_ink)) > 0.04:
        issues.append("artwork touches the page edge")

    rows, columns = np.nonzero(ink)
    if rows.size:
        if (
            columns.min() < width * 0.005
            or columns.max() > width * 0.995
            or rows.min() < height * 0.005
            or rows.max() > height * 0.995
        ):
            issues.append("character or prop is cropped")

        center_x = (columns.min() + columns.max()) / (2 * width)
        center_y = (rows.min() + rows.max()) / (2 * height)
        if abs(center_x - 0.5) > 0.30 or abs(center_y - 0.5) > 0.32:
            issues.append("artwork is not centered")

    return issues

def convert_solid_regions_to_outlines(gray_pil_img):
    """Convert dark regions to contours and discard oversized circular backdrops."""
    pixels = np.asarray(gray_pil_img.convert("L"), dtype=np.uint8)
    ink = pixels <= 140
    if float(np.mean(ink)) < 0.08:
        return gray_pil_img.convert("L")

    contours, hierarchy = cv2.findContours(
        ink.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    result = np.full(pixels.shape, 255, dtype=np.uint8)
    if hierarchy is None:
        return Image.fromarray(result)

    hierarchy = hierarchy[0]
    minimum_contour_area = max(8, ink.size * 0.00001)
    oversized_circles = set()
    for index, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < minimum_contour_area:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter else 0
        is_oversized_circle = (
            width * height > ink.size * 0.15
            and 0.8 <= width / height <= 1.25
            and circularity > 0.72
        )
        if is_oversized_circle and hierarchy[index][3] == -1:
            child = hierarchy[index][2]
            while child != -1:
                if cv2.contourArea(contours[child]) > ink.size * 0.08:
                    oversized_circles.add(index)
                    break
                child = hierarchy[child][0]

    for index, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < minimum_contour_area:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter else 0
        ancestor = hierarchy[index][3]
        inside_oversized_circle = False
        while ancestor != -1:
            if ancestor in oversized_circles:
                inside_oversized_circle = True
                break
            ancestor = hierarchy[ancestor][3]

        is_backdrop_circle = index in oversized_circles or (
            inside_oversized_circle
            and 0.8 <= width / height <= 1.25
            and circularity > 0.72
        )
        if is_backdrop_circle:
            continue

        cv2.drawContours(result, contours, index, 0, thickness=3)

    return Image.fromarray(result)

def fetch_cloudflare_image(prompt, model, max_retries=6, negative_prompt=None, model_options=None):
    global CLOUDFLARE_QUOTA_EXHAUSTED

    if not ACCOUNT_ID or not API_TOKEN:
        raise ValueError("Missing CLOUDFLARE_ACCOUNT_ID or CLOUDFLARE_API_TOKEN environment variables.")
    if CLOUDFLARE_QUOTA_EXHAUSTED:
        raise CloudflareQuotaExceeded(
            "Cloudflare's daily AI allocation was already exhausted; skipping further requests."
        )

    url = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/{model}"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    payload = {"prompt": prompt}
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    if model_options:
        payload.update(model_options)

    for attempt in range(max_retries):
        try:
            response = session.post(url, headers=headers, json=payload, timeout=90)
            quota_error = cloudflare_quota_error(response)
            if quota_error:
                CLOUDFLARE_QUOTA_EXHAUSTED = True
                raise CloudflareQuotaExceeded(
                    f"Cloudflare daily AI allocation is exhausted: {quota_error}"
                )

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_time = int(retry_after) if retry_after and retry_after.isdigit() else (12 * (attempt + 1))
                if attempt == max_retries - 1:
                    detail = response.text[:500]
                    raise RuntimeError(
                        f"Cloudflare rate limit persisted after {max_retries} attempts "
                        f"for model {model}. Last response: {detail}"
                    )
                time.sleep(wait_time)
                continue

            if 400 <= response.status_code < 500:
                prompt_rejection = cloudflare_prompt_rejection(response)
                if prompt_rejection:
                    raise CloudflarePromptRejected(
                        f"Cloudflare rejected the image prompt: {prompt_rejection}"
                    )
                raise RuntimeError(
                    f"Cloudflare rejected image request for model {model} "
                    f"(HTTP {response.status_code}): {response.text[:500]}"
                )

            response.raise_for_status()
            try:
                json_data = response.json()
                if "result" in json_data and "image" in json_data["result"]:
                    return base64.b64decode(json_data["result"]["image"])
            except ValueError:
                pass
            return response.content

        except CloudflareQuotaExceeded:
            raise
        except RuntimeError:
            raise
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"Cloudflare image request for model {model} failed after "
                    f"{max_retries} attempts: {e}"
                ) from e
            time.sleep(6 * (attempt + 1))

    raise RuntimeError(f"Failed to fetch image after {max_retries} attempts.")

def create_blank_page():
    return Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")

def generate_procedural_maze_page(subject):
    img = create_blank_page()
    draw = ImageDraw.Draw(img)
    margin_x, margin_y = 140, 140
    draw.rounded_rectangle([margin_x, margin_y, PAGE_WIDTH - margin_x, PAGE_HEIGHT - margin_y], radius=40, outline="black", width=12)

    title_font = get_font(95)
    sub_font = get_font(52)
    draw.text((PAGE_WIDTH // 2, 320), "PUZZLE ADVENTURE: THE MAZE!", fill="black", font=title_font, anchor="mm")
    draw.text((PAGE_WIDTH // 2, 430), f"Help the baby {subject} find the bubble party!", fill="#444444", font=sub_font, anchor="mm")

    cols, rows = 8, 8
    cell_w, cell_h = 210, 210
    start_x = (PAGE_WIDTH - (cols * cell_w)) // 2
    start_y = 680

    draw.line([(start_x + cell_w, start_y), (start_x + cols * cell_w, start_y)], fill="black", width=14)
    draw.line([(start_x, start_y + rows * cell_h), (start_x + (cols - 1) * cell_w, start_y + rows * cell_h)], fill="black", width=14)
    draw.line([(start_x, start_y), (start_x, start_y + rows * cell_h)], fill="black", width=14)
    draw.line([(start_x + cols * cell_w, start_y), (start_x + cols * cell_w, start_y + rows * cell_h)], fill="black", width=14)

    random.seed(42)
    for r in range(rows):
        for c in range(cols):
            x = start_x + c * cell_w
            y = start_y + r * cell_h
            if random.random() > 0.45 and c < cols - 1 and r > 0:
                draw.line([(x + cell_w, y), (x + cell_w, y + cell_h)], fill="black", width=12)
            if random.random() > 0.50 and r < rows - 1 and c > 0:
                draw.line([(x, y + cell_h), (x + cell_w, y + cell_h)], fill="black", width=12)

    draw.text((start_x + cell_w // 2, start_y - 70), "START ★", fill="black", font=get_font(60), anchor="mm")
    draw.text((start_x + (cols - 1) * cell_w + cell_w // 2, start_y + rows * cell_h + 70), "FINISH 🎈", fill="black", font=get_font(60), anchor="mm")
    draw.text((PAGE_WIDTH // 2, PAGE_HEIGHT - 220), "Trace the path with your crayon, then color the decorations!", fill="#555555", font=get_font(48), anchor="mm")
    draw.text((PAGE_WIDTH // 2, PAGE_HEIGHT - 65), "- Activity Page 1 -", fill="#777777", font=get_font(44), anchor="mm")
    return img

def generate_procedural_dot_to_dot_page(subject):
    img = create_blank_page()
    draw = ImageDraw.Draw(img)
    margin_x, margin_y = 140, 140
    draw.rounded_rectangle([margin_x, margin_y, PAGE_WIDTH - margin_x, PAGE_HEIGHT - margin_y], radius=40, outline="black", width=12)

    title_font = get_font(95)
    sub_font = get_font(52)
    draw.text((PAGE_WIDTH // 2, 320), "CONNECT THE DOTS: 1 TO 20!", fill="black", font=title_font, anchor="mm")
    draw.text((PAGE_WIDTH // 2, 430), "Connect all the numbered dots in order to reveal the secret gift!", fill="#444444", font=sub_font, anchor="mm")

    center_x, center_y = PAGE_WIDTH // 2, 1500
    points = []
    num_points = 20

    for i in range(num_points):
        angle = (2 * math.pi * i) / num_points - (math.pi / 2)
        radius = 700 if i % 2 == 0 else 380
        px = int(center_x + radius * math.cos(angle))
        py = int(center_y + radius * math.sin(angle))
        points.append((px, py))

    num_font = get_font(52)
    for i, (px, py) in enumerate(points, start=1):
        draw.ellipse([px - 16, py - 16, px + 16, py + 16], fill="black")
        text_offset_x = 40 if px >= center_x else -40
        text_offset_y = 40 if py >= center_y else -40
        draw.text((px + text_offset_x, py + text_offset_y), str(i), fill="black", font=num_font, anchor="mm")

    draw.ellipse([center_x - 80, center_y - 80, center_x + 80, center_y + 80], outline="black", width=10)
    draw.text((center_x, center_y), "★", fill="black", font=get_font(80), anchor="mm")
    draw.text((PAGE_WIDTH // 2, PAGE_HEIGHT - 220), "Connect from 1 to 20 without lifting your crayon!", fill="#555555", font=get_font(48), anchor="mm")
    draw.text((PAGE_WIDTH // 2, PAGE_HEIGHT - 65), "- Activity Page 2 -", fill="#777777", font=get_font(44), anchor="mm")
    return img

def generate_procedural_spot_differences_page(subject):
    img = create_blank_page()
    draw = ImageDraw.Draw(img)
    margin_x, margin_y = 140, 140
    draw.rounded_rectangle([margin_x, margin_y, PAGE_WIDTH - margin_x, PAGE_HEIGHT - margin_y], radius=40, outline="black", width=12)

    title_font = get_font(95)
    sub_font = get_font(52)
    draw.text((PAGE_WIDTH // 2, 320), "SPOT THE 4 DIFFERENCES!", fill="black", font=title_font, anchor="mm")
    draw.text((PAGE_WIDTH // 2, 430), "Look closely at Picture A and Picture B. Can you find all 4 changes?", fill="#444444", font=sub_font, anchor="mm")

    box_w, box_h = 1000, 1300
    y0 = 650
    x_left = 220
    x_right = PAGE_WIDTH - 220 - box_w

    for x_pos, label in [(x_left, "PICTURE A"), (x_right, "PICTURE B")]:
        draw.rounded_rectangle([x_pos, y0, x_pos + box_w, y0 + box_h], radius=30, outline="black", width=10)
        draw.text((x_pos + box_w // 2, y0 - 45), label, fill="black", font=get_font(55), anchor="mm")

        cx, cy = x_pos + box_w // 2, y0 + box_h // 2
        draw.ellipse([cx - 200, cy - 180, cx + 200, cy + 220], outline="black", width=12)
        draw.ellipse([cx - 90, cy - 30, cx - 40, cy + 20], fill="black")
        draw.ellipse([cx + 40, cy - 30, cx + 90, cy + 20], fill="black")
        draw.arc([cx - 70, cy + 30, cx + 70, cy + 110], 0, 180, fill="black", width=10)
        draw.line([(x_pos + 40, y0 + box_h - 120), (x_pos + box_w - 40, y0 + box_h - 120)], fill="black", width=10)

    draw.ellipse([x_right + 180, y0 + 180, x_right + 260, y0 + 260], outline="black", width=10)
    draw.ellipse([x_right + 220, y0 + 290, x_right + 280, y0 + 350], outline="black", width=10)
    draw.arc([x_right + box_w // 2 - 70, y0 + box_h // 2 + 10, x_right + box_w // 2 + 70, y0 + box_h // 2 + 110], 0, 180, fill="black", width=14)
    draw.text((x_right + box_w - 180, y0 + 200), "★", fill="black", font=get_font(80), anchor="mm")

    draw.text((PAGE_WIDTH // 2, PAGE_HEIGHT - 220), "Circle the 4 differences with your favorite bright marker!", fill="#555555", font=get_font(48), anchor="mm")
    draw.text((PAGE_WIDTH // 2, PAGE_HEIGHT - 65), "- Activity Page 3 -", fill="#777777", font=get_font(44), anchor="mm")
    return img

def generate_cover_page(subject, category):
    print(f"Generating full-color cover for '{subject}'...")
    prompt = (
        f"A beautiful, vibrant children's book cover illustration of an adorable {subject}. "
        "Style: clean, simple cartoon illustration with bright cheerful daylight colors and thick smooth outlines. "
        "Large readable shapes, adorable smiling animal character centered in its natural environment. "
        "High key lighting, completely clean, strictly NO text, NO letters, NO words, NO logos."
    )
    try:
        image_bytes = fetch_book_image(prompt, COVER_MODEL)
    except ImageProviderUnavailable:
        raise
    except RuntimeError as error:
        print(f"[Warning] Cover illustration unavailable: {error}")
        print("[Warning] Creating a local cover and continuing book generation.")
        return generate_fallback_cover_page(subject, category)
    img = Image.open(BytesIO(image_bytes)).convert("RGB")

    img_ratio = img.width / img.height
    page_ratio = PAGE_WIDTH / PAGE_HEIGHT

    if img_ratio > page_ratio:
        new_width = PAGE_WIDTH
        new_height = int(PAGE_WIDTH / img_ratio)
    else:
        new_height = PAGE_HEIGHT
        new_width = int(PAGE_HEIGHT * img_ratio)

    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    final_page = create_blank_page()
    x_offset = (PAGE_WIDTH - new_width) // 2
    y_offset = (PAGE_HEIGHT - new_height) // 2
    final_page.paste(img, (x_offset, y_offset))

    draw = ImageDraw.Draw(final_page)
    title_font = get_font(130)
    sub_font = get_font(60)

    draw.rectangle([100, 120, PAGE_WIDTH - 100, 360], fill=(255, 255, 255), outline="black", width=10)
    draw.text((PAGE_WIDTH // 2, 200), "TODDLER COLORING BOOK", fill="black", font=title_font, anchor="mm")
    draw.text((PAGE_WIDTH // 2, 300), f"Simple Adventures in {category}", fill="#333333", font=sub_font, anchor="mm")

    return final_page

def generate_fallback_cover_page(subject, category):
    """Create a clean local cover when the remote cover illustration is unavailable."""
    img = create_blank_page()
    draw = ImageDraw.Draw(img)
    navy = (35, 67, 112)

    draw.rounded_rectangle(
        [100, 100, PAGE_WIDTH - 100, PAGE_HEIGHT - 100],
        radius=60,
        outline=navy,
        width=16,
    )
    draw.text((PAGE_WIDTH // 2, 500), "MY COLORING BOOK", fill=navy, font=get_font(120), anchor="mm")
    draw_wrapped_centered_text(
        draw,
        category,
        PAGE_WIDTH // 2,
        680,
        get_font(65),
        fill="#444444",
        max_width=1900,
    )

    center_y = 1550
    for center_x, radius, color in [
        (PAGE_WIDTH // 2 - 390, 220, (239, 190, 187)),
        (PAGE_WIDTH // 2, 300, (255, 224, 142)),
        (PAGE_WIDTH // 2 + 390, 220, (170, 212, 190)),
    ]:
        draw.ellipse(
            [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
            fill=color,
            outline=navy,
            width=12,
        )
        draw.ellipse(
            [
                center_x - radius // 2,
                center_y - radius // 2,
                center_x + radius // 2,
                center_y + radius // 2,
            ],
            outline="white",
            width=10,
        )

    draw.rounded_rectangle(
        [250, 2200, PAGE_WIDTH - 250, 2460],
        radius=45,
        fill="white",
        outline=navy,
        width=8,
    )
    draw_wrapped_centered_text(
        draw,
        subject,
        PAGE_WIDTH // 2,
        2330,
        get_font(82),
        fill=navy,
        max_width=1950,
    )
    draw.text(
        (PAGE_WIDTH // 2, 2700),
        "A book full of pictures to color!",
        fill="#555555",
        font=get_font(58),
        anchor="mm",
    )
    return img

def generate_ownership_page():
    img = create_blank_page()
    draw = ImageDraw.Draw(img)
    title_font = get_font(120)
    sub_font = get_font(70)

    draw.rectangle([120, 120, PAGE_WIDTH - 120, PAGE_HEIGHT - 120], outline="black", width=12)
    draw.rectangle([150, 150, PAGE_WIDTH - 150, PAGE_HEIGHT - 150], outline="black", width=4)
    draw.text((PAGE_WIDTH // 2, 800), "This Coloring Book", fill="black", font=sub_font, anchor="mm")
    draw.text((PAGE_WIDTH // 2, 980), "BELONGS TO:", fill="black", font=title_font, anchor="mm")

    box_x0, box_y0 = PAGE_WIDTH // 2 - 800, 1300
    box_x1, box_y1 = PAGE_WIDTH // 2 + 800, 1600
    draw.rectangle([box_x0, box_y0, box_x1, box_y1], outline="black", width=10)
    draw.line([(box_x0 + 100, 1530), (box_x1 - 100, 1530)], fill="black", width=6)

    for x in [PAGE_WIDTH // 4, PAGE_WIDTH // 2, (PAGE_WIDTH // 4) * 3]:
        draw.ellipse([x - 70, 2400, x + 70, 2540], outline="black", width=8)

    draw.text((PAGE_WIDTH // 2, 2800), "Color your own adventure!", fill="#444444", font=sub_font, anchor="mm")
    return img

def generate_terms_page():
    img = create_blank_page()
    draw = ImageDraw.Draw(img)
    title_font = get_font(100)
    body_font = get_font(55)

    draw.text((PAGE_WIDTH // 2, 1000), "Terms of Use", fill="black", font=title_font, anchor="mm")
    terms_text = (
        "Thank you for downloading this simple toddler coloring book!\n\n"
        "This digital file is intended strictly for personal,\n"
        "classroom, or non-commercial home use.\n\n"
        "Please do not resell, redistribute, or host these files\n"
        "publicly online. All illustrations are proprietary."
    )
    draw.multiline_text((PAGE_WIDTH // 2, 1500), terms_text, fill="#333333", font=body_font, anchor="mm", align="center", spacing=25)
    return img

def generate_color_test_page():
    img = create_blank_page()
    draw = ImageDraw.Draw(img)
    title_font = get_font(110)
    sub_font = get_font(60)

    draw.text((PAGE_WIDTH // 2, 450), "Color & Swatch Test Page", fill="black", font=title_font, anchor="mm")
    draw.text((PAGE_WIDTH // 2, 600), "Test your crayons, markers, and colored pencils below!", fill="#444444", font=sub_font, anchor="mm")

    cols, rows = 3, 4
    box_w, box_h = 550, 400
    start_x = (PAGE_WIDTH - (cols * box_w + (cols - 1) * 150)) // 2
    start_y = 850

    for r in range(rows):
        for c in range(cols):
            x0 = start_x + c * (box_w + 150)
            y0 = start_y + r * (box_h + 120)
            x1, y1 = x0 + box_w, y0 + box_h
            draw.rectangle([x0, y0, x1, y1], outline="black", width=8)
            draw.text((x0 + 20, y1 + 35), f"Color #{r * cols + c + 1}", fill="black", font=get_font(40))

    return img

def generate_vector_interior_page(character_dna, scene_data, page_num):
    action = scene_data["action"]
    setting = scene_data.get("setting", "")
    composition = scene_data.get("composition", "")
    micro_prompt = scene_data["micro_prompt"]
    print(f"Generating page {page_num}/30: {action[:42]}...")

    prompt = (
        "Create one uncluttered toddler coloring-book story illustration in clean black ink outline art on pure white paper. "
        f"Character design: {character_dna}. "
        f"Story action: {action}. "
        f"Setting and story details: {setting or 'a simple open setting that fits the action'}. "
        f"Composition: {composition or 'a clear, balanced layout suited to the scene'}. "
        "Represent the stated action literally and make it unmistakable at a glance. Use one single scene, not a collage, circular frame, halo, badge, or corner vignette. "
        "Make the main animal the large, recognizable focal character, fully visible and doing the stated action. "
        "Include only one or two large, simple objects or scenery details named in the scene; keep them separate and easy to color. "
        "Vary the pose, viewpoint, and arrangement to match this page's specific story rather than using a generic animal portrait. "
        "Draw smooth, clean black outlines with simple facial and body-separation lines; avoid speckles, stray marks, and fragmented strokes. "
        "Keep every enclosed area white and open for coloring. Fit the complete animal and every story detail within the central 80% of the image, "
        "leaving a wide, uninterrupted white margin on all four sides. "
        "Do not add unrelated objects, busy background clutter, borders, or rendered words. "
        "No color, gray fill, shading, shadows, gradients, textures, or filled black areas. This is not a colored illustration."
    )

    img = None
    for attempt in range(1, 4):
        image_bytes = fetch_book_image(
            prompt,
            INTERIOR_MODEL,
            model_options={
                "guidance": 7.5,
                "num_steps": 20,
                "width": 1024,
                "height": 1024,
            },
        )
        try:
            generated = Image.open(BytesIO(image_bytes)).convert("RGBA")
            white_background = Image.new("RGBA", generated.size, "white")
            candidate = Image.alpha_composite(white_background, generated).convert("L")
            candidate = convert_solid_regions_to_outlines(candidate)
            candidate = remove_isolated_specks(candidate)
            issues = line_art_quality_issues(candidate)
        except (OSError, ValueError) as error:
            candidate = None
            issues = [f"image could not be decoded ({error})"]

        if not issues:
            img = candidate
            break

        print(f"[Warning] Page {page_num} rejected on attempt {attempt}: {', '.join(issues)}")
        if attempt < 3:
            prompt += (
                " Regenerate as simple unfilled coloring-book line art: use thin black outlines only, "
                "leave the animal's body and every enclosed area white, and include no solid black shapes."
            )
            if "image contains excessive isolated specks" in issues:
                prompt += (
                    " Remove every stray dot and speck. Keep the outlines smooth and continuous, "
                    "with no scattered marks or fragments anywhere in the white background."
                )
            if "character or prop is cropped" in issues or "artwork touches the page edge" in issues:
                if attempt == 1:
                    prompt += (
                        " Pull the viewpoint farther back. Show the complete animal and every prop, plant, and other character "
                        "entirely inside the central 70% of the image, with at least 15% plain white space on every side. "
                        "Nothing may touch or cross the image edges."
                    )
                else:
                    prompt += (
                        " Simplify the scene and make the complete animal and its single most important story detail small enough "
                        "to fit inside the central half of the image. Remove all other scenery and secondary objects. "
                        "Leave the outer quarter of the image completely blank white; nothing may touch or cross an edge."
                    )
            if "artwork is not centered" in issues:
                prompt += " Position the complete animal in the center of the image with balanced white margins."

    if img is None and candidate is not None and set(issues).issubset({
        "artwork touches the page edge",
        "character or prop is cropped",
        "artwork is not centered",
    }):
        framed_candidate = add_print_safe_margin(candidate)
        if not line_art_quality_issues(framed_candidate):
            print(f"[Notice] Page {page_num} accepted after adding a print-safe white margin.")
            img = framed_candidate

    if img is None:
        raise RuntimeError(
            f"Could not generate line art that passes quality checks for page {page_num} after 3 attempts."
        )

    # Apply OpenCV Thresholding & Morphological Dilation (Kernel 5 = Extra Thick Toddler Lines)
    img = clean_and_thicken_line_art(img)

    border_margin_x = 140
    border_margin_y = 140
    inner_w = PAGE_WIDTH - (2 * border_margin_x) - 40
    inner_h = PAGE_HEIGHT - (2 * border_margin_y) - 260

    img_ratio = img.width / img.height
    target_ratio = inner_w / inner_h

    if img_ratio > target_ratio:
        new_width = inner_w
        new_height = int(inner_w / img_ratio)
    else:
        new_height = inner_h
        new_width = int(inner_h * img_ratio)

    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    _, resized_binary = cv2.threshold(np.array(img), 140, 255, cv2.THRESH_BINARY)
    img = Image.fromarray(resized_binary)

    final_page = create_blank_page()
    draw = ImageDraw.Draw(final_page)
    draw.rounded_rectangle([border_margin_x, border_margin_y, PAGE_WIDTH - border_margin_x, PAGE_HEIGHT - border_margin_y], radius=40, outline="black", width=10)

    x_offset = border_margin_x + 20 + ((inner_w - new_width) // 2)
    y_offset = border_margin_y + 40 + ((inner_h - new_height) // 2)
    final_page.paste(img.convert("RGB"), (x_offset, y_offset))

    micro_prompt = " ".join(str(micro_prompt).split())
    prompt_font_size = 50
    prompt_font = get_font(prompt_font_size)
    while draw.textbbox((0, 0), micro_prompt, font=prompt_font)[2] > PAGE_WIDTH - 2 * border_margin_x - 80 and prompt_font_size > 30:
        prompt_font_size -= 2
        prompt_font = get_font(prompt_font_size)
    draw.text((PAGE_WIDTH // 2, PAGE_HEIGHT - border_margin_y - 100), micro_prompt, fill="#333333", font=prompt_font, anchor="mm")

    page_font = get_font(44)
    draw.text((PAGE_WIDTH // 2, PAGE_HEIGHT - 65), f"- {page_num} -", fill="#666666", font=page_font, anchor="mm")

    return final_page

def generate_certificate_page(subject):
    img = create_blank_page()
    draw = ImageDraw.Draw(img)
    draw.rectangle([140, 140, PAGE_WIDTH - 140, PAGE_HEIGHT - 140], outline="black", width=16)
    draw.rectangle([180, 180, PAGE_WIDTH - 180, PAGE_HEIGHT - 180], outline="black", width=4)

    title_font = get_font(135)
    sub_font = get_font(70)
    body_font = get_font(55)

    draw.text((PAGE_WIDTH // 2, 600), "CERTIFICATE", fill="black", font=title_font, anchor="mm")
    draw.text((PAGE_WIDTH // 2, 770), "OF COMPLETION", fill="black", font=sub_font, anchor="mm")
    draw.text((PAGE_WIDTH // 2, 1120), "This certificate is proudly awarded to:", fill="#444444", font=body_font, anchor="mm")
    draw.line([(PAGE_WIDTH // 2 - 700, 1400), (PAGE_WIDTH // 2 + 700, 1400)], fill="black", width=6)
    draw.text((PAGE_WIDTH // 2, 1480), "Super Colorist", fill="#777777", font=get_font(45), anchor="mm")

    subject_str = subject if subject.lower().startswith("the ") or subject.lower().startswith("barnaby") or subject.lower().startswith("mochi") or subject.lower().startswith("kip") else f"the {subject}"
    desc = f"For creatively completing all simple coloring adventures with {subject_str}!"
    draw_wrapped_centered_text(draw, desc, PAGE_WIDTH // 2, 1780, body_font, fill="black", max_width=1750)

    draw.line([(350, 2450), (950, 2450)], fill="black", width=5)
    draw.text((650, 2510), "Date", fill="black", font=get_font(48), anchor="mm")
    draw.ellipse([PAGE_WIDTH // 2 - 120, 2270, PAGE_WIDTH // 2 + 120, 2510], outline="black", width=8)
    draw.text((PAGE_WIDTH // 2, 2390), "★", fill="black", font=get_font(95), anchor="mm")
    draw.line([(PAGE_WIDTH - 950, 2450), (PAGE_WIDTH - 350, 2450)], fill="black", width=5)
    draw.text((PAGE_WIDTH - 650, 2510), "Artist Signature", fill="black", font=get_font(48), anchor="mm")
    return img

def generate_retail_back_cover_with_thumbnails(subject, sample_pages):
    img = create_blank_page()
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 100, PAGE_WIDTH - 100, PAGE_HEIGHT - 100], outline="black", width=12)

    title_font = get_font(105)
    sub_font = get_font(55)
    draw.text((PAGE_WIDTH // 2, 380), "30 Simple Coloring Pages Inside!", fill="black", font=title_font, anchor="mm")

    subject_str = subject if subject.lower().startswith("the ") or subject.lower().startswith("barnaby") or subject.lower().startswith("mochi") or subject.lower().startswith("kip") else f"the {subject}"
    draw.text((PAGE_WIDTH // 2, 500), f"Peek inside to see what you will color with {subject_str}!", fill="#555555", font=sub_font, anchor="mm")

    thumb_w, thumb_h = 750, 970
    cols, rows = 2, 2
    spacing_x, spacing_y = 180, 120
    start_x = (PAGE_WIDTH - (cols * thumb_w + spacing_x)) // 2
    start_y = 660

    for idx, sample_page in enumerate(sample_pages[:4]):
        c = idx % 2
        r = idx // 2
        x0 = start_x + c * (thumb_w + spacing_x)
        y0 = start_y + r * (thumb_h + spacing_y)
        draw.rounded_rectangle([x0 + 15, y0 + 15, x0 + thumb_w + 15, y0 + thumb_h + 15], radius=25, fill="#dddddd")
        thumb = sample_page.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        img.paste(thumb, (x0, y0))
        draw.rounded_rectangle([x0, y0, x0 + thumb_w, y0 + thumb_h], radius=25, outline="black", width=6)

    draw.text((PAGE_WIDTH // 2, 2900), "Perfect for crayons, colored pencils, and soft markers.", fill="#333333", font=get_font(52), anchor="mm")
    draw.text((PAGE_WIDTH // 2, 3010), "Proudly printed for young artists. Share your artwork with family!", fill="#777777", font=get_font(44), anchor="mm")
    return img

def build_book():
    plan_file = "daily_book_plan.json"
    if not os.path.exists(plan_file):
        raise FileNotFoundError(f"Missing '{plan_file}'. Run 'python research_trend.py' first.")

    with open(plan_file, "r", encoding="utf-8") as f:
        plan = json.load(f)

    subject = plan["subject"]
    category = plan.get("category", "Toddler Animal Adventures")
    character_dna = plan["character_dna"]
    scenes = plan["scenes"]
    if len(scenes) != 30:
        raise ValueError(f"Expected exactly 30 interior scenes, found {len(scenes)}.")

    print("=====================================================")
    print(f"Starting High-End Book Production: '{subject}'")
    print(f"Total Unique Habitat Scenes: {len(scenes)}")
    print("=====================================================")

    pages = [
        generate_cover_page(subject, category),
        generate_ownership_page(),
        generate_terms_page(),
        generate_color_test_page()
    ]

    time.sleep(3)
    sample_thumbnails = []

    for i, scene in enumerate(scenes[:30], start=1):
        if i == 8:
            pages.append(generate_procedural_maze_page(subject))
        elif i == 16:
            pages.append(generate_procedural_dot_to_dot_page(subject))
        elif i == 24:
            pages.append(generate_procedural_spot_differences_page(subject))

        page_img = generate_vector_interior_page(character_dna, scene, i)
        pages.append(page_img)

        if i in (3, 11, 19, 27):
            sample_thumbnails.append(page_img)

        time.sleep(2)

    pages.append(generate_certificate_page(subject))
    pages.append(generate_retail_back_cover_with_thumbnails(subject, sample_thumbnails))
    if len(pages) != 39:
        raise RuntimeError(f"Expected exactly 39 pages, assembled {len(pages)}.")

    safe_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", subject.strip().lower())
    safe_slug = re.sub(r"_+", "_", safe_slug)
    filename = f"Digital_Coloring_Book_{safe_slug}.pdf"

    if pages:
        pages[0].save(
            filename, "PDF", resolution=300, save_all=True, append_images=pages[1:]
        )

    print("=====================================================")
    print(f"Successfully generated high-end book: {filename}")
    print(f"Total compiled pages: {len(pages)}")
    print("=====================================================")

def main():
    try:
        build_book()
    except (CloudflareQuotaExceeded, ImageProviderUnavailable) as error:
        print(f"[Error] Book generation stopped: {error}")
        print(
            "[Error] No PDF was created. Restore Cloudflare quota or make the "
            "Pollinations image service available, then rerun."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
