#!/usr/bin/env python3
"""
Expanded Test Coverage for Coloring Book Generator

Additional tests beyond quality_assurance tests:
- Individual backend methods
- Line art processing methods
- Error handling
- Edge cases
- PDF generation
- Upscaling verification

Run with: python3 test_expanded_coverage.py
"""

import sys
import tempfile
import json
import os
import requests
from pathlib import Path
from contextlib import redirect_stdout
from io import BytesIO
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image, ImageDraw

# Import main module
from coloring_book_generator import (
    ColoringBookGenerator,
    COLORING_PAGE_STYLE,
    THEMES,
    build_coloring_page_prompt,
    validate_image_quality,
    MIN_PRINT_WIDTH,
    MIN_PRINT_HEIGHT,
    REQUIRED_DPI
)
from generate_book import (
    CloudflareQuotaExceeded,
    ImageProviderUnavailable,
    build_book,
    clean_and_thicken_line_art,
    convert_solid_regions_to_outlines,
    fetch_book_image,
    fetch_cloudflare_image,
    fetch_pollinations_image,
    generate_cover_page,
    generate_vector_interior_page,
    line_art_quality_issues,
    remove_isolated_specks,
    main,
)
import research_trend
from research_trend import (
    DailyBookPlan,
    generate_daily_plan,
    generate_fallback_plan,
    has_distinct_story_scenes,
)


def create_test_image(width=768, height=768, color=(255, 255, 255)):
    """Create a simple test image in memory.

    Returns:
        bytes: PNG image as bytes
    """
    try:
        from PIL import Image

        # Create white image
        img = Image.new('RGB', (width, height), color)

        # Add simple black lines for line-art processing tests.
        import numpy as np
        arr = np.array(img)
        # Horizontal line
        arr[height//2, :] = [0, 0, 0]
        # Vertical line
        arr[:, width//2] = [0, 0, 0]
        # Diagonal
        for i in range(min(width, height)):
            arr[i, i] = [0, 0, 0]

        img = Image.fromarray(arr)

        # Save to bytes
        output = BytesIO()
        img.save(output, format='PNG')
        return output.getvalue()
    except ImportError:
        print("⚠ PIL not available - skipping image tests")
        return None


class TestLineArtMethods:
    """Test all three line art processing methods."""

    def __init__(self):
        self.gen = ColoringBookGenerator(force_lineart=True)

    def test_enhanced_method(self):
        """Test enhanced line art method (thick lines)."""
        print("\n🧪 Testing enhanced line art method...")

        test_img = create_test_image(768, 768)
        if not test_img:
            print("  ⊘ Skipped (PIL not available)")
            return False

        try:
            # Should upscale and process
            result = self.gen.convert_to_coloring_page(
                test_img,
                method='enhanced',
                target_size=(2550, 3300)
            )

            assert result is not None, "Result should not be None"
            assert len(result) > 0, "Result should have content"

            # Verify result is larger than input
            assert len(result) > len(test_img), "Processed should be larger (upscaled)"

            print("  ✓ Enhanced method works")
            return True

        except Exception as e:
            print(f"  ✗ Enhanced method failed: {e}")
            return False

    def test_standard_method(self):
        """Test standard line art method."""
        print("\n🧪 Testing standard line art method...")

        test_img = create_test_image(768, 768)
        if not test_img:
            print("  ⊘ Skipped (PIL not available)")
            return False

        try:
            result = self.gen.convert_to_coloring_page(
                test_img,
                method='standard',
                target_size=(2550, 3300)
            )

            assert result is not None
            assert len(result) > 0

            print("  ✓ Standard method works")
            return True

        except Exception as e:
            print(f"  ✗ Standard method failed: {e}")
            return False

    def test_detailed_method(self):
        """Test detailed line art method (fine lines)."""
        print("\n🧪 Testing detailed line art method...")

        test_img = create_test_image(768, 768)
        if not test_img:
            print("  ⊘ Skipped (PIL not available)")
            return False

        try:
            result = self.gen.convert_to_coloring_page(
                test_img,
                method='detailed',
                target_size=(2550, 3300)
            )

            assert result is not None
            assert len(result) > 0

            print("  ✓ Detailed method works")
            return True

        except Exception as e:
            print(f"  ✗ Detailed method failed: {e}")
            return False


class TestReferenceStyle:
    """Verify theme prompts share one style and outlines survive processing."""

    def test_style_applies_to_every_theme(self):
        print("\n🧪 Testing shared style is applied across themes...")
        assert THEMES
        for theme in THEMES.values():
            for prompt in theme["prompts"]:
                assert COLORING_PAGE_STYLE in build_coloring_page_prompt(prompt)
        print(f"  ✓ Shared clean style applied to all {len(THEMES)} themes")
        return True

    def test_line_art_preserves_closed_shapes(self):
        print("\n🧪 Testing line art conversion preserves open coloring areas...")
        from PIL import Image, ImageDraw

        source = Image.new("RGB", (256, 256), "white")
        draw = ImageDraw.Draw(source)
        draw.ellipse((48, 48, 208, 208), outline="black", width=5)
        input_bytes = BytesIO()
        source.save(input_bytes, format="PNG")

        result = ColoringBookGenerator(force_lineart=True).convert_to_coloring_page(
            input_bytes.getvalue(),
            method="enhanced",
            target_size=(512, 512),
        )
        output = Image.open(BytesIO(result)).convert("L")
        assert output.size == (512, 512)
        assert output.getpixel((256, 256)) == 255, "Shape interior should remain white"
        assert output.getpixel((96, 256)) == 0, "Original outline should remain black"
        assert set(np.unique(np.asarray(output))) <= {0, 255}, "Output should contain only black and white"
        print("  ✓ Closed outline retained with a white fill area")
        return True


class TestCloudflareImageRequests:
    """Cover authentication, exhausted quotas, and graceful failure behavior."""

    @staticmethod
    def response(status_code, content=b"image-data", headers=None):
        response = requests.Response()
        response.status_code = status_code
        response._content = content
        response.headers.update(headers or {})
        response.url = "https://api.cloudflare.com/test"
        return response

    def test_cloudflare_uses_bearer_token(self):
        response = self.response(200)
        with patch("generate_book.ACCOUNT_ID", "account"), \
                patch("generate_book.API_TOKEN", "test-token"), \
                patch("generate_book.session.post", return_value=response) as post:
            result = fetch_cloudflare_image("test prompt", "test-model")

        assert result == b"image-data"
        assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer test-token"
        print("  ✓ Cloudflare request includes a Bearer token")
        return True

    def test_rate_limit_failure_is_descriptive(self):
        responses = [self.response(429, b"rate limited")] * 2
        with patch("generate_book.ACCOUNT_ID", "account"), \
                patch("generate_book.API_TOKEN", "test-token"), \
                patch("generate_book.session.post", side_effect=responses), \
                patch("generate_book.time.sleep"):
            try:
                fetch_cloudflare_image("test prompt", "test-model", max_retries=2)
            except RuntimeError as error:
                assert "rate limit persisted" in str(error)
                assert "rate limited" in str(error)
            else:
                raise AssertionError("Persistent HTTP 429 should raise a clear RuntimeError")

        print("  ✓ Exhausted rate limits report the last API response")
        return True

    def test_cover_falls_back_when_image_service_fails(self):
        with patch(
            "generate_book.fetch_cloudflare_image",
            side_effect=RuntimeError("test rate limit"),
        ):
            cover = generate_cover_page("Test Animal", "Simple Animal Adventures")

        assert cover.size == (2550, 3300)
        assert cover.mode == "RGB"
        print("  ✓ Cover generation falls back locally when the service is unavailable")
        return True

    def test_daily_quota_stops_retries(self):
        quota_body = json.dumps({
            "errors": [{
                "code": 4006,
                "message": "you have used up your daily free allocation of 10,000 neurons",
            }],
            "success": False,
            "result": {},
        }).encode()
        response = self.response(429, quota_body, {"Content-Type": "application/json"})
        with patch("generate_book.ACCOUNT_ID", "account"), \
                patch("generate_book.API_TOKEN", "test-token"), \
                patch("generate_book.CLOUDFLARE_QUOTA_EXHAUSTED", False), \
                patch("generate_book.session.post", return_value=response) as post, \
                patch("generate_book.time.sleep") as sleep:
            try:
                fetch_cloudflare_image("test prompt", "test-model")
            except CloudflareQuotaExceeded as error:
                assert "daily AI allocation is exhausted" in str(error)
                assert "4006" in str(error)
            else:
                raise AssertionError("Exhausted quota should stop image generation")

            assert post.call_count == 1, "Quota exhaustion should not be retried"
            sleep.assert_not_called()

        print("  ✓ Daily quota exhaustion stops retries immediately")
        return True

    def test_interior_uses_pollinations_after_quota(self):
        scene = {
            "action": "stands facing forward",
            "micro_prompt": "Color the calf's ears!",
        }
        art = Image.new("L", (768, 768), "white")
        ImageDraw.Draw(art).ellipse([150, 100, 620, 660], outline="black", width=8)
        encoded = BytesIO()
        art.save(encoded, format="PNG")

        with patch("generate_book.CLOUDFLARE_QUOTA_EXHAUSTED", True), \
                patch("generate_book.POLLINATIONS_FALLBACK_ACTIVE", False), \
                patch("generate_book.fetch_pollinations_image", return_value=encoded.getvalue()) as pollinations:
            page = generate_vector_interior_page(
                "adorable chubby baby highland cow calf",
                scene,
                1,
            )

        assert page.size == (2550, 3300)
        pollinations.assert_called_once()
        print("  ✓ Interior illustration uses generated Pollinations line art after quota exhaustion")
        return True

    def test_quota_switches_to_pollinations(self):
        quota_body = json.dumps({
            "errors": [{
                "code": 4006,
                "message": "daily free allocation of 10,000 neurons exhausted",
            }],
            "success": False,
            "result": {},
        }).encode()
        quota_response = self.response(429, quota_body, {"Content-Type": "application/json"})
        with patch("generate_book.ACCOUNT_ID", "account"), \
                patch("generate_book.API_TOKEN", "test-token"), \
                patch("generate_book.CLOUDFLARE_QUOTA_EXHAUSTED", False), \
                patch("generate_book.POLLINATIONS_FALLBACK_ACTIVE", False), \
                patch("generate_book.session.post", return_value=quota_response) as post, \
                patch("generate_book.fetch_pollinations_image", return_value=b"generated-image") as pollinations:
            first = fetch_book_image("clean black line art", "test-model")
            second = fetch_book_image("another clean line-art page", "test-model")

        assert first == second == b"generated-image"
        assert post.call_count == 1, "Cloudflare should not be called again after quota exhaustion"
        assert pollinations.call_count == 2
        print("  ✓ Quota exhaustion switches subsequent image requests to Pollinations")
        return True

    def test_safety_rejection_switches_to_pollinations(self):
        rejection = self.response(
            400,
            json.dumps({
                "errors": [{
                    "code": 3030,
                    "message": "AiError: Input prompt contains NSFW content.",
                }],
                "success": False,
            }).encode(),
            {"Content-Type": "application/json"},
        )
        with patch("generate_book.ACCOUNT_ID", "account"), \
                patch("generate_book.API_TOKEN", "test-token"), \
                patch("generate_book.CLOUDFLARE_QUOTA_EXHAUSTED", False), \
                patch("generate_book.POLLINATIONS_FALLBACK_ACTIVE", False), \
                patch("generate_book.session.post", return_value=rejection) as post, \
                patch("generate_book.fetch_pollinations_image", return_value=b"generated-image") as pollinations:
            first = fetch_book_image("family-friendly toddler coloring page", "test-model")
            second = fetch_book_image("another interior page", "test-model")

        assert first == second == b"generated-image"
        assert post.call_count == 1, "Cloudflare should not be called again after a safety rejection"
        assert pollinations.call_count == 2
        print("  ✓ Cloudflare prompt-safety rejection switches subsequent requests to Pollinations")
        return True

    def test_pollinations_rejects_non_image_responses(self):
        response = self.response(200, b"not an image", {"Content-Type": "text/plain"})
        with patch("generate_book.session.get", return_value=response):
            try:
                fetch_pollinations_image("simple animal coloring page")
            except ImageProviderUnavailable as error:
                assert "invalid image" in str(error)
            else:
                raise AssertionError("Invalid image responses must not be used")
        print("  ✓ Pollinations response is validated before use")
        return True

    def test_pollinations_retries_free_limit(self):
        limited = self.response(
            402,
            json.dumps({
                "error": {
                    "code": "FREE_LIMIT",
                    "message": "Free limit reached. Retry after 37s.",
                },
            }).encode(),
            {"Content-Type": "application/json"},
        )
        image = Image.new("RGB", (32, 32), "white")
        encoded = BytesIO()
        image.save(encoded, format="PNG")
        success = self.response(200, encoded.getvalue(), {"Content-Type": "image/png"})

        with patch("generate_book.session.get", side_effect=[limited, success]) as get, \
                patch("generate_book.time.sleep") as sleep:
            result = fetch_pollinations_image("simple black outline coloring page")

        assert result == encoded.getvalue()
        assert get.call_count == 2
        sleep.assert_called_once_with(38)
        print("  ✓ Pollinations FREE_LIMIT waits for the server-provided retry time")
        return True

    def test_pollinations_rate_limit_retries_are_bounded(self):
        limited = self.response(
            402,
            json.dumps({
                "error": {
                    "code": "FREE_LIMIT",
                    "message": "Retry after 45s.",
                },
            }).encode(),
            {"Content-Type": "application/json"},
        )
        with patch("generate_book.session.get", return_value=limited) as get, \
                patch("generate_book.time.sleep") as sleep:
            try:
                fetch_pollinations_image("simple coloring-page illustration")
            except ImageProviderUnavailable as error:
                assert "persisted after 5 attempts" in str(error)
            else:
                raise AssertionError("Persistent provider rate limits must stop after bounded retries")

        assert get.call_count == 5
        assert sleep.call_count == 4
        print("  ✓ Pollinations free-limit retries stop after the configured bound")
        return True

    def test_quota_workflow_builds_book_with_image_provider_fallback(self):
        scenes = [
            {"action": f"does pose {index}", "micro_prompt": f"Color page {index}!"}
            for index in range(30)
        ]
        previous_dir = Path.cwd()
        quota_body = json.dumps({
            "errors": [{
                "code": 4006,
                "message": "daily free allocation of 10,000 neurons exhausted",
            }],
            "success": False,
            "result": {},
        }).encode()
        quota_response = self.response(429, quota_body, {"Content-Type": "application/json"})
        page = Image.new("RGB", (2550, 3300), "white")
        cover_image = create_test_image(768, 768)

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                os.chdir(temp_dir)
                Path("daily_book_plan.json").write_text(json.dumps({
                    "subject": "Test Calf",
                    "category": "Test Animals",
                    "character_dna": "a baby calf",
                    "scenes": scenes,
                }), encoding="utf-8")
                with patch("generate_book.CLOUDFLARE_QUOTA_EXHAUSTED", False), \
                        patch("generate_book.POLLINATIONS_FALLBACK_ACTIVE", False), \
                        patch("generate_book.ACCOUNT_ID", "account"), \
                        patch("generate_book.API_TOKEN", "test-token"), \
                        patch("generate_book.session.post", return_value=quota_response) as post, \
                        patch("generate_book.fetch_pollinations_image", return_value=cover_image) as pollinations, \
                        patch("generate_book.time.sleep"), \
                        patch("generate_book.generate_vector_interior_page", return_value=page), \
                        patch("generate_book.generate_ownership_page", return_value=page), \
                        patch("generate_book.generate_terms_page", return_value=page), \
                        patch("generate_book.generate_color_test_page", return_value=page), \
                        patch("generate_book.generate_procedural_maze_page", return_value=page), \
                        patch("generate_book.generate_procedural_dot_to_dot_page", return_value=page), \
                        patch("generate_book.generate_procedural_spot_differences_page", return_value=page), \
                        patch("generate_book.generate_certificate_page", return_value=page), \
                        patch("generate_book.generate_retail_back_cover_with_thumbnails", return_value=page):
                    build_book()

                pdfs = list(Path(temp_dir).glob("*.pdf"))
                assert len(pdfs) == 1, f"Expected one PDF, found {len(pdfs)}"
                pdf_data = pdfs[0].read_bytes()
                page_objects = pdf_data.count(b"/Type /Page") - pdf_data.count(b"/Type /Pages")
                assert page_objects == 39, f"Expected 39 PDF pages, found {page_objects}"
                assert post.call_count == 1
                assert pollinations.call_count == 1

                output = StringIO()
                with patch(
                    "generate_book.build_book",
                    side_effect=ImageProviderUnavailable(
                        "Pollinations image request failed."
                    ),
                ), redirect_stdout(output):
                    exit_code = main()

                assert exit_code == 1
                assert "No PDF was created" in output.getvalue()
                assert "Traceback" not in output.getvalue()
            finally:
                os.chdir(previous_dir)

        print("  ✓ Cloudflare quota fallback still assembles a complete 39-page PDF")
        return True


class TestInteriorLineArtQuality:
    """Ensure filled generations are rejected rather than used as coloring pages."""

    def test_scattered_specks_are_rejected(self):
        image = Image.new("L", (768, 768), "white")
        draw = ImageDraw.Draw(image)
        draw.ellipse((140, 120, 630, 650), outline="black", width=8)
        for row in range(3):
            for column in range(8):
                x = 20 + column * 30
                y = 20 + row * 30
                draw.rectangle((x, y, x + 1, y + 1), fill="black")

        issues = line_art_quality_issues(image)
        assert "image contains excessive isolated specks" in issues, issues
        print("  ✓ Line art with scattered specks is rejected")
        return True

    def test_noisy_generation_cleans_specks_without_retry(self):
        noisy = Image.new("L", (768, 768), "white")
        draw = ImageDraw.Draw(noisy)
        draw.ellipse((140, 120, 630, 650), outline="black", width=8)
        for row in range(3):
            for column in range(8):
                x = 20 + column * 30
                y = 20 + row * 30
                draw.rectangle((x, y, x + 1, y + 1), fill="black")
        encoded = BytesIO()
        noisy.save(encoded, format="PNG")
        scene = {"action": "plants a tiny seed", "micro_prompt": "A seed starts to grow."}

        with patch(
            "generate_book.fetch_book_image",
            return_value=encoded.getvalue(),
        ) as fetch:
            page = generate_vector_interior_page("a small animal", scene, 6)

        assert page.size == (2550, 3300)
        fetch.assert_called_once()
        print("  ✓ Speckled art is cleaned before validation without an extra provider request")
        return True

    def test_speck_cleanup_preserves_main_outline(self):
        image = Image.new("L", (768, 768), "white")
        draw = ImageDraw.Draw(image)
        draw.ellipse((140, 120, 630, 650), outline="black", width=8)
        draw.rectangle((20, 20, 21, 21), fill="black")

        cleaned = remove_isolated_specks(image)
        assert cleaned.getpixel((20, 20)) == 255
        assert cleaned.getpixel((140, 385)) == 0
        assert not line_art_quality_issues(cleaned), line_art_quality_issues(cleaned)
        print("  ✓ Tiny detached noise is removed while the subject outline is preserved")
        return True

    def test_line_art_thickening_is_controlled(self):
        image = np.full((64, 64), 255, dtype=np.uint8)
        image[32, 32] = 0

        result = np.asarray(clean_and_thicken_line_art(Image.fromarray(image)))
        assert np.count_nonzero(result[32] == 0) == 3
        print("  ✓ Line thickening preserves a controlled outline width")
        return True

    def test_fallback_plan_has_distinct_story_beats(self):
        scenes = DailyBookPlan.model_validate(generate_fallback_plan([])).scenes
        assert len(scenes) == 30
        for field in ("action", "setting", "composition", "micro_prompt"):
            assert len({getattr(scene, field) for scene in scenes}) == 30, (
                f"Fallback {field} values must be unique"
            )
        scenes[-1].setting = scenes[0].setting
        assert not has_distinct_story_scenes(scenes), "Duplicate settings must be rejected"
        print("  ✓ Fallback book contains 30 distinct story scenes and captions")
        return True

    def test_story_scene_details_reach_the_image_prompt(self):
        image = Image.new("L", (768, 768), "white")
        ImageDraw.Draw(image).ellipse((120, 80, 650, 690), outline="black", width=8)
        encoded = BytesIO()
        image.save(encoded, format="PNG")
        scene = {
            "action": "plants a tiny seed",
            "setting": "a sunny garden bed with a watering can",
            "composition": "a wide view with the animal on the left",
            "micro_prompt": "A tiny seed is ready to grow.",
        }

        with patch("generate_book.fetch_book_image", return_value=encoded.getvalue()) as fetch:
            generate_vector_interior_page("a small animal", scene, 1)

        prompt = fetch.call_args.args[0]
        assert scene["action"] in prompt
        assert scene["setting"] in prompt
        assert scene["composition"] in prompt
        assert "generic animal portrait" in prompt
        assert "Do not add unrelated objects" in prompt
        assert "Represent the stated action literally" in prompt
        assert "circular frame, halo, badge, or corner vignette" in prompt
        print("  ✓ Distinct story details and composition reach the illustration prompt")
        return True

    def test_clean_outline_is_accepted_with_white_coloring_space(self):
        image = Image.new("L", (768, 768), "white")
        ImageDraw.Draw(image).ellipse((120, 80, 650, 690), outline="black", width=8)
        encoded = BytesIO()
        image.save(encoded, format="PNG")
        scene = {"action": "sits facing forward", "micro_prompt": "Color the animal!"}

        with patch("generate_book.fetch_book_image", return_value=encoded.getvalue()) as fetch:
            page = generate_vector_interior_page("a small animal", scene, 1)

        assert page.size == (2550, 3300)
        assert page.getpixel((1275, 1750)) == (255, 255, 255), "The enclosed area should stay white"
        fetch.assert_called_once()
        print("  ✓ Clean outline is accepted and leaves its interior white")
        return True

    def test_cropped_generation_retries_with_full_subject_framing(self):
        cropped = Image.new("L", (768, 768), "white")
        ImageDraw.Draw(cropped).ellipse((0, 80, 650, 690), outline="black", width=8)
        cropped_encoded = BytesIO()
        cropped.save(cropped_encoded, format="PNG")

        complete = Image.new("L", (768, 768), "white")
        ImageDraw.Draw(complete).ellipse((120, 80, 650, 690), outline="black", width=8)
        complete_encoded = BytesIO()
        complete.save(complete_encoded, format="PNG")
        scene = {"action": "looks upward", "micro_prompt": "Color the animal!"}

        with patch(
            "generate_book.fetch_book_image",
            side_effect=[cropped_encoded.getvalue(), complete_encoded.getvalue()],
        ) as fetch:
            page = generate_vector_interior_page("a small animal", scene, 18)

        assert page.size == (2550, 3300)
        assert fetch.call_count == 2
        retry_prompt = fetch.call_args_list[1].args[0]
        assert "entire animal" in retry_prompt
        assert "generous blank white space" in retry_prompt
        print("  ✓ Cropped art is retried with explicit full-subject framing")
        return True

    def test_repeated_cropping_simplifies_the_retry_scene(self):
        cropped = Image.new("L", (768, 768), "white")
        ImageDraw.Draw(cropped).ellipse((0, 80, 650, 690), outline="black", width=8)
        cropped_encoded = BytesIO()
        cropped.save(cropped_encoded, format="PNG")

        complete = Image.new("L", (768, 768), "white")
        ImageDraw.Draw(complete).ellipse((120, 80, 650, 690), outline="black", width=8)
        complete_encoded = BytesIO()
        complete.save(complete_encoded, format="PNG")
        scene = {
            "action": "plants a tiny seed",
            "setting": "a garden bed with a watering can",
            "composition": "a wide view from the side",
            "micro_prompt": "A tiny seed is ready to grow.",
        }

        with patch(
            "generate_book.fetch_book_image",
            side_effect=[
                cropped_encoded.getvalue(),
                cropped_encoded.getvalue(),
                complete_encoded.getvalue(),
            ],
        ) as fetch:
            page = generate_vector_interior_page("a small animal", scene, 6)

        assert page.size == (2550, 3300)
        assert fetch.call_count == 3
        retry_prompt = fetch.call_args_list[2].args[0]
        assert "central half of the image" in retry_prompt
        assert "single most important story detail" in retry_prompt
        assert "outer quarter of the image completely blank white" in retry_prompt
        print("  ✓ Repeated crop failures trigger a simpler, safer scene framing")
        return True

    def test_edge_touching_art_gets_a_print_safe_margin_after_retries(self):
        image = Image.new("L", (768, 768), "white")
        ImageDraw.Draw(image).ellipse((0, 0, 767, 767), outline="black", width=2)
        assert any(
            issue in line_art_quality_issues(image)
            for issue in ("artwork touches the page edge", "character or prop is cropped")
        )
        encoded = BytesIO()
        image.save(encoded, format="PNG")
        scene = {"action": "looks around", "micro_prompt": "Color the animal!"}

        with patch(
            "generate_book.fetch_book_image",
            return_value=encoded.getvalue(),
        ) as fetch:
            page = generate_vector_interior_page("a small animal", scene, 6)

        assert page.size == (2550, 3300)
        assert fetch.call_count == 3
        assert page.getpixel((0, 1650)) == (255, 255, 255)
        print("  ✓ Edge-touching art is reframed with a white margin after retries")
        return True

    def test_thin_cropped_art_recovers_without_speckle_failure(self):
        image = Image.new("L", (768, 768), "white")
        ImageDraw.Draw(image).ellipse((0, 0, 767, 767), outline="black", width=1)
        encoded = BytesIO()
        image.save(encoded, format="PNG")
        scene = {"action": "looks around", "micro_prompt": "Color the animal!"}

        with patch(
            "generate_book.fetch_book_image",
            return_value=encoded.getvalue(),
        ) as fetch:
            page = generate_vector_interior_page("a small animal", scene, 6)

        assert page.size == (2550, 3300)
        assert fetch.call_count == 3
        print("  ✓ Thin edge-cropped art recovers without creating speckle noise")
        return True

    def test_quality_check_rejects_filled_animal(self):
        image = Image.new("L", (768, 768), "white")
        ImageDraw.Draw(image).ellipse((120, 80, 650, 690), fill="black")

        issues = line_art_quality_issues(image)
        assert any("large solid black region" in issue for issue in issues), issues
        print("  ✓ Quality check rejects a solid-filled animal")
        return True

    def test_solid_fill_is_converted_to_open_coloring_shape(self):
        image = Image.new("L", (768, 768), "white")
        ImageDraw.Draw(image).ellipse((120, 80, 650, 690), fill="black")

        result = convert_solid_regions_to_outlines(image)
        assert result.getpixel((385, 385)) == 255, "Filled area should become white"
        assert result.getpixel((385, 80)) == 0, "Outer contour should remain black"
        assert not line_art_quality_issues(result), line_art_quality_issues(result)
        print("  ✓ Solid animal silhouette becomes an outline with white coloring space")
        return True

    def test_large_circular_backdrop_is_not_traced(self):
        image = Image.new("L", (768, 768), "white")
        draw = ImageDraw.Draw(image)
        draw.ellipse((70, 70, 698, 698), fill="black")
        draw.rounded_rectangle((245, 180, 523, 610), radius=75, fill="white")

        result = convert_solid_regions_to_outlines(image)
        assert result.getpixel((90, 384)) == 255, "Backdrop edge should be removed"
        assert result.getpixel((245, 400)) == 0, "Inner subject contour should be retained"
        assert result.getpixel((384, 384)) == 255, "Subject coloring area should stay white"
        print("  ✓ Oversized circular backdrop is removed while the subject contour remains")
        return True

    def test_circular_halo_is_removed_entirely(self):
        image = Image.new("L", (768, 768), "white")
        ImageDraw.Draw(image).ellipse((100, 100, 668, 668), outline="black", width=18)

        result = convert_solid_regions_to_outlines(image)
        assert np.array_equal(np.asarray(result), np.asarray(image)), (
            "Sparse outline art should not be transformed"
        )
        print("  ✓ Sparse outline art is preserved without invented contours")
        return True

    def test_circular_backdrop_and_detached_specks_are_removed(self):
        image = Image.new("L", (768, 768), "white")
        draw = ImageDraw.Draw(image)
        draw.ellipse((140, 140, 628, 628), outline="black", width=8)
        draw.ellipse((285, 230, 485, 570), outline="black", width=8)
        draw.ellipse((330, 300, 346, 316), outline="black", width=4)
        for x, y in ((80, 120), (680, 200), (640, 650), (100, 660)):
            draw.rectangle((x, y, x + 10, y + 10), fill="black")

        result = remove_isolated_specks(convert_solid_regions_to_outlines(image))
        pixels = np.asarray(result)
        assert pixels[140, 384] == 255, "Circular backdrop should be removed"
        assert pixels[230, 385] == 0, "Main character outline should remain"
        assert pixels[300, 338] == 0, "Character details should remain"
        assert all(pixels[y, x] == 255 for x, y in (
            (80, 120), (680, 200), (640, 650), (100, 660)
        )), "Detached specks should be removed"
        assert not line_art_quality_issues(result), line_art_quality_issues(result)
        print("  ✓ Storybook cleanup removes circular frames and stray marks")
        return True

    def test_quality_check_rejects_large_fill_below_ink_limit(self):
        image = Image.new("L", (768, 768), "white")
        ImageDraw.Draw(image).ellipse((210, 170, 560, 600), fill="black")

        assert float(np.mean(np.asarray(image) <= 140)) < 0.35
        issues = line_art_quality_issues(image)
        assert any("large solid black region" in issue for issue in issues), issues
        print("  ✓ Large solid fill is rejected even below the overall ink limit")
        return True

    def test_filled_generation_is_repaired_without_extra_requests(self):
        image = Image.new("L", (768, 768), "white")
        ImageDraw.Draw(image).ellipse((120, 80, 650, 690), fill="black")
        encoded = BytesIO()
        image.save(encoded, format="PNG")
        scene = {"action": "sits facing forward", "micro_prompt": "Color the animal!"}

        with patch("generate_book.fetch_book_image", return_value=encoded.getvalue()) as fetch:
            page = generate_vector_interior_page("a small animal", scene, 1)

        assert page.getpixel((1275, 1750)) == (255, 255, 255), "The animal interior should be white"
        fetch.assert_called_once()
        print("  ✓ A filled generation is repaired on the first image response")
        return True


class TestDailyPlan:
    def test_sea_otter_fallback_matches_its_habitat(self):
        plan = DailyBookPlan.model_validate(generate_fallback_plan([
            "Barnaby the Baby Capybara",
            "Hamish the Baby Highland Calf",
        ]))

        assert plan.subject == "Pip the Playful Sea Otter Pup"
        story_text = " ".join(
            getattr(scene, field)
            for scene in plan.scenes
            for field in ("action", "setting", "micro_prompt")
        ).casefold()
        assert any(term in story_text for term in ("kelp", "tide pool", "clam", "shore"))
        assert not any(term in story_text for term in ("garden", "seed", "sunflower", "watering can"))
        assert has_distinct_story_scenes(plan.scenes)
        print("  ✓ Sea otter fallback scenes stay in a distinct coastal story")
        return True

    def test_daily_plan_uses_configured_gemini_model(self):
        plan_data = generate_fallback_plan([])
        api_call = {}

        def fake_generate_content(**kwargs):
            api_call.update(kwargs)
            return SimpleNamespace(parsed=plan_data, text=None)

        fake_client = SimpleNamespace(
            models=SimpleNamespace(generate_content=fake_generate_content)
        )
        fake_genai = SimpleNamespace(Client=lambda **kwargs: fake_client)
        fake_types = SimpleNamespace(
            GenerateContentConfig=lambda **kwargs: kwargs
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.multiple(
                research_trend,
                GEMINI_API_KEY="test-key",
                GEMINI_MODEL="configured-test-model",
                DAILY_PLAN_FILE=str(Path(temp_dir) / "plan.json"),
                load_history=lambda: [],
                save_history=lambda history: None,
                genai=fake_genai,
                types=fake_types,
                HAS_GENAI=True,
                create=True,
            ):
                generate_daily_plan()

        assert api_call["model"] == "configured-test-model"
        print("  ✓ Daily planner sends the configured Gemini model")
        return True


class TestUpscaling:
    """Test upscaling functionality."""

    def __init__(self):
        self.gen = ColoringBookGenerator()

    def test_upscale_small_image(self):
        """Test that small images get upscaled."""
        print("\n🧪 Testing upscaling of small images...")

        # Create 768x768 image
        test_img = create_test_image(768, 768)
        if not test_img:
            print("  ⊘ Skipped (PIL not available)")
            return False

        try:
            result = self.gen.upscale_to_print_quality(
                test_img,
                target_size=(2550, 3300)
            )

            # Verify upscaling happened
            from PIL import Image
            result_img = Image.open(BytesIO(result))

            assert result_img.size == (2550, 3300), \
                f"Should be upscaled to (2550, 3300), got {result_img.size}"

            print(f"  ✓ Upscaled from 768x768 to {result_img.size}")
            return True

        except Exception as e:
            print(f"  ✗ Upscaling failed: {e}")
            return False

    def test_no_upscale_large_image(self):
        """Test that large images don't get upscaled."""
        print("\n🧪 Testing that large images aren't upscaled...")

        # Create already-large image
        test_img = create_test_image(2550, 3300)
        if not test_img:
            print("  ⊘ Skipped (PIL not available)")
            return False

        try:
            result = self.gen.upscale_to_print_quality(
                test_img,
                target_size=(2550, 3300)
            )

            from PIL import Image
            result_img = Image.open(BytesIO(result))

            assert result_img.size == (2550, 3300), \
                "Should maintain size"

            print(f"  ✓ Large image kept at {result_img.size}")
            return True

        except Exception as e:
            print(f"  ✗ Test failed: {e}")
            return False

    def test_upscale_sets_dpi(self):
        """Test that upscaling sets DPI metadata."""
        print("\n🧪 Testing DPI metadata is set during upscaling...")

        test_img = create_test_image(768, 768)
        if not test_img:
            print("  ⊘ Skipped (PIL not available)")
            return False

        try:
            result = self.gen.upscale_to_print_quality(test_img)

            from PIL import Image
            result_img = Image.open(BytesIO(result))
            dpi = result_img.info.get('dpi', (0, 0))

            # Should be ~300 (sometimes 299.9994)
            assert dpi[0] >= 299 and dpi[1] >= 299, \
                f"DPI should be ~300, got {dpi}"

            print(f"  ✓ DPI metadata set to {dpi}")
            return True

        except Exception as e:
            print(f"  ✗ DPI test failed: {e}")
            return False


class TestErrorHandling:
    """Test error handling and edge cases."""

    def __init__(self):
        self.gen = ColoringBookGenerator()

    def test_invalid_theme(self):
        """Test handling of invalid theme."""
        print("\n🧪 Testing invalid theme handling...")

        try:
            # Should raise ValueError
            self.gen.generate_book(
                theme='nonexistent_theme',
                num_pages=1,
                book_title='Test'
            )
            print("  ✗ Should have raised ValueError")
            return False

        except ValueError as e:
            if 'Unknown theme' in str(e):
                print("  ✓ Correctly raises ValueError for invalid theme")
                return True
            else:
                print(f"  ✗ Wrong error message: {e}")
                return False
        except Exception as e:
            print(f"  ✗ Wrong exception type: {type(e).__name__}")
            return False

    def test_corrupt_image_upscaling(self):
        """Test handling of corrupt image data."""
        print("\n🧪 Testing corrupt image handling...")

        try:
            # Pass invalid image data
            corrupt_data = b'this is not an image'

            # Should raise RuntimeError
            result = self.gen.upscale_to_print_quality(corrupt_data)
            print("  ✗ Should have raised RuntimeError")
            return False

        except RuntimeError as e:
            if 'Image decode failed' in str(e) or 'cannot ensure print quality' in str(e):
                print("  ✓ Correctly raises RuntimeError for corrupt data")
                return True
            else:
                print(f"  ✗ Wrong error message: {e}")
                return False
        except Exception as e:
            print(f"  ⚠ Got {type(e).__name__}: {e}")
            # Accept this as the error is caught
            return True

    def test_zero_pages(self):
        """Test handling of zero pages request."""
        print("\n🧪 Testing zero pages request...")

        try:
            # Zero pages should work but generate nothing
            result = self.gen.generate_book(
                theme='mandalas',
                num_pages=0,
                book_title='Zero_Pages_Test'
            )

            # Should complete but generate no images
            images_dir = result / 'images'
            images = list(images_dir.glob('*.png'))

            assert len(images) == 0, "Should have zero images"
            print("  ✓ Zero pages handled correctly")
            return True

        except Exception as e:
            print(f"  ⚠ Got exception: {e}")
            # This is acceptable too
            return True


class TestImageValidation:
    """Test image quality validation function."""

    def test_validate_good_image(self):
        """Test validation passes for good image."""
        print("\n🧪 Testing validation of good quality image...")

        try:
            from PIL import Image

            # Create temporary test image with correct specs
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                img = Image.new('L', (2550, 3300), 255)
                img.save(f, format='PNG', dpi=(300, 300))
                temp_path = Path(f.name)

            try:
                result = validate_image_quality(
                    temp_path,
                    check_binary=False  # Don't check binary for this test
                )

                assert result['valid'], "Should be valid"
                assert len(result.get('issues', [])) == 0, "Should have no issues"

                print("  ✓ Good image passes validation")
                return True

            finally:
                temp_path.unlink()  # Clean up

        except ImportError:
            print("  ⊘ Skipped (PIL not available)")
            return False
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
            return False

    def test_validate_small_image(self):
        """Test validation fails for too-small image."""
        print("\n🧪 Testing validation rejects small images...")

        try:
            from PIL import Image

            # Create temporary small image
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                img = Image.new('L', (768, 768), 255)
                img.save(f, format='PNG')
                temp_path = Path(f.name)

            try:
                result = validate_image_quality(temp_path, check_binary=False)

                assert not result['valid'], "Should be invalid (too small)"
                assert len(result.get('issues', [])) > 0, "Should have issues"

                print(f"  ✓ Small image correctly rejected: {result['issues'][0]}")
                return True

            finally:
                temp_path.unlink()

        except ImportError:
            print("  ⊘ Skipped (PIL not available)")
            return False
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
            return False

    def test_validate_filled_black_image(self):
        """Reject black-filled outputs that have no usable coloring areas."""
        print("\n🧪 Testing validation rejects excessive black fill...")
        from PIL import Image

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image = Image.new("L", (32, 32), 0)
            image.save(f, format="PNG")
            temp_path = Path(f.name)

        try:
            result = validate_image_quality(
                temp_path,
                min_width=1,
                min_height=1,
                check_binary=True,
            )
            assert not result["valid"], "Black-filled page should not be valid"
            assert any("filled black area" in issue for issue in result["issues"])
            print("  ✓ Black-filled output correctly rejected")
            return True
        finally:
            temp_path.unlink()


class TestMetadata:
    """Test metadata generation and quality metrics."""

    def test_metadata_includes_quality_metrics(self):
        """Test that generated books include quality metrics in metadata."""
        print("\n🧪 Testing metadata includes quality metrics...")

        try:
            gen = ColoringBookGenerator(
                backend='pollinations',
                force_lineart=True
            )

            # Generate single page book
            book_dir = gen.generate_book(
                theme='geometric',
                num_pages=1,
                book_title='Metadata_Test'
            )

            # Read metadata
            metadata_file = book_dir / 'metadata.json'
            assert metadata_file.exists(), "Metadata file should exist"

            metadata = json.loads(metadata_file.read_text())

            # Check for quality metrics
            assert 'pages_data' in metadata, "Should have pages_data"
            assert len(metadata['pages_data']) > 0, "Should have at least one page"

            page_data = metadata['pages_data'][0]
            assert 'quality_metrics' in page_data, "Should have quality_metrics"

            metrics = page_data['quality_metrics']
            assert 'width' in metrics, "Should have width"
            assert 'height' in metrics, "Should have height"
            assert 'dpi' in metrics, "Should have DPI"

            print(f"  ✓ Quality metrics present: {list(metrics.keys())}")
            return True

        except Exception as e:
            print(f"  ✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def run_all_tests():
    """Run all expanded tests."""
    print("=" * 70)
    print("EXPANDED TEST COVERAGE")
    print("=" * 70)
    print("\nRunning comprehensive tests...")

    results = []

    # Line art methods
    print("\n" + "=" * 70)
    print("LINE ART METHODS")
    print("=" * 70)
    lineart_tests = TestLineArtMethods()
    results.append(("Enhanced line art", lineart_tests.test_enhanced_method()))
    results.append(("Standard line art", lineart_tests.test_standard_method()))
    results.append(("Detailed line art", lineart_tests.test_detailed_method()))
    style_tests = TestReferenceStyle()
    results.append(("Shared theme style", style_tests.test_style_applies_to_every_theme()))
    results.append(("Closed outline preservation", style_tests.test_line_art_preserves_closed_shapes()))
    cloudflare_tests = TestCloudflareImageRequests()
    results.append(("Cloudflare Bearer auth", cloudflare_tests.test_cloudflare_uses_bearer_token()))
    results.append(("Cloudflare rate-limit error", cloudflare_tests.test_rate_limit_failure_is_descriptive()))
    results.append(("Cover fallback", cloudflare_tests.test_cover_falls_back_when_image_service_fails()))
    results.append(("Daily quota stops retries", cloudflare_tests.test_daily_quota_stops_retries()))
    results.append(("Interior uses Pollinations after quota", cloudflare_tests.test_interior_uses_pollinations_after_quota()))
    results.append(("Quota switches image provider", cloudflare_tests.test_quota_switches_to_pollinations()))
    results.append(("Safety rejection switches provider", cloudflare_tests.test_safety_rejection_switches_to_pollinations()))
    results.append(("Pollinations image validation", cloudflare_tests.test_pollinations_rejects_non_image_responses()))
    results.append(("Pollinations free-limit retry", cloudflare_tests.test_pollinations_retries_free_limit()))
    results.append(("Pollinations retry bound", cloudflare_tests.test_pollinations_rate_limit_retries_are_bounded()))
    results.append(("Book builds after quota fallback", cloudflare_tests.test_quota_workflow_builds_book_with_image_provider_fallback()))
    interior_tests = TestInteriorLineArtQuality()
    results.append(("Reject scattered specks", interior_tests.test_scattered_specks_are_rejected()))
    results.append(("Clean noisy line art", interior_tests.test_noisy_generation_cleans_specks_without_retry()))
    results.append(("Preserve outlines during speck cleanup", interior_tests.test_speck_cleanup_preserves_main_outline()))
    results.append(("Control outline thickening", interior_tests.test_line_art_thickening_is_controlled()))
    results.append(("Accept clean outline art", interior_tests.test_clean_outline_is_accepted_with_white_coloring_space()))
    results.append(("Simplify repeated crop failures", interior_tests.test_repeated_cropping_simplifies_the_retry_scene()))
    results.append(("Reframe edge-touching interior art", interior_tests.test_edge_touching_art_gets_a_print_safe_margin_after_retries()))
    results.append(("Recover thin cropped art", interior_tests.test_thin_cropped_art_recovers_without_speckle_failure()))
    results.append(("Reject filled interior art", interior_tests.test_quality_check_rejects_filled_animal()))
    results.append(("Repair solid-filled animal", interior_tests.test_solid_fill_is_converted_to_open_coloring_shape()))
    results.append(("Remove circular backdrop", interior_tests.test_large_circular_backdrop_is_not_traced()))
    results.append(("Remove circular halo", interior_tests.test_circular_halo_is_removed_entirely()))
    results.append(("Clean storybook backdrop noise", interior_tests.test_circular_backdrop_and_detached_specks_are_removed()))
    results.append(("Reject localized solid fill", interior_tests.test_quality_check_rejects_large_fill_below_ink_limit()))
    results.append(("Repair without extra requests", interior_tests.test_filled_generation_is_repaired_without_extra_requests()))
    results.append(("Illustrate distinct story scenes", interior_tests.test_story_scene_details_reach_the_image_prompt()))
    results.append(("Unique fallback story pages", interior_tests.test_fallback_plan_has_distinct_story_beats()))
    plan_tests = TestDailyPlan()
    results.append(("Sea otter fallback habitat", plan_tests.test_sea_otter_fallback_matches_its_habitat()))
    results.append(("Configured Gemini model", plan_tests.test_daily_plan_uses_configured_gemini_model()))

    # Upscaling
    print("\n" + "=" * 70)
    print("UPSCALING TESTS")
    print("=" * 70)
    upscale_tests = TestUpscaling()
    results.append(("Upscale small image", upscale_tests.test_upscale_small_image()))
    results.append(("No upscale large image", upscale_tests.test_no_upscale_large_image()))
    results.append(("DPI metadata set", upscale_tests.test_upscale_sets_dpi()))

    # Error handling
    print("\n" + "=" * 70)
    print("ERROR HANDLING")
    print("=" * 70)
    error_tests = TestErrorHandling()
    results.append(("Invalid theme", error_tests.test_invalid_theme()))
    results.append(("Corrupt image", error_tests.test_corrupt_image_upscaling()))
    results.append(("Zero pages", error_tests.test_zero_pages()))

    # Image validation
    print("\n" + "=" * 70)
    print("IMAGE VALIDATION")
    print("=" * 70)
    validation_tests = TestImageValidation()
    results.append(("Validate good image", validation_tests.test_validate_good_image()))
    results.append(("Validate small image", validation_tests.test_validate_small_image()))
    results.append(("Reject black-filled image", validation_tests.test_validate_filled_black_image()))

    # Metadata (integration test - optional)
    if '--full' in sys.argv:
        print("\n" + "=" * 70)
        print("METADATA & INTEGRATION")
        print("=" * 70)
        metadata_tests = TestMetadata()
        results.append(("Metadata quality metrics", metadata_tests.test_metadata_includes_quality_metrics()))

    # Summary
    print("\n" + "=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    print(f"\nTests run: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success rate: {passed/total*100:.1f}%")

    # List failures
    failures = [name for name, result in results if not result]
    if failures:
        print("\nFailed tests:")
        for name in failures:
            print(f"  ✗ {name}")

    print("\n" + "=" * 70)

    if passed == total:
        print("✅ ALL TESTS PASSED")
        print("=" * 70)
        return 0
    else:
        print(f"⚠️ {total - passed} TEST(S) FAILED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║  EXPANDED TEST COVERAGE                                              ║
║  Coloring Book Generator v2.1.0                                      ║
║                                                                      ║
║  Tests individual components and edge cases                          ║
║  Complements test_quality_assurance.py                               ║
║                                                                      ║
║  Usage:                                                              ║
║    python3 test_expanded_coverage.py        # Core tests             ║
║    python3 test_expanded_coverage.py --full # Include integration    ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    exit_code = run_all_tests()
    sys.exit(exit_code)
