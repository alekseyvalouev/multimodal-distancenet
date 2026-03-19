from pathlib import Path
import os

from dotenv import load_dotenv  # pip install python-dotenv
from google import genai        # pip install google-genai
from PIL import Image, ImageDraw, ImageFont

from prompts import annotate_landmarks

from pydantic import BaseModel
from typing import List

class Landmarks(BaseModel):
    landmarks: List[str]


def draw_landmarks_on_image(image_path: str, landmarks: Landmarks, output_path: str) -> None:
    """Create a PNG with the image and landmark labels drawn on it."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # Layout: image on left, landmarks panel on right (max 400px wide)
    panel_width = min(400, max(280, w // 3))
    total_width = w + panel_width
    pad = 16
    font_size = 14
    line_height = 22

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    # Use a temporary draw to measure text and compute wrapped lines
    dummy = Image.new("RGB", (1, 1))
    draw_tmp = ImageDraw.Draw(dummy)
    max_text_w = panel_width - 2 * pad

    all_entries: list[tuple[int, list[str]]] = []  # (index, wrapped lines)
    for i, label in enumerate(landmarks.landmarks, 1):
        words = label.split()
        lines = []
        current = []
        for word in words:
            test = " ".join(current + [word])
            bbox = draw_tmp.textbbox((0, 0), test, font=font)
            if bbox[2] - bbox[0] <= max_text_w:
                current.append(word)
            else:
                if current:
                    lines.append(" ".join(current))
                current = [word]
        if current:
            lines.append(" ".join(current))
        all_entries.append((i, lines))

    # Height needed for panel: title + spacing + all lines
    panel_content_height = pad + line_height + 8
    for i, lines in all_entries:
        panel_content_height += len(lines) * line_height + 4
    panel_content_height += pad

    total_height = max(h, panel_content_height)
    out = Image.new("RGB", (total_width, total_height), color=(248, 248, 250))
    out.paste(img, (0, 0))

    draw = ImageDraw.Draw(out)
    draw.rectangle([w, 0, total_width, total_height], fill=(248, 248, 250), outline=(220, 220, 220))

    # Title
    draw.text((w + pad, pad), "Landmarks", fill=(40, 40, 40), font=font)
    y = pad + line_height + 8

    for i, lines in all_entries:
        for idx, line in enumerate(lines):
            prefix = f"{i}. " if idx == 0 else "   "
            draw.text((w + pad, y), prefix + line, fill=(60, 60, 60), font=font)
            y += line_height
        y += 4

    out.save(output_path, "PNG")
    print(f"Saved: {output_path}")


def main() -> None:
    # Load .env next to this script
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(env_path)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not found in .env")
        return

    # Client picks up GEMINI_API_KEY from env, but we pass it explicitly too
    client = genai.Client(api_key=api_key)

    image_path = "/hdd/sacson/Nov-17-2022-bww8_00000000_0/1.jpg"
    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    try:
        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=[
                genai.types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/jpeg',
                ),
                annotate_landmarks
            ],
            #config={
            #    "response_mime_type": "application/json",
            #    "response_schema": Landmarks,
            #},
        )
        print("✅ Gemini API call succeeded.")
        print("Model response:")
        print(response.text)

        #landmarks = Landmarks.model_validate_json(response.text)
        #out_dir = Path(__file__).resolve().parent
        #output_png = out_dir / "landmarks_output.png"
        #draw_landmarks_on_image(image_path, landmarks, str(output_png))
    except Exception as e:
        print("❌ Gemini API call failed.")
        print(f"Error: {e!r}")


if __name__ == "__main__":
    main()