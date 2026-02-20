
"""
This module implements the Evaluator Agent using Google Gemini 1.5 Pro.
It analyzes Grad-CAM visualizations to explain the model's predictions.
"""

import os
import glob
import textwrap
from datetime import datetime
import google.generativeai as genai
from PIL import Image
from backend.src import config
from backend.src import evaluate
from backend.src.utilities.rate_limiter import AsyncRateLimiter

# Initialize global rate limiter for Gemini API
# Default for free tier is usually 15 RPM for Gemini 1.5 Flash/Pro
limiter = AsyncRateLimiter(rpm=15)


def _parse_gradcam_filename(image_path: str):
    """
    Parse filename format produced by evaluate.py:
    {record_name}_min{minute}_{class_label}_{confidence:.2f}.png
    """
    filename = os.path.basename(image_path)
    parts = filename.replace(".png", "").split("_")
    confidence = float(parts[-1])
    prediction = parts[-2]
    minute = int(parts[-3].replace("min", ""))
    return minute, prediction, confidence

def configure_genai():
    """Configures the Gemini API with the key from config."""
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in configuration or environment variables.")
    genai.configure(api_key=config.GEMINI_API_KEY)

async def analyze_image(image_path: str, record_name: str, minute: int, prediction: str, confidence: float) -> str:
    """
    Sends a Grad-CAM image to Gemini 1.5 Pro for analysis.
    Wait for the rate limiter before making the API call.
    """
    await limiter.wait()
    
    model = genai.GenerativeModel('gemini-1.5-flash') # Updated to a stable model name

    img = Image.open(image_path)

    prompt = textwrap.dedent(f"""
        You are an expert cardiologist and AI interpretability specialist.
        
        **Context**:
        - Patient Record: {record_name}
        - ECG Segment: Minute {minute}
        - Model Prediction: **{prediction}**
        - Confidence: {confidence:.2f}
        
        **Input Image**:
        The attached image shows a 1-minute ECG signal (split into 2-second segments).
        The overlaid heatmap (red/warm colors) indicates the "Grad-CAM" attention - the regions the model found most important for its prediction.
        
        **Task**:
        1.  **Morphological Analysis**: Look at the regions with high attention (red). Do you see specific ECG features? (e.g., R-peak amplitude changes, missed beats, irregular intervals, signal quality issues).
        2.  **Physiological Correlation**: correct to physiology, does this attention make sense for a prediction of "{prediction}"? (e.g., Amplitude reduction often indicates Apnea).
        3.  **Explain**: Provide a concise explanation of *why* the model likely made this prediction based on the visual evidence.
        
        Output in Markdown format.
    """)

    try:
        response = await model.generate_content_async([prompt, img])
        return response.text
    except Exception as e:
        return f"Error evaluating image: {e}"


async def generate_chat_analysis_for_record(record_name: str, visualize_count: int = 3) -> dict:
    """
    Generate a single chat-friendly analysis message for a record.
    Returns:
        {
            "analysis": str,
            "meta": {"analyzed_minutes": int, "report_path": str}
        }
    """
    configure_genai()

    gradcam_dir = os.path.join(config.RESULTS_DIR, "gradcam")
    os.makedirs(gradcam_dir, exist_ok=True)

    pattern = os.path.join(gradcam_dir, f"{record_name}_min*.png")
    images = glob.glob(pattern)

    if not images:
        evaluate.generate_gradcam_for_record(record_name, visualize_count=visualize_count)
        images = glob.glob(pattern)

    parsed_images = []
    for image_path in images:
        try:
            minute, prediction, confidence = _parse_gradcam_filename(image_path)
            parsed_images.append({
                "image_path": image_path,
                "minute": minute,
                "prediction": prediction,
                "confidence": confidence
            })
        except Exception:
            continue

    if not parsed_images:
        raise ValueError(f"No Grad-CAM images available for record {record_name}.")

    parsed_images.sort(key=lambda item: item["confidence"], reverse=True)
    selected = parsed_images[:max(1, int(visualize_count))]

    sections = [f"## ECG Agent Analysis for `{record_name}`", ""]
    report_content = f"# Evaluator Agent Report: {record_name}\n\n"
    report_content += f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    for item in selected:
        analysis = await analyze_image(
            item["image_path"],
            record_name,
            item["minute"],
            item["prediction"],
            item["confidence"],
        )
        sections.append(
            f"### Minute {item['minute']} - {item['prediction']} (confidence {item['confidence']:.2f})"
        )
        sections.append(analysis)
        sections.append("")

        report_content += f"## Analysis: Minute {item['minute']}\n"
        report_content += (
            f"**Prediction**: {item['prediction']} (Confidence: {item['confidence']:.2f})\n\n"
        )
        report_content += f"![Grad-CAM]({item['image_path']})\n\n"
        report_content += analysis + "\n\n"
        report_content += "---\n\n"

    reports_dir = os.path.join(config.RESULTS_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, f"{record_name}_analysis.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    return {
        "analysis": "\n".join(sections).strip(),
        "meta": {
            "analyzed_minutes": len(selected),
            "report_path": report_path,
        },
    }

async def generate_report_for_record(record_name: str):
    """
    Generates a full report for a record by analyzing available Grad-CAM images.
    If no images exist, it generates them first.
    """
    configure_genai()
    
    gradcam_dir = os.path.join(config.RESULTS_DIR, "gradcam")
    
    # Check if images exist, if not, generate them
    # Pattern: {record_name}_min{minute}_{class_label}_{confidence:.2f}.png
    pattern = os.path.join(gradcam_dir, f"{record_name}_min*.png")
    images = glob.glob(pattern)
    
    if not images:
        print(f"No existing visualizations found for {record_name}. Generating them now...")
        evaluate.generate_gradcam_for_record(record_name)
        images = glob.glob(pattern)
        
    if not images:
        print(f"Could not generate or find images for {record_name}.")
        return

    print(f"Found {len(images)} visualizations for {record_name}. Starting analysis...", flush=True)
    
    report_content = f"# Evaluator Agent Report: {record_name}\n\n"
    report_content += f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    for image_path in images:
        filename = os.path.basename(image_path)
        # Parse info from filename: a01_min489_Apnea_1.00.png
        try:
            parts = filename.replace(".png", "").split("_")
            # Expected: [record, "min"+minute, label, confidence]
            # But record name might contain underscores if changed, though usually 'a01'
            # Let's assume standard format from evaluate.py
            
            # evaluate.py save format: f"{record_name}_min{minute}_{class_label}_{confidence:.2f}.png"
            confidence = float(parts[-1])
            prediction = parts[-2]
            minute = int(parts[-3].replace("min", ""))
            
            print(f"Analyzing Minute {minute} ({prediction})...", flush=True)
            analysis = await analyze_image(image_path, record_name, minute, prediction, confidence)
            
            report_content += f"## Analysis: Minute {minute}\n"
            report_content += f"**Prediction**: {prediction} (Confidence: {confidence:.2f})\n\n"
            report_content += f"![Grad-CAM]({image_path})\n\n"
            report_content += analysis + "\n\n"
            report_content += "---\n\n"
            
        except Exception as e:
            print(f"Skipping file {filename} due to parse/analyzing error: {e}")
            continue

    # Save Report
    reports_dir = os.path.join(config.RESULTS_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, f"{record_name}_analysis.md")
    
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(report_content)
        
    print(f"\nReport saved to: {report_path}")
