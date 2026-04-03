"""
Generate EC⁸ logo variations using Azure FLUX.1-Kontext-pro.
Outputs saved to static/img/branding/
"""
import os
import httpx
import base64
import json
from datetime import datetime

# Azure AI Services — East US (FLUX.1-Kontext-pro)
ENDPOINT = "https://ai-peterwilson7092ai011379814834.services.ai.azure.com"
API_KEY = os.environ.get("AZURE_EASTUS_KEY", "")
MODEL = "FLUX.1-Kontext-pro"

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "static", "img", "branding")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# EC⁸ brand prompts — Miami treatment variations
PROMPTS = [
    {
        "name": "ec8_primary_dark",
        "prompt": (
            "A premium minimalist logo on a pure black background. "
            "Bold white letters 'EC' in a clean modern sans-serif font, with a small superscript '8' "
            "tucked tight to the upper right of the 'C' like a mathematical exponent (EC to the 8th power). "
            "Below the EC⁸ text, the tagline 'LET'S EAT.' in a thin elegant font with a subtle cyan-to-hot-pink gradient. "
            "A thin vertical gradient line (cyan to pink) separates the logo from descriptive text. "
            "Professional, sleek, Miami nightlife aesthetic. No other text or decoration."
        ),
    },
    {
        "name": "ec8_neon_glow",
        "prompt": (
            "A glowing neon logo on a dark background. The letters 'EC' in large bold white font with "
            "a small raised '8' superscript like a math exponent. The entire EC⁸ text has a soft cyan and pink neon glow effect, "
            "like a Miami South Beach neon sign at night. Below it 'LET'S EAT.' in thin pink neon lettering. "
            "Reflections on a dark glossy surface below. Premium sports brand aesthetic. No background clutter."
        ),
    },
    {
        "name": "ec8_gradient_badge",
        "prompt": (
            "A modern badge/shield logo. Dark charcoal background with a subtle gradient border from cyan (#00F0FF) "
            "to hot pink (#FF1493). Inside the badge: 'EC' in bold white with a superscript '8' as an exponent. "
            "Below it 'EDGE CREW' in small spaced-out letters. The badge has a slight 3D metallic feel. "
            "Clean, premium, sports analytics brand. Miami color palette."
        ),
    },
    {
        "name": "ec8_app_icon",
        "prompt": (
            "A square app icon with rounded corners. Dark gradient background from deep navy to black. "
            "Centered: 'EC' in bold white with a superscript '8' like a math power notation. "
            "Subtle cyan-to-pink gradient glow behind the text. Clean, minimal, no other text. "
            "Professional sports tech app icon style. High contrast, readable at small sizes."
        ),
    },
    {
        "name": "ec8_miami_full",
        "prompt": (
            "A full brand lockup logo. Black background. Left side: large bold 'EC' with superscript '8' "
            "in white with subtle gradient highlights. A thin vertical line in cyan-to-pink gradient. "
            "Right side: seven words stacked vertically in thin gray text: EYE, EDGE, NUMBER, FORMULA, "
            "CALCULUS, MARKET, MACHINE. At bottom right: 'EC⁸' in small pink text. "
            "Premium minimalist sports analytics brand. Miami aesthetic."
        ),
    },
    {
        "name": "ec8_white_bg",
        "prompt": (
            "A clean logo on a pure white background. 'EC' in bold black with a small superscript '8' "
            "as a mathematical exponent. Below: 'LET'S EAT.' in thin font with cyan-to-pink gradient color. "
            "Minimal, professional, no other elements. Suitable for light mode UI and print materials."
        ),
    },
]


def generate_image(prompt: str, name: str, size: str = "1024x1024"):
    """Call Azure FLUX.1-Kontext-pro to generate an image."""
    url = f"{ENDPOINT}/openai/deployments/{MODEL}/images/generations?api-version=2024-12-01-preview"
    headers = {
        "api-key": API_KEY,
        "Content-Type": "application/json",
    }
    body = {
        "model": MODEL,
        "prompt": prompt,
        "n": 1,
        "size": size,
    }

    print(f"\n🎨 Generating: {name}")
    print(f"   Prompt: {prompt[:80]}...")

    try:
        resp = httpx.post(url, json=body, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        # Handle different response formats
        if "data" in data and len(data["data"]) > 0:
            img_data = data["data"][0]
            if "b64_json" in img_data:
                img_bytes = base64.b64decode(img_data["b64_json"])
            elif "url" in img_data:
                # Download from URL
                img_resp = httpx.get(img_data["url"], timeout=60)
                img_bytes = img_resp.content
            else:
                print(f"   ❌ Unexpected response format: {list(img_data.keys())}")
                return None
        else:
            print(f"   ❌ No image data in response: {json.dumps(data)[:200]}")
            return None

        out_path = os.path.join(OUTPUT_DIR, f"{name}.png")
        with open(out_path, "wb") as f:
            f.write(img_bytes)
        print(f"   ✅ Saved: {out_path} ({len(img_bytes) // 1024} KB)")
        return out_path

    except httpx.HTTPStatusError as e:
        print(f"   ❌ HTTP {e.response.status_code}: {e.response.text[:300]}")
        return None
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


def main():
    print("=" * 60)
    print("EC⁸ Logo Generator — FLUX.1-Kontext-pro")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    results = []
    for p in PROMPTS:
        path = generate_image(p["prompt"], p["name"])
        results.append({"name": p["name"], "path": path, "success": path is not None})

    print("\n" + "=" * 60)
    print("RESULTS:")
    for r in results:
        status = "✅" if r["success"] else "❌"
        print(f"  {status} {r['name']}")

    success = sum(1 for r in results if r["success"])
    print(f"\n{success}/{len(results)} generated successfully")


if __name__ == "__main__":
    main()
