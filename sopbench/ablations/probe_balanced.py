"""Probe v3: Balanced test — query BOTH halves of each second to verify sampling bias.

Tests whether fps=1 can truly see 0.5s codes, or if our previous 92% result
was due to only querying codes that aligned with Gemini's ~0.47s sampling offset.
"""

import json
import os
import random
import string
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parent.parent
PROBE_DIR = ROOT / "videos" / "probe"


def run_test(client, video_path, codes, code_dur, fps_val, max_retries=3):
    """Query both first-half and second-half codes within each second."""

    # Upload with retry
    for attempt in range(max_retries):
        try:
            video_file = client.files.upload(
                file=str(video_path),
                config=types.UploadFileConfig(mime_type="video/mp4"),
            )
            while video_file.state == "PROCESSING":
                time.sleep(3)
                video_file = client.files.get(name=video_file.name)
            break
        except Exception as e:
            print(f"    Upload attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(10)
            else:
                raise

    # Build BALANCED queries
    first_half = []
    second_half = []
    for i, info in codes.items():
        center = (info["start"] + info["end"]) / 2
        if center >= 27:
            continue
        frac = info["start"] % 1.0
        if frac < 0.01:  # starts at x.0
            first_half.append((center, info["code"]))
        elif abs(frac - 0.5) < 0.01:  # starts at x.5
            second_half.append((center, info["code"]))

    # For 0.25s codes: first quarter vs third quarter
    if code_dur <= 0.25:
        first_half = []
        second_half = []
        for i, info in codes.items():
            center = (info["start"] + info["end"]) / 2
            if center >= 27:
                continue
            frac = info["start"] % 1.0
            if abs(frac) < 0.01:  # x.00
                first_half.append((center, info["code"]))
            elif abs(frac - 0.5) < 0.01:  # x.50
                second_half.append((center, info["code"]))

    step1 = max(1, len(first_half) // 6)
    step2 = max(1, len(second_half) // 6)
    # Tag each entry with its half BEFORE sorting
    selected = [(c, e, "1st") for c, e in first_half[::step1][:6]] + \
               [(c, e, "2nd") for c, e in second_half[::step2][:6]]
    selected.sort(key=lambda x: x[0])

    ts_list = "\n".join(
        f"- {int(t//60):02d}:{t%60:05.2f}" for t, _, _ in selected
    )

    prompt = f"""This is a 30-second test video with a 4-character code in large white text on black background.
Each code is displayed for exactly {code_dur} seconds before changing.

Read the EXACT 4-character code at each timestamp:

{ts_list}

Return JSON: {{"readings": [{{"timestamp": "MM:SS.ss", "code": "XXXX"}}, ...]}}"""

    # Call with retry
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part.from_text(text=prompt),
                    types.Part(
                        file_data=types.FileData(
                            file_uri=video_file.uri, mime_type="video/mp4"
                        ),
                        video_metadata=types.VideoMetadata(fps=float(fps_val)),
                    ),
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", temperature=0.0
                ),
            )
            break
        except Exception as e:
            print(f"    Generate attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(10)
            else:
                client.files.delete(name=video_file.name)
                raise

    client.files.delete(name=video_file.name)

    readings = json.loads(response.text).get("readings", [])

    correct_1st = 0
    correct_2nd = 0
    total_1st = 0
    total_2nd = 0

    for idx, ((center, expected, half_tag), r) in enumerate(zip(selected, readings)):
        got = str(r.get("code", "")).strip().upper()
        match = expected == got
        is_first = half_tag == "1st"
        half = half_tag
        icon = "Y" if match else "X"
        print(
            f"    [{icon}] t={center:05.2f}s ({half}-half) "
            f"expected={expected} got={got}"
        )

        if is_first:
            total_1st += 1
            if match:
                correct_1st += 1
        else:
            total_2nd += 1
            if match:
                correct_2nd += 1

    n = total_1st + total_2nd
    total_correct = correct_1st + correct_2nd
    acc_1st = correct_1st / total_1st if total_1st else 0
    acc_2nd = correct_2nd / total_2nd if total_2nd else 0
    acc_total = total_correct / n if n else 0

    print(f"  Total: {total_correct}/{n} ({acc_total:.0%})")
    print(f"  1st half: {correct_1st}/{total_1st} ({acc_1st:.0%})")
    print(f"  2nd half: {correct_2nd}/{total_2nd} ({acc_2nd:.0%})")

    return {
        "total": acc_total,
        "first_half": acc_1st,
        "second_half": acc_2nd,
    }


def main():
    load_dotenv(ROOT / ".env")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    # Regenerate code maps (same seed as probe v2)
    def make_codes(code_dur, seed=42):
        rng = random.Random(seed)
        n = int(30 / code_dur)
        codes = {}
        for i in range(n):
            ts = i * code_dur
            code = "".join(rng.choices(string.ascii_uppercase + string.digits, k=4))
            codes[i] = {"code": code, "start": ts, "end": ts + code_dur}
        return codes

    results = []

    # 0.5s codes at fps=1,2,4
    codes_05 = make_codes(0.5)
    video_05 = PROBE_DIR / "probe_0.5s_codes.mp4"
    for fps_val in [1, 2, 4]:
        print(f"\n{'='*50}")
        print(f"0.5s codes | fps={fps_val} | BALANCED")
        print(f"{'='*50}")
        r = run_test(client, video_05, codes_05, 0.5, fps_val)
        r["code_dur"] = 0.5
        r["fps"] = fps_val
        results.append(r)
        time.sleep(5)

    # 0.25s codes at fps=1,2,4
    codes_025 = make_codes(0.25)
    video_025 = PROBE_DIR / "probe_0.25s_codes.mp4"
    for fps_val in [1, 2, 4]:
        print(f"\n{'='*50}")
        print(f"0.25s codes | fps={fps_val} | BALANCED")
        print(f"{'='*50}")
        r = run_test(client, video_025, codes_025, 0.25, fps_val)
        r["code_dur"] = 0.25
        r["fps"] = fps_val
        results.append(r)
        time.sleep(5)

    # Summary
    print(f"\n{'='*60}")
    print("BALANCED PROBE RESULTS")
    print(f"{'='*60}")
    print(f"{'Config':<25s} | {'1st half':>8s} | {'2nd half':>8s} | {'Total':>7s}")
    print("-" * 55)
    for r in results:
        label = f"code={r['code_dur']}s fps={r['fps']}"
        print(
            f"{label:<25s} | {r['first_half']:>7.0%} | {r['second_half']:>7.0%} | {r['total']:>6.0%}"
        )

    with open(PROBE_DIR / "probe_v3_balanced.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
