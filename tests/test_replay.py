"""Offline test: strip a synthetic Pangram response, rehydrate it, get the same bytes back."""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "scripts", "pangram_replay.py")

TEXT = "First paragraph of a made-up abstract.\n\nSecond paragraph,‌ with an invisible joiner."
SCORED = TEXT.replace("\n\n", "\n").replace("‌", "")
W1, W2 = SCORED[:38], SCORED[38:]
RAW = {"id": "10.0000/test-1", "task_id": "t-1", "response": {
    "stage": "STAGE_SUCCESS", "text": SCORED, "version": "4.1", "prediction": "test",
    "prediction_short": "Mixed", "fraction_ai": 0.5, "fraction_ai_assisted": 0.0, "fraction_human": 0.5,
    "headline": "AI Detected", "num_ai_segments": 1, "num_ai_assisted_segments": 0, "num_human_segments": 1,
    "windows": [
        {"text": W1, "label": "Human Written", "ai_assistance_score": 0.01, "confidence": "High",
         "start_index": 0, "end_index": len(W1), "word_count": 6},
        {"text": W2, "label": "AI-Generated", "ai_assistance_score": 0.97, "confidence": "Medium",
         "start_index": len(W1), "end_index": len(SCORED), "word_count": 7}]}}
SAMPLE = {"id": "10.0000/test-1", "year": 2099, "n_words": len(TEXT.split()), "eligible": True, "text": TEXT}


def main():
    d = tempfile.mkdtemp()
    try:
        for sub in ("data/samples", "results", "replay", "out"):
            os.makedirs(os.path.join(d, sub))
        with open(os.path.join(d, "data/samples/sample_2099_n1_seed1.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(SAMPLE, ensure_ascii=False) + "\n")
        raw_line = json.dumps(RAW, ensure_ascii=False) + "\n"
        with open(os.path.join(d, "results/t.raw.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            f.write(raw_line)
        py = [sys.executable, "-I", SCRIPT]
        subprocess.run([*py, "strip", "--results", f"{d}/results", "--out", f"{d}/replay", "--root", d], check=True)
        stripped = open(os.path.join(d, "replay/t.replay.jsonl"), encoding="utf-8").read()
        assert "paragraph" not in stripped, "text survived stripping"
        subprocess.run([*py, "rehydrate", "--replay", f"{d}/replay", "--results", f"{d}/out", "--root", d], check=True)
        back = open(os.path.join(d, "out/t.raw.jsonl"), encoding="utf-8").read()
        assert back == raw_line, "rehydrated response differs from the original"
        print("TEST PASSED: replay round trip is exact and text-free")
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
