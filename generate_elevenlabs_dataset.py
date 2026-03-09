import os
import json
import time
import uuid
import argparse
import requests
from pathlib import Path

# Default known working voices (Multilingual v2 supported)
VOICES = {
    "Brian": "nPczCjzI2devNBz1zQrb",
    "Chris": "iP95p4xoKVk53GoZ742B",
    "Laura": "FGY2WhTYpPnrIDTdsi5Y",
    "George": "JBFqnCBsd6RMkjVDRZzb",
    "Sarah": "EXAVITQu4vr4xnSDxMaL",
    "Eric": "cjVigY5qzO86Huf0OWal"
}

# Real-world challenging phrases for deepfake detection
# (Mix of short commands, emotional phrases, and long monotonic reads)
PHRASES = [
    "I need you to wire the funds to this account immediately before the deadline passes.",
    "Hey, this is me calling from my new number, the old one broke.",
    "The quarterly financial report shows a fifteen percent increase in overall revenue.",
    "Can you hear me? Hello? Yeah, my signal is really bad.",
    "Don't forget to submit your timesheets by 5 PM on Friday, otherwise payroll will be delayed.",
    "This is an automated alert from your bank regarding a suspicious charge on your account.",
    "Mom, I'm at the police station, I was in an accident and they need bail money.",
    "Please press one to speak with a customer support representative.",
    "I am absolutely certain that the data shown here is completely authentic.",
    "Well, if you really think about it, there are only so many ways to solve this problem.",
    "I'm sorry, I don't think I can make it to the meeting today, I'm feeling a bit under the weather.",
    "If you want to access the exclusive content, you need to click the link in the description.",
    "Let me know when you get this message so we can coordinate the drop off.",
    "Are you kidding me? There is absolutely no way I'm agreeing to those terms.",
    "Welcome back to the channel, today we're going to dive right into the newest update."
]

def generate_audio(voice_id, text, api_key, output_path):
    """Hits the ElevenLabs API to generate speech."""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    print(f"  [>] Generating {voice_id}... (Please wait, ElevenLabs can take 2-5 seconds)")
    try:
        response = requests.post(url, json=data, headers=headers, timeout=30)
    except requests.exceptions.Timeout:
        print("  [!] Request timed out while waiting for ElevenLabs servers.")
        return False
    except requests.exceptions.RequestException as e:
        print(f"  [!] Connection Error: {e}")
        return False
    
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        return True
    else:
        print(f"  [!] Failed to generate ({response.status_code}): {response.text}")
        return False

def main():
    parser = argparse.ArgumentParser(description="ElevenLabs Bulk Dataset Generator")
    parser.add_argument("--key", type=str, help="ElevenLabs API Key (or use ELEVENLABS_API_KEY env var)")
    parser.add_argument("--count", type=int, default=10, help="Number of audio samples to generate")
    parser.add_argument("--out", type=str, default="data/audio_forensics/raw/fake/elevenlabs", help="Output directory")
    args = parser.parse_args()

    api_key = args.key or os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("[!] ERROR: No ElevenLabs API key provided.")
        print("    Usage: python generate_elevenlabs_dataset.py --key YOUR_API_KEY")
        print("    Or export ELEVENLABS_API_KEY=YOUR_API_KEY")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[*] Starting ElevenLabs generation ({args.count} samples requested)")
    print(f"[*] Target Directory: {out_dir}\n")

    import random
    success_count = 0
    
    for i in range(args.count):
        voice_name, voice_id = random.choice(list(VOICES.items()))
        phrase = random.choice(PHRASES)
        
        file_name = f"el_{voice_name.lower()}_{uuid.uuid4().hex[:8]}.mp3"
        out_path = out_dir / file_name
        
        print(f"  [{i+1}/{args.count}] Generating {voice_name}...")
        
        if generate_audio(voice_id, phrase, api_key, out_path):
            success_count += 1
            print(f"      -> Saved to {out_path}")
        
        # Rate limit protection (prevent spamming API too fast)
        time.sleep(0.5)

    print(f"\n[+] Generation complete! Successfully created {success_count} ElevenLabs samples.")
    print(f"[*] The new fake audio sits at: {out_dir}")
    print("[*] Automatically picked up by train_audio_model.py!")

if __name__ == "__main__":
    main()
