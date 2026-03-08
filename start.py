#!/usr/bin/env python3
"""Quick start script for KugelAudio.

This script provides easy access to all KugelAudio functionality:
- Launch the web UI
- Generate speech from command line
- Verify watermarks

Usage:
    # Launch web UI (default)
    python start.py
    python start.py ui
    python start.py ui --share  # With public link

    # Generate speech
    python start.py generate "Hello world!" -o output.wav
    python start.py generate "Clone my voice" -r reference.wav -o cloned.wav

    # Verify watermark
    python start.py verify audio.wav
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="KugelAudio - Open-source text-to-speech with voice cloning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch web interface (default action)
  python start.py
  python start.py ui
  python start.py ui --share
  python start.py ui --host 0.0.0.0 --port 8080
  
  # Generate speech from command line
  python start.py generate "Hello world!" -o output.wav
  python start.py generate "Clone my voice" -r reference.wav -o cloned.wav
  python start.py generate "Premium quality" --model kugelaudio/kugelaudio-0-open -o premium.wav
  
  # Verify watermark in audio
  python start.py verify audio.wav
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # UI command (default)
    ui_parser = subparsers.add_parser("ui", help="Launch Gradio web interface")
    ui_parser.add_argument("--share", action="store_true", help="Create public share link")
    ui_parser.add_argument(
        "--host", default="127.0.0.1", help="Server hostname (use 0.0.0.0 for network access)"
    )
    ui_parser.add_argument("--port", type=int, default=7860, help="Server port")
    ui_parser.add_argument(
        "--model", default="kugelaudio/kugelaudio-0-open", help="Default model to load"
    )
    ui_parser.add_argument(
        "--low-vram",
        action="store_true",
        help="Enable low-VRAM mode (~3-4GB VRAM, slower inference, needs ~15GB RAM)",
    )
    ui_parser.add_argument(
        "--quantize",
        action="store_true",
        help="Load model in 4-bit (NF4) quantization (~4GB VRAM, ~4GB RAM). Requires: pip install bitsandbytes",
    )

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate speech from text")
    gen_parser.add_argument("text", help="Text to synthesize")
    gen_parser.add_argument("-o", "--output", default="output.wav", help="Output file path")
    gen_parser.add_argument("-r", "--reference", help="Reference audio for voice cloning")
    gen_parser.add_argument("--model", default="kugelaudio/kugelaudio-0-open", help="Model ID")
    gen_parser.add_argument(
        "--cfg-scale", type=float, default=3.0, help="Guidance scale (1.0-10.0)"
    )
    gen_parser.add_argument(
        "--max-tokens", type=int, default=4096, help="Maximum generation tokens"
    )
    gen_parser.add_argument(
        "--low-vram",
        action="store_true",
        help="Enable low-VRAM mode (~3-4GB VRAM, slower inference, needs ~15GB RAM)",
    )
    gen_parser.add_argument(
        "--quantize",
        action="store_true",
        help="Load model in 4-bit (NF4) quantization (~4GB VRAM, ~4GB RAM). Requires: pip install bitsandbytes",
    )

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Check watermark in audio")
    verify_parser.add_argument("audio", help="Audio file to check")

    args = parser.parse_args()

    # Default to UI if no command specified
    if args.command is None:
        args.command = "ui"
        args.share = False
        args.host = "127.0.0.1"
        args.port = 7860
        args.model = "kugelaudio/kugelaudio-0-open"
        args.low_vram = False
        args.quantize = False

    if args.command == "ui":
        print("🎙️ Starting KugelAudio Web Interface...")
        print(f"   Host: {args.host}")
        print(f"   Port: {args.port}")
        print(f"   Share: {args.share}")
        if args.quantize:
            print(f"   Mode: 4-bit quantized (NF4, ~4GB VRAM, ~4GB RAM)")
        elif args.low_vram:
            print(f"   Mode: Low-VRAM (offloading to CPU, ~3-4GB VRAM, ~15GB RAM)")
        print()

        from kugelaudio_open.ui import launch_app

        launch_app(
            share=args.share,
            server_name=args.host,
            server_port=args.port,
            low_vram=args.low_vram,
            quantize=args.quantize,
        )

    elif args.command == "generate":
        import torch
        from kugelaudio_open.processors import KugelAudioProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        # Determine loading mode
        use_quantize = args.quantize and device == "cuda"
        use_low_vram = args.low_vram and device == "cuda" and not use_quantize

        print(f"🎙️ KugelAudio Speech Generation")
        print(f"   Model: {args.model}")
        print(f"   Device: {device}")
        if use_quantize:
            print(f"   Mode: 4-bit quantized (NF4, ~4GB VRAM, ~4GB RAM)")
        elif use_low_vram:
            print(f"   Mode: Low-VRAM (offloading to CPU, ~3-4GB VRAM, ~15GB RAM)")
        print(f"   Text: {args.text[:50]}..." if len(args.text) > 50 else f"   Text: {args.text}")
        if args.reference:
            print(f"   Reference: {args.reference}")
        print()

        print("Loading model...")
        if use_quantize:
            from kugelaudio_open.models import load_model_quantized

            model = load_model_quantized(args.model, device=device)
        elif use_low_vram:
            from kugelaudio_open.models import load_model_low_vram

            model = load_model_low_vram(args.model, device=device)
        else:
            from kugelaudio_open.models import KugelAudioForConditionalGenerationInference

            model = KugelAudioForConditionalGenerationInference.from_pretrained(
                args.model, torch_dtype=dtype
            ).to(device)
            model.eval()

        processor = KugelAudioProcessor.from_pretrained(args.model)

        # Process inputs — move to GPU unless using low-vram offloading (wrapper handles it)
        inputs = processor(text=args.text, voice_prompt=args.reference, return_tensors="pt")
        if not use_low_vram:
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
            }

        print("Generating speech...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                cfg_scale=args.cfg_scale,
                max_new_tokens=args.max_tokens,
            )

        audio = outputs.speech_outputs[0]

        # Save
        processor.save_audio(audio, args.output)
        print(f"✅ Audio saved to {args.output}")

    elif args.command == "verify":
        import numpy as np
        import soundfile as sf
        from kugelaudio_open.watermark import AudioWatermark

        print(f"🔍 Checking watermark in: {args.audio}")
        print()

        audio, sr = sf.read(args.audio)

        watermark = AudioWatermark()
        result = watermark.detect(audio, sample_rate=sr)

        if result.detected:
            print(f"✅ Watermark DETECTED")
            print(f"   Confidence: {result.confidence:.1%}")
            print("   This audio was generated by KugelAudio.")
        else:
            print(f"❌ No watermark detected")
            print(f"   Confidence: {result.confidence:.1%}")
            print("   This audio does not appear to be generated by KugelAudio.")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
