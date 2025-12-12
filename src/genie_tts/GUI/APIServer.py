from genie_tts import start_server
import argparse

parser = argparse.ArgumentParser(description="Genie TTS FastAPI Server")
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)
args, _ = parser.parse_known_args()
start_server(host=args.host, port=args.port, workers=1)
