import argparse
from enigma.core.training import train_model
from enigma.core.inference import EnigmaEngine
from enigma.comms.api_server import create_app
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the toy model")
    parser.add_argument("--serve", action="store_true", help="Start API server")
    parser.add_argument("--run", action="store_true", help="Run a demo CLI loop")
    parser.add_argument("--gui", action="store_true", help="Start PyQt GUI")
    args = parser.parse_args()

    if args.train:
        train_model(force=False)

    if args.serve:
        app = create_app()
        app.run(host="127.0.0.1", port=5000, debug=True)

    if args.run:
        engine = EnigmaEngine()
        print("Enigma demo CLI. Type 'quit' to exit.")
        while True:
            prompt = input("You: ")
            if prompt.strip().lower() in ("quit", "exit"):
                break
            resp = engine.generate(prompt)
            print("Enigma:", resp)

    if args.gui:
        from enigma.gui.main_window import run_app
        run_app()
        
if __name__ == "__main__":
    main()
