from enigma.core.inference import EnigmaEngine

if __name__ == "__main__":
    engine = EnigmaEngine()
    print(engine.generate("Hello Enigma,", max_gen=20))
