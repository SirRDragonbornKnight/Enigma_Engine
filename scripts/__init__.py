"""
Enigma Scripts - Utility tools for advanced users

These scripts provide additional functionality beyond run.py:

- benchmark.py: Benchmark model performance
    python -m scripts.benchmark --size medium

- convert.py: Convert/resize models, export to ONNX
    python -m scripts.convert --model my_model --to onnx
    python -m scripts.convert --model my_model --grow large

- train_tokenizer.py: Train a custom BPE tokenizer
    python -m scripts.train_tokenizer --data data/*.txt --vocab-size 8000

- add_unique_qa_chunk.py: Generate unique Q&A training data
    python -m scripts.add_unique_qa_chunk

For basic usage (train, run, serve), use run.py instead:
    python run.py --train
    python run.py --run
    python run.py --serve
    python run.py --gui
"""
