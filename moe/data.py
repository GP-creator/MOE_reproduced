"""Data pipeline: TinyStories (or tiny Shakespeare fallback) -> BPE tokens -> uint16 .bin files.

Steps (all cached under ``<repo_root>/<data.cache_dir>``):
  1. Download TinyStories (``roneneldan/TinyStories``, train + validation) with HF ``datasets``
     in streaming mode. If that fails (no network, etc.) fall back to tiny Shakespeare
     (karpathy char-rnn ``input.txt``), split 90/10 train/val. The dataset actually used is
     recorded in ``meta.json``.
  2. Train a byte-level BPE (vocab 8192) with HF ``tokenizers`` on the first
     ``tokenizer_train_stories`` training stories; special token ``<|endoftext|>`` (EOS).
     Saved to ``tokenizer.json`` and reused if present. All configs share this tokenizer.
  3. Pack: stories are joined with EOS after each story, tokenized in streamed chunks
     (CHUNK_STORIES at a time) and appended to ``train.bin`` / ``val.bin`` (uint16), stopping at
     ``max_train_tokens`` / ``max_val_tokens``.

Caching policy: the cache holds the LARGEST cap ever requested. A config asking for a smaller
cap just slices the cached file (``TokenBatcher(max_tokens=...)``). If a config asks for a
larger cap than the cached one, the .bin files are rebuilt (the tokenizer is kept) -- unless
the source split was already exhausted, in which case more tokens do not exist.

Batching: ``TokenBatcher`` is stateless. The stream is cut into non-overlapping windows of
``seq_len + 1`` tokens; epoch ``e`` uses a permutation drawn from ``default_rng([seed, e])``;
batch ``step`` takes global window slots ``[step*bs, (step+1)*bs)``. So a batch depends only
on (seed, step), which makes it identical across model variants and after resume.

CLI: ``python -m moe.data --config configs/main.yaml``
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch

EOS_TOKEN = "<|endoftext|>"
CHUNK_STORIES = 10_000
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


# --------------------------------------------------------------------------- paths / meta
def _cache_dir(data_cfg: dict, repo_root: Path | str) -> Path:
    return Path(repo_root) / data_cfg.get("cache_dir", "data_cache")


def _read_meta(cache: Path) -> dict:
    meta_path = cache / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {}


def _write_meta(cache: Path, meta: dict) -> None:
    (cache / "meta.json").write_text(json.dumps(meta, indent=2))


# --------------------------------------------------------------------------- text sources
def _tinystories_stream(split: str) -> Iterator[str]:
    """Yield stories of one TinyStories split, streamed (never fully in memory)."""
    from datasets import load_dataset

    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    for row in ds:
        yield row["text"]


def _shakespeare_text(cache: Path) -> str:
    """Download (once) and return the tiny Shakespeare text."""
    raw_path = cache / "shakespeare_input.txt"
    if not raw_path.exists():
        with urllib.request.urlopen(SHAKESPEARE_URL, timeout=60) as resp:
            raw_path.write_bytes(resp.read())
    return raw_path.read_text(encoding="utf-8")


def _shakespeare_docs(cache: Path, split: str) -> Iterator[str]:
    """Shakespeare 'documents' = blank-line separated paragraphs; first 90% train, last 10% val."""
    paragraphs = [p for p in _shakespeare_text(cache).split("\n\n") if p.strip()]
    cut = int(0.9 * len(paragraphs))
    chosen = paragraphs[:cut] if split == "train" else paragraphs[cut:]
    yield from chosen


def _open_source(dataset: str, cache: Path) -> tuple[str, callable]:
    """Return (dataset_name_actually_used, split -> iterator of stories).

    Tries TinyStories first; on any failure falls back to Shakespeare.
    """
    if dataset == "tinystories":
        try:
            first = next(_tinystories_stream("validation"))  # forces a real network/cache hit
            if first:
                return "tinystories", _tinystories_stream
        except Exception as exc:  # noqa: BLE001 - any failure triggers the fallback
            print(f"[data] TinyStories unavailable ({type(exc).__name__}: {exc}); falling back to Shakespeare")
    _shakespeare_text(cache)  # make sure it can be fetched
    return "shakespeare", lambda split: _shakespeare_docs(cache, split)


# --------------------------------------------------------------------------- tokenizer
def _train_tokenizer(stories: Iterable[str], vocab_size: int, n_stories: int, path: Path) -> None:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOS_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )

    def first_n() -> Iterator[str]:
        for i, story in enumerate(stories):
            if i >= n_stories:
                break
            yield story

    tok.train_from_iterator(first_n(), trainer=trainer, length=n_stories)
    tok.save(str(path))


def load_tokenizer(data_cfg: dict, repo_root: Path | str = "."):
    """Load the shared tokenizer from ``<cache_dir>/tokenizer.json`` (run prepare_data first)."""
    from tokenizers import Tokenizer

    path = _cache_dir(data_cfg, repo_root) / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run prepare_data() first")
    return Tokenizer.from_file(str(path))


# --------------------------------------------------------------------------- packing
def _pack_split(stories: Iterator[str], tokenizer, out_path: Path, max_tokens: int) -> tuple[int, bool]:
    """Tokenize stories in chunks, append uint16 ids (+EOS per story) to out_path.

    Returns (n_tokens_written, exhausted) where exhausted means the source ended before the cap.
    """
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    n_written = 0
    exhausted = True
    chunk: list[str] = []

    def flush(texts: list[str]) -> np.ndarray:
        encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
        ids: list[int] = []
        for enc in encodings:
            ids.extend(enc.ids)
            ids.append(eos_id)
        return np.asarray(ids, dtype=np.uint16)

    with open(out_path, "wb") as f:
        for story in stories:
            chunk.append(story)
            if len(chunk) < CHUNK_STORIES:
                continue
            arr = flush(chunk)
            chunk = []
            arr = arr[: max_tokens - n_written]
            f.write(arr.tobytes())
            n_written += len(arr)
            if n_written >= max_tokens:
                exhausted = False
                break
        else:
            # source ended; write the final partial chunk
            if chunk:
                arr = flush(chunk)[: max_tokens - n_written]
                f.write(arr.tobytes())
                n_written += len(arr)
                if n_written >= max_tokens:
                    exhausted = False
    return n_written, exhausted


def prepare_data(data_cfg: dict, repo_root: Path | str = ".") -> dict:
    """Ensure tokenizer + .bin caches exist; return paths, token counts and dataset name."""
    cache = _cache_dir(data_cfg, repo_root)
    cache.mkdir(parents=True, exist_ok=True)
    tok_path, train_bin, val_bin = cache / "tokenizer.json", cache / "train.bin", cache / "val.bin"
    meta = _read_meta(cache)
    vocab_size = int(data_cfg.get("vocab_size", 8192))
    want_train = int(data_cfg["max_train_tokens"])
    want_val = int(data_cfg["max_val_tokens"])

    bins_ok = (
        meta.get("vocab_size") == vocab_size
        and train_bin.exists()
        and val_bin.exists()
        and (meta.get("train_cap", 0) >= want_train or meta.get("train_exhausted"))
        and (meta.get("val_cap", 0) >= want_val or meta.get("val_exhausted"))
    )
    if bins_ok and tok_path.exists():
        return _summary(meta, tok_path, train_bin, val_bin)

    # Need to (re)build something: figure out which dataset to use.
    if meta.get("dataset") and meta.get("vocab_size") == vocab_size:
        dataset_name = meta["dataset"]  # keep the previously chosen dataset for consistency
        _, source = _open_source(dataset_name, cache)
    else:
        dataset_name, source = _open_source(data_cfg.get("dataset", "tinystories"), cache)
    print(f"[data] dataset: {dataset_name}")

    if not tok_path.exists() or meta.get("vocab_size") != vocab_size:
        t0 = time.time()
        n_tok_stories = int(data_cfg.get("tokenizer_train_stories", 200_000))
        print(f"[data] training BPE (vocab {vocab_size}) on first {n_tok_stories} stories ...")
        _train_tokenizer(source("train"), vocab_size, n_tok_stories, tok_path)
        print(f"[data] tokenizer trained in {time.time() - t0:.1f}s")

    tokenizer = load_tokenizer(data_cfg, repo_root)
    new_cap_train = max(want_train, int(meta.get("train_cap", 0)))
    new_cap_val = max(want_val, int(meta.get("val_cap", 0)))

    t0 = time.time()
    n_train, train_exh = _pack_split(source("train"), tokenizer, train_bin, new_cap_train)
    print(f"[data] train.bin: {n_train:,} tokens in {time.time() - t0:.1f}s")
    t0 = time.time()
    n_val, val_exh = _pack_split(source("validation"), tokenizer, val_bin, new_cap_val)
    print(f"[data] val.bin: {n_val:,} tokens in {time.time() - t0:.1f}s")

    meta = {
        "dataset": dataset_name,
        "vocab_size": vocab_size,
        "tokenizer_train_stories": int(data_cfg.get("tokenizer_train_stories", 200_000)),
        "train_cap": new_cap_train,
        "val_cap": new_cap_val,
        "n_train_tokens": n_train,
        "n_val_tokens": n_val,
        "train_exhausted": train_exh,
        "val_exhausted": val_exh,
        "dtype": "uint16",
    }
    _write_meta(cache, meta)
    return _summary(meta, tok_path, train_bin, val_bin)


def _summary(meta: dict, tok_path: Path, train_bin: Path, val_bin: Path) -> dict:
    return {
        "tokenizer_path": str(tok_path),
        "train_bin": str(train_bin),
        "val_bin": str(val_bin),
        "n_train_tokens": int(meta["n_train_tokens"]),
        "n_val_tokens": int(meta["n_val_tokens"]),
        "dataset": meta["dataset"],
    }


# --------------------------------------------------------------------------- batching
class TokenBatcher:
    """Deterministic, stateless batcher over a flat token stream.

    ``source`` is either a path to a uint16 .bin file (memory-mapped) or an in-memory
    1-D numpy array (used by the synthetic batcher). ``max_tokens`` keeps only the first
    max_tokens tokens (so small configs can slice a bigger cache).
    """

    def __init__(self, bin_path, seq_len: int, batch_size: int, seed: int, max_tokens: int | None = None):
        if isinstance(bin_path, np.ndarray):
            tokens = bin_path
        else:
            tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
        if max_tokens is not None:
            tokens = tokens[:max_tokens]
        self.tokens = tokens
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.seed = seed
        self.window = seq_len + 1  # x = window[:-1], y = window[1:]
        self.n_windows = len(tokens) // self.window
        if self.n_windows < batch_size:
            raise ValueError(f"only {self.n_windows} windows of {self.window} tokens; need >= batch_size {batch_size}")
        self._perm_epoch = -1
        self._perm: np.ndarray | None = None

    def _permutation(self, epoch: int) -> np.ndarray:
        """Window-order permutation for an epoch (cached; pure function of seed, epoch)."""
        if epoch != self._perm_epoch:
            rng = np.random.default_rng([self.seed, epoch])
            self._perm = rng.permutation(self.n_windows)
            self._perm_epoch = epoch
        return self._perm

    def _to_tensors(self, window_ids: list[int], device) -> tuple[torch.Tensor, torch.Tensor]:
        rows = np.stack([self.tokens[w * self.window : (w + 1) * self.window] for w in window_ids])
        data = torch.from_numpy(rows.astype(np.int64))
        x = data[:, :-1].contiguous().to(device)
        y = data[:, 1:].contiguous().to(device)
        return x, y

    def get_train_batch(self, step: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch for ``step``: same (seed, step) -> same batch. x, y: [batch_size, seq_len] int64."""
        window_ids = []
        for slot in range(step * self.batch_size, (step + 1) * self.batch_size):
            epoch, pos = divmod(slot, self.n_windows)
            window_ids.append(int(self._permutation(epoch)[pos]))
        return self._to_tensors(window_ids, device)

    def get_eval_batches(self, n_batches: int, device) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """First n_batches*batch_size non-overlapping windows, in order (seed-independent)."""
        n_needed = n_batches * self.batch_size
        if n_needed > self.n_windows:
            raise ValueError(f"requested {n_needed} eval windows but only {self.n_windows} exist")
        batches = []
        for b in range(n_batches):
            ids = list(range(b * self.batch_size, (b + 1) * self.batch_size))
            batches.append(self._to_tensors(ids, device))
        return batches


def make_synthetic_batcher(vocab_size: int, seq_len: int, batch_size: int, seed: int, n_tokens: int = 200_000) -> TokenBatcher:
    """Random-token in-memory batcher for unit tests (no download)."""
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, vocab_size, size=n_tokens, dtype=np.int64).astype(np.uint16)
    return TokenBatcher(tokens, seq_len, batch_size, seed)


# --------------------------------------------------------------------------- CLI
def main() -> None:
    import yaml

    parser = argparse.ArgumentParser(description="Prepare tokenizer and token caches.")
    parser.add_argument("--config", required=True, help="YAML file with a data: section")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    with open(args.config) as f:
        data_cfg = yaml.safe_load(f)["data"]

    t0 = time.time()
    info = prepare_data(data_cfg, args.repo_root)
    elapsed = time.time() - t0

    print("---- summary ----")
    print(f"dataset        : {info['dataset']}")
    print(f"train tokens   : {info['n_train_tokens']:,}")
    print(f"val tokens     : {info['n_val_tokens']:,}")
    print(f"tokenizer      : {info['tokenizer_path']}")
    print(f"train_bin      : {info['train_bin']}")
    print(f"val_bin        : {info['val_bin']}")
    print(f"time taken     : {elapsed:.1f}s")
    tok = load_tokenizer(data_cfg, args.repo_root)
    sample = np.memmap(info["train_bin"], dtype=np.uint16, mode="r")[:200]
    print("sample (first 200 train tokens):")
    print(tok.decode([int(t) for t in sample], skip_special_tokens=False))


if __name__ == "__main__":
    main()
