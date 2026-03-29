import os
from pathlib import Path
from urllib.request import Request, urlopen
from html.parser import HTMLParser
import ssl
import sys


class _VisibleTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0  # >0 when inside script/style

    def handle_starttag(self, tag, attrs):
        t = (tag or "").lower()
        if t in ("script", "style"):
            self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return
        if t in ("p", "br", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "code"):
            self._parts.append("\n")

    def handle_endtag(self, tag):
        t = (tag or "").lower()
        if t in ("script", "style"):
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if self._skip_depth > 0:
            return
        if t in ("p", "div", "li", "pre"):
            self._parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth > 0:
            return
        if not data:
            return
        self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def _collapse_consecutive_empty(lines: list[str]) -> list[str]:
    out: list[str] = []
    prev_empty = False
    for ln in lines:
        s = ln.strip()
        if s == "":
            if prev_empty:
                continue
            out.append("")
            prev_empty = True
        else:
            out.append(s)
            prev_empty = False
    return out


def _filter_short_lines(lines: list[str], min_len: int = 30) -> list[str]:
    out: list[str] = []
    for ln in lines:
        if ln == "":
            out.append("")
            continue
        if len(ln) >= min_len:
            out.append(ln)
    return out


def _clean_text_block(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    lines = [ln.strip() for ln in lines]
    lines = _filter_short_lines(lines, min_len=30)
    lines = _collapse_consecutive_empty(lines)
    return "\n".join(lines).strip() + "\n"


def _download(url: str, timeout_s: int = 30) -> bytes:
    req = Request(
        url,
        headers={
            "User-Agent": "VIERNES-corpus-downloader/1.0 (local; urllib.request)",
            "Accept": "text/html, text/plain, */*",
        },
        method="GET",
    )
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            return resp.read()
    except Exception as e:
        # Some macOS Python installs lack a configured CA bundle.
        # Retry once with an unverified SSL context to keep the workflow unblocked.
        if "CERTIFICATE_VERIFY_FAILED" not in str(e):
            raise
        ctx = ssl._create_unverified_context()
        with urlopen(req, timeout=timeout_s, context=ctx) as resp:
            return resp.read()


def _decode_bytes(data: bytes) -> str:
    # Conservative decoding: try utf-8 first, then latin-1.
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace")


def _extract_text_from_html(html: str) -> str:
    parser = _VisibleTextExtractor()
    parser.feed(html)
    parser.close()
    return parser.text()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> int:
    project_root = Path(__file__).resolve().parents[3]
    base = project_root / "viernes" / "data" / "corpus_raw"

    targets: list[tuple[str, Path, bool]] = [
        ("https://docs.python.org/3/tutorial/introduction.html", base / "tecnico" / "python_intro.txt", False),
        ("https://docs.python.org/3/tutorial/controlflow.html", base / "tecnico" / "python_controlflow.txt", False),
        ("https://docs.python.org/3/tutorial/classes.html", base / "tecnico" / "python_classes.txt", False),
        ("https://docs.python.org/3/tutorial/errors.html", base / "tecnico" / "python_errors.txt", False),
        ("https://numpy.org/doc/stable/user/quickstart.html", base / "tecnico" / "numpy_quickstart.txt", False),
        ("https://git-scm.com/book/en/v2/Getting-Started-About-Version-Control", base / "herramientas" / "git_intro.txt", False),
        ("https://git-scm.com/book/en/v2/Git-Basics-Getting-a-Git-Repository", base / "herramientas" / "git_basics.txt", False),
        ("https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell", base / "herramientas" / "git_branches.txt", False),
        ("https://raw.githubusercontent.com/tomasvanpez2/FRIDAY/main/README.md", base / "proyecto" / "readme.txt", True),
        # --- tecnico (Python / NumPy / PyTorch) ---
        (
            "https://raw.githubusercontent.com/python/cpython/main/Doc/tutorial/datastructures.rst",
            base / "tecnico" / "python_datastructures.txt",
            False,
        ),
        ("https://raw.githubusercontent.com/python/cpython/main/Doc/tutorial/modules.rst", base / "tecnico" / "python_modules.txt", False),
        (
            "https://raw.githubusercontent.com/python/cpython/main/Doc/tutorial/inputoutput.rst",
            base / "tecnico" / "python_io.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/python/cpython/main/Doc/howto/threading-model.rst",
            base / "tecnico" / "python_threading.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/python/cpython/main/Doc/library/threading.rst",
            base / "tecnico" / "python_threading_lib.txt",
            False,
        ),
        ("https://raw.githubusercontent.com/python/cpython/main/Doc/library/queue.rst", base / "tecnico" / "python_queue.txt", False),
        (
            "https://raw.githubusercontent.com/python/cpython/main/Doc/library/multiprocessing.rst",
            base / "tecnico" / "python_multiprocessing.txt",
            False,
        ),
        ("https://raw.githubusercontent.com/python/cpython/main/Doc/library/asyncio.rst", base / "tecnico" / "python_asyncio.txt", False),
        (
            "https://raw.githubusercontent.com/numpy/numpy/main/doc/source/user/basics.broadcasting.rst",
            base / "tecnico" / "numpy_broadcasting.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/numpy/numpy/main/doc/source/user/basics.types.rst",
            base / "tecnico" / "numpy_types.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/numpy/numpy/main/doc/source/reference/arrays.ndarray.rst",
            base / "tecnico" / "numpy_ndarray.txt",
            False,
        ),
        ("https://raw.githubusercontent.com/pytorch/pytorch/main/docs/source/nn.rst", base / "tecnico" / "pytorch_nn.txt", False),
        (
            "https://raw.githubusercontent.com/pytorch/pytorch/main/docs/source/optim.rst",
            base / "tecnico" / "pytorch_optim.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/pytorch/pytorch/main/docs/source/tensor_attributes.rst",
            base / "tecnico" / "pytorch_tensors.txt",
            False,
        ),
        # --- herramientas (Git docs) ---
        (
            "https://raw.githubusercontent.com/git/git/master/Documentation/gitglossary.txt",
            base / "herramientas" / "git_glossary.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/git/git/master/Documentation/gitcore-tutorial.txt",
            base / "herramientas" / "git_core_tutorial.txt",
            False,
        ),
        ("https://raw.githubusercontent.com/git/git/master/Documentation/git-log.txt", base / "herramientas" / "git_log.txt", False),
        ("https://raw.githubusercontent.com/git/git/master/Documentation/git-diff.txt", base / "herramientas" / "git_diff.txt", False),
        (
            "https://raw.githubusercontent.com/git/git/master/Documentation/git-commit.txt",
            base / "herramientas" / "git_commit.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/git/git/master/Documentation/git-branch.txt",
            base / "herramientas" / "git_branch.txt",
            False,
        ),
        ("https://raw.githubusercontent.com/git/git/master/Documentation/git-stash.txt", base / "herramientas" / "git_stash.txt", False),
        # --- tecnico (machine learning) ---
        (
            "https://raw.githubusercontent.com/karpathy/nn-zero-to-hero/master/lectures/makemore/makemore_part1_bigrams.ipynb",
            base / "tecnico" / "makemore_bigrams.txt",
            False,
        ),
        ("https://raw.githubusercontent.com/karpathy/nanoGPT/master/model.py", base / "tecnico" / "nanogpt_model.txt", False),
        ("https://raw.githubusercontent.com/karpathy/nanoGPT/master/train.py", base / "tecnico" / "nanogpt_train.txt", False),
        ("https://raw.githubusercontent.com/karpathy/nanoGPT/master/README.md", base / "tecnico" / "nanogpt_readme.txt", False),
        (
            "https://raw.githubusercontent.com/DLR-RM/stable-baselines3/master/stable_baselines3/ppo/ppo.py",
            base / "tecnico" / "sb3_ppo_source.txt",
            False,
        ),
        ("https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/main/gymnasium/core.py", base / "tecnico" / "gymnasium_core.txt", False),
        (
            "https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/main/gymnasium/spaces/discrete.py",
            base / "tecnico" / "gymnasium_discrete.txt",
            False,
        ),
        ("https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/main/gymnasium/spaces/box.py", base / "tecnico" / "gymnasium_box.txt", False),
        # --- proyecto ---
        (
            "https://raw.githubusercontent.com/DLR-RM/stable-baselines3/master/docs/guide/custom_env.rst",
            base / "proyecto" / "sb3_custom_env.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/DLR-RM/stable-baselines3/master/docs/guide/rl_tips.rst",
            base / "proyecto" / "sb3_rl_tips.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/DLR-RM/stable-baselines3/master/docs/guide/callbacks.rst",
            base / "proyecto" / "sb3_callbacks.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/google/mediapipe/master/docs/solutions/hands.md",
            base / "proyecto" / "mediapipe_hands.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/google/mediapipe/master/docs/solutions/face_detection.md",
            base / "proyecto" / "mediapipe_face.txt",
            False,
        ),
        # --- tecnico (Python avanzado) ---
        ("https://raw.githubusercontent.com/python/cpython/main/Doc/tutorial/appetite.rst", base / "tecnico" / "python_appetite.txt", False),
        (
            "https://raw.githubusercontent.com/python/cpython/main/Doc/tutorial/interpreter.rst",
            base / "tecnico" / "python_interpreter.txt",
            False,
        ),
        ("https://raw.githubusercontent.com/python/cpython/main/Doc/tutorial/numbers.rst", base / "tecnico" / "python_numbers.txt", False),
        (
            "https://raw.githubusercontent.com/python/cpython/main/Doc/howto/descriptor.rst",
            base / "tecnico" / "python_descriptors.txt",
            False,
        ),
        ("https://raw.githubusercontent.com/python/cpython/main/Doc/howto/enum.rst", base / "tecnico" / "python_enum.txt", False),
        ("https://raw.githubusercontent.com/python/cpython/main/Doc/howto/argparse.rst", base / "tecnico" / "python_argparse.txt", False),
        ("https://raw.githubusercontent.com/python/cpython/main/Doc/howto/regex.rst", base / "tecnico" / "python_regex.txt", False),
        ("https://raw.githubusercontent.com/python/cpython/main/Doc/howto/sorting.rst", base / "tecnico" / "python_sorting.txt", False),
        (
            "https://raw.githubusercontent.com/python/cpython/main/Doc/library/collections.rst",
            base / "tecnico" / "python_collections.txt",
            False,
        ),
        ("https://raw.githubusercontent.com/python/cpython/main/Doc/library/itertools.rst", base / "tecnico" / "python_itertools.txt", False),
        ("https://raw.githubusercontent.com/python/cpython/main/Doc/library/functools.rst", base / "tecnico" / "python_functools.txt", False),
        ("https://raw.githubusercontent.com/python/cpython/main/Doc/library/pathlib.rst", base / "tecnico" / "python_pathlib.txt", False),
        ("https://raw.githubusercontent.com/python/cpython/main/Doc/library/json.rst", base / "tecnico" / "python_json.txt", False),
        ("https://raw.githubusercontent.com/python/cpython/main/Doc/library/re.rst", base / "tecnico" / "python_re.txt", False),
        ("https://raw.githubusercontent.com/python/cpython/main/Doc/library/datetime.rst", base / "tecnico" / "python_datetime.txt", False),
        ("https://raw.githubusercontent.com/python/cpython/main/Doc/library/math.rst", base / "tecnico" / "python_math.txt", False),
        ("https://raw.githubusercontent.com/python/cpython/main/Doc/library/os.rst", base / "tecnico" / "python_os.txt", False),
        ("https://raw.githubusercontent.com/python/cpython/main/Doc/library/sys.rst", base / "tecnico" / "python_sys.txt", False),
        (
            "https://raw.githubusercontent.com/python/cpython/main/Doc/library/subprocess.rst",
            base / "tecnico" / "python_subprocess.txt",
            False,
        ),
        ("https://raw.githubusercontent.com/python/cpython/main/Doc/library/logging.rst", base / "tecnico" / "python_logging.txt", False),
        ("https://raw.githubusercontent.com/python/cpython/main/Doc/library/unittest.rst", base / "tecnico" / "python_unittest.txt", False),
        (
            "https://raw.githubusercontent.com/python/cpython/main/Doc/library/dataclasses.rst",
            base / "tecnico" / "python_dataclasses.txt",
            False,
        ),
        ("https://raw.githubusercontent.com/python/cpython/main/Doc/library/typing.rst", base / "tecnico" / "python_typing.txt", False),
        ("https://raw.githubusercontent.com/python/cpython/main/Doc/library/abc.rst", base / "tecnico" / "python_abc.txt", False),
        (
            "https://raw.githubusercontent.com/python/cpython/main/Doc/library/contextlib.rst",
            base / "tecnico" / "python_contextlib.txt",
            False,
        ),
        # --- tecnico (NumPy profundo) ---
        (
            "https://raw.githubusercontent.com/numpy/numpy/main/doc/source/user/absolute_beginners.rst",
            base / "tecnico" / "numpy_beginners.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/numpy/numpy/main/doc/source/user/basics.creation.rst",
            base / "tecnico" / "numpy_creation.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/numpy/numpy/main/doc/source/user/basics.ufuncs.rst",
            base / "tecnico" / "numpy_ufuncs.txt",
            False,
        ),
        ("https://raw.githubusercontent.com/numpy/numpy/main/doc/source/user/basics.rec.rst", base / "tecnico" / "numpy_rec.txt", False),
        (
            "https://raw.githubusercontent.com/numpy/numpy/main/doc/source/reference/routines.linalg.rst",
            base / "tecnico" / "numpy_linalg.txt",
            False,
        ),
        ("https://raw.githubusercontent.com/numpy/numpy/main/doc/source/reference/routines.fft.rst", base / "tecnico" / "numpy_fft.txt", False),
        (
            "https://raw.githubusercontent.com/numpy/numpy/main/doc/source/reference/routines.statistics.rst",
            base / "tecnico" / "numpy_statistics.txt",
            False,
        ),
        # --- tecnico (PyTorch profundo) ---
        (
            "https://raw.githubusercontent.com/pytorch/pytorch/main/docs/source/autograd.rst",
            base / "tecnico" / "pytorch_autograd.rst",
            False,
        ),
        ("https://raw.githubusercontent.com/pytorch/pytorch/main/docs/source/cuda.rst", base / "tecnico" / "pytorch_cuda.txt", False),
        (
            "https://raw.githubusercontent.com/pytorch/pytorch/main/docs/source/notes/cpu_threading_torchscript_inference.rst",
            base / "tecnico" / "pytorch_cpu_threading.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/pytorch/pytorch/main/docs/source/notes/randomness.rst",
            base / "tecnico" / "pytorch_randomness.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/pytorch/pytorch/main/docs/source/notes/serialization.rst",
            base / "tecnico" / "pytorch_serialization.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/pytorch/pytorch/main/docs/source/quantization.rst",
            base / "tecnico" / "pytorch_quantization.txt",
            False,
        ),
        # --- tecnico (ML/RL: más repos) ---
        (
            "https://raw.githubusercontent.com/karpathy/nanoGPT/master/config/train_gpt2.py",
            base / "tecnico" / "nanogpt_config.txt",
            False,
        ),
        ("https://raw.githubusercontent.com/karpathy/minGPT/master/mingpt/model.py", base / "tecnico" / "mingpt_model.txt", False),
        (
            "https://raw.githubusercontent.com/karpathy/minGPT/master/mingpt/trainer.py",
            base / "tecnico" / "mingpt_trainer.txt",
            False,
        ),
        ("https://raw.githubusercontent.com/karpathy/minGPT/master/README.md", base / "tecnico" / "mingpt_readme.txt", False),
        (
            "https://raw.githubusercontent.com/karpathy/micrograd/master/micrograd/engine.py",
            base / "tecnico" / "micrograd_engine.txt",
            False,
        ),
        ("https://raw.githubusercontent.com/karpathy/micrograd/master/README.md", base / "tecnico" / "micrograd_readme.txt", False),
        ("https://raw.githubusercontent.com/karpathy/char-rnn/master/model.lua", base / "tecnico" / "char_rnn_model.txt", False),
        # --- tecnico (Transformers) ---
        (
            "https://raw.githubusercontent.com/huggingface/transformers/main/README.md",
            base / "tecnico" / "transformers_readme.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/huggingface/transformers/main/docs/source/en/quicktour.md",
            base / "tecnico" / "transformers_quicktour.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/huggingface/transformers/main/docs/source/en/training.md",
            base / "tecnico" / "transformers_training.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/huggingface/transformers/main/docs/source/en/performance.md",
            base / "tecnico" / "transformers_performance.txt",
            False,
        ),
        # --- tecnico (Visión por computador) ---
        ("https://raw.githubusercontent.com/opencv/opencv/master/modules/core/README.md", base / "tecnico" / "opencv_core.txt", False),
        (
            "https://raw.githubusercontent.com/opencv/opencv/master/doc/tutorials/introduction/table_of_content_introduction/table_of_content_introduction.markdown",
            base / "tecnico" / "opencv_intro.txt",
            False,
        ),
        # --- tecnico (Sentence Transformers) ---
        (
            "https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/README.md",
            base / "tecnico" / "sentence_transformers_readme.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/quickstart.md",
            base / "tecnico" / "sentence_transformers_quickstart.txt",
            False,
        ),
        # --- herramientas (Git profundo) ---
        ("https://raw.githubusercontent.com/git/git/master/Documentation/git-rebase.txt", base / "herramientas" / "git_rebase.txt", False),
        ("https://raw.githubusercontent.com/git/git/master/Documentation/git-merge.txt", base / "herramientas" / "git_merge.txt", False),
        ("https://raw.githubusercontent.com/git/git/master/Documentation/git-reset.txt", base / "herramientas" / "git_reset.txt", False),
        (
            "https://raw.githubusercontent.com/git/git/master/Documentation/git-checkout.txt",
            base / "herramientas" / "git_checkout.txt",
            False,
        ),
        ("https://raw.githubusercontent.com/git/git/master/Documentation/git-fetch.txt", base / "herramientas" / "git_fetch.txt", False),
        ("https://raw.githubusercontent.com/git/git/master/Documentation/git-push.txt", base / "herramientas" / "git_push.txt", False),
        ("https://raw.githubusercontent.com/git/git/master/Documentation/git-pull.txt", base / "herramientas" / "git_pull.txt", False),
        ("https://raw.githubusercontent.com/git/git/master/Documentation/git-remote.txt", base / "herramientas" / "git_remote.txt", False),
        ("https://raw.githubusercontent.com/git/git/master/Documentation/git-tag.txt", base / "herramientas" / "git_tag.txt", False),
        (
            "https://raw.githubusercontent.com/git/git/master/Documentation/git-config.txt",
            base / "herramientas" / "git_config.txt",
            False,
        ),
        ("https://raw.githubusercontent.com/git/git/master/Documentation/gitignore.txt", base / "herramientas" / "gitignore.txt", False),
        (
            "https://raw.githubusercontent.com/git/git/master/Documentation/git-bisect.txt",
            base / "herramientas" / "git_bisect.txt",
            False,
        ),
        # --- herramientas (Packaging Python) ---
        ("https://raw.githubusercontent.com/pypa/pip/main/docs/html/user_guide.rst", base / "herramientas" / "pip_userguide.txt", False),
        (
            "https://raw.githubusercontent.com/pypa/virtualenv/main/docs/user_guide.rst",
            base / "herramientas" / "virtualenv_guide.txt",
            False,
        ),
        # --- proyecto (SB3 completo) ---
        ("https://raw.githubusercontent.com/DLR-RM/stable-baselines3/master/README.md", base / "proyecto" / "sb3_readme.txt", False),
        (
            "https://raw.githubusercontent.com/DLR-RM/stable-baselines3/master/docs/guide/vec_envs.rst",
            base / "proyecto" / "sb3_vec_envs.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/DLR-RM/stable-baselines3/master/docs/guide/tensorboard.rst",
            base / "proyecto" / "sb3_tensorboard.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/DLR-RM/stable-baselines3/master/stable_baselines3/common/policies.py",
            base / "proyecto" / "sb3_policies_source.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/DLR-RM/stable-baselines3/master/stable_baselines3/common/buffers.py",
            base / "proyecto" / "sb3_buffers_source.txt",
            False,
        ),
        # --- proyecto (Gymnasium completo) ---
        ("https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/main/README.md", base / "proyecto" / "gymnasium_readme.txt", False),
        (
            "https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/main/docs/introduction/basic_usage.md",
            base / "proyecto" / "gymnasium_basic.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/main/docs/tutorials/gymnasium_basics/handling_time_limits.md",
            base / "proyecto" / "gymnasium_timelimits.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/main/gymnasium/wrappers/time_limit.py",
            base / "proyecto" / "gymnasium_timelimit_wrapper.txt",
            False,
        ),
        # --- proyecto (LanceDB) ---
        ("https://raw.githubusercontent.com/lancedb/lancedb/main/README.md", base / "proyecto" / "lancedb_readme.txt", False),
        ("https://raw.githubusercontent.com/lancedb/lancedb/main/docs/src/basic.md", base / "proyecto" / "lancedb_basic.txt", False),
        (
            "https://raw.githubusercontent.com/lancedb/lancedb/main/docs/src/python/python.md",
            base / "proyecto" / "lancedb_python.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/lancedb/lancedb/main/docs/src/embeddings/embedding_functions.md",
            base / "proyecto" / "lancedb_embeddings.txt",
            False,
        ),
        # --- proyecto (MediaPipe - google-ai-edge) ---
        (
            "https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/docs/solutions/hands.md",
            base / "proyecto" / "mediapipe_hands.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/docs/solutions/face_detection.md",
            base / "proyecto" / "mediapipe_face.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/docs/solutions/holistic.md",
            base / "proyecto" / "mediapipe_holistic.txt",
            False,
        ),
        # --- conversacion (Spinning Up en RL) ---
        (
            "https://raw.githubusercontent.com/openai/spinningup/master/docs/spinningup/rl_intro.rst",
            base / "conversacion" / "rl_intro.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/openai/spinningup/master/docs/spinningup/rl_intro2.rst",
            base / "conversacion" / "rl_intro2.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/openai/spinningup/master/docs/spinningup/rl_intro3.rst",
            base / "conversacion" / "rl_intro3.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/openai/spinningup/master/docs/spinningup/algorithms.rst",
            base / "conversacion" / "rl_algorithms.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/openai/spinningup/master/docs/spinningup/bench.rst",
            base / "conversacion" / "rl_benchmarks.txt",
            False,
        ),
        # --- tecnico (Matemáticas / estadística / señales) ---
        ("https://raw.githubusercontent.com/python/cpython/main/Doc/library/math.rst", base / "tecnico" / "math_stdlib.txt", False),
        (
            "https://raw.githubusercontent.com/python/cpython/main/Doc/library/statistics.rst",
            base / "tecnico" / "statistics_stdlib.txt",
            False,
        ),
        ("https://raw.githubusercontent.com/python/cpython/main/Doc/library/random.rst", base / "tecnico" / "random_stdlib.txt", False),
        (
            "https://raw.githubusercontent.com/numpy/numpy/main/doc/source/reference/routines.statistics.rst",
            base / "tecnico" / "numpy_stats.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/numpy/numpy/main/doc/source/reference/routines.math.rst",
            base / "tecnico" / "numpy_math.txt",
            False,
        ),
        ("https://raw.githubusercontent.com/scipy/scipy/main/doc/source/tutorial/stats.rst", base / "tecnico" / "scipy_stats.txt", False),
        (
            "https://raw.githubusercontent.com/scipy/scipy/main/doc/source/tutorial/optimize.rst",
            base / "tecnico" / "scipy_optimize.txt",
            False,
        ),
        ("https://raw.githubusercontent.com/scipy/scipy/main/doc/source/tutorial/signal.rst", base / "tecnico" / "scipy_signal.txt", False),
        ("https://raw.githubusercontent.com/scipy/scipy/main/doc/source/tutorial/fft.rst", base / "tecnico" / "scipy_fft.txt", False),
        ("https://raw.githubusercontent.com/librosa/librosa/main/docs/tutorial.rst", base / "tecnico" / "librosa_tutorial.txt", False),
        ("https://raw.githubusercontent.com/librosa/librosa/main/README.md", base / "tecnico" / "librosa_readme.txt", False),
        (
            "https://raw.githubusercontent.com/jameslyons/python_speech_features/master/README.md",
            base / "tecnico" / "speech_features_readme.txt",
            False,
        ),
        # --- tecnico (Redes / serialización / seguridad / tiempo) ---
        ("https://raw.githubusercontent.com/python/cpython/main/Doc/library/socket.rst", base / "tecnico" / "python_socket.txt", False),
        (
            "https://raw.githubusercontent.com/python/cpython/main/Doc/library/http.client.rst",
            base / "tecnico" / "python_http.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/python/cpython/main/Doc/library/urllib.request.rst",
            base / "tecnico" / "python_urllib.txt",
            False,
        ),
        ("https://raw.githubusercontent.com/python/cpython/main/Doc/library/ssl.rst", base / "tecnico" / "python_ssl.txt", False),
        ("https://raw.githubusercontent.com/python/cpython/main/Doc/library/pickle.rst", base / "tecnico" / "python_pickle.txt", False),
        ("https://raw.githubusercontent.com/python/cpython/main/Doc/library/struct.rst", base / "tecnico" / "python_struct.txt", False),
        ("https://raw.githubusercontent.com/python/cpython/main/Doc/library/csv.rst", base / "tecnico" / "python_csv.txt", False),
        ("https://raw.githubusercontent.com/python/cpython/main/Doc/library/zipfile.rst", base / "tecnico" / "python_zipfile.txt", False),
        ("https://raw.githubusercontent.com/python/cpython/main/Doc/library/sqlite3.rst", base / "tecnico" / "python_sqlite3.txt", False),
        ("https://raw.githubusercontent.com/python/cpython/main/Doc/library/hashlib.rst", base / "tecnico" / "python_hashlib.txt", False),
        ("https://raw.githubusercontent.com/python/cpython/main/Doc/library/secrets.rst", base / "tecnico" / "python_secrets.txt", False),
        ("https://raw.githubusercontent.com/python/cpython/main/Doc/library/hmac.rst", base / "tecnico" / "python_hmac.txt", False),
        ("https://raw.githubusercontent.com/python/cpython/main/Doc/library/time.rst", base / "tecnico" / "python_time.txt", False),
        ("https://raw.githubusercontent.com/python/cpython/main/Doc/library/sched.rst", base / "tecnico" / "python_sched.txt", False),
        ("https://raw.githubusercontent.com/python/cpython/main/Doc/library/timeit.rst", base / "tecnico" / "python_timeit.txt", False),
        # --- tecnico (Arquitectura de computadores) ---
        (
            "https://raw.githubusercontent.com/torvalds/linux/master/Documentation/process/howto.rst",
            base / "tecnico" / "linux_howto.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/torvalds/linux/master/Documentation/admin-guide/README.rst",
            base / "tecnico" / "linux_readme.txt",
            False,
        ),
        # --- herramientas (VSCode / terminal / docker / make) ---
        (
            "https://raw.githubusercontent.com/microsoft/vscode-docs/main/docs/getstarted/tips-and-tricks.md",
            base / "herramientas" / "vscode_tips.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/microsoft/vscode-docs/main/docs/editor/debugging.md",
            base / "herramientas" / "vscode_debugging.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/microsoft/vscode-docs/main/docs/python/debugging.md",
            base / "herramientas" / "vscode_python_debug.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/microsoft/vscode-docs/main/docs/python/environments.md",
            base / "herramientas" / "vscode_environments.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/microsoft/vscode-docs/main/docs/editor/integrated-terminal.md",
            base / "herramientas" / "vscode_terminal.txt",
            False,
        ),
        ("https://raw.githubusercontent.com/git/git/master/Documentation/git-worktree.txt", base / "herramientas" / "git_worktree.txt", False),
        (
            "https://raw.githubusercontent.com/git/git/master/Documentation/git-submodule.txt",
            base / "herramientas" / "git_submodule.txt",
            False,
        ),
        ("https://raw.githubusercontent.com/git/git/master/Documentation/gitattributes.txt", base / "herramientas" / "git_attributes.txt", False),
        ("https://raw.githubusercontent.com/git/git/master/Documentation/githooks.txt", base / "herramientas" / "git_hooks.txt", False),
        ("https://raw.githubusercontent.com/docker/docs/main/get-started/overview.md", base / "herramientas" / "docker_overview.txt", False),
        ("https://raw.githubusercontent.com/docker/docs/main/get-started/02_our_app.md", base / "herramientas" / "docker_app.txt", False),
        ("https://raw.githubusercontent.com/docker/docs/main/get-started/04_sharing_app.md", base / "herramientas" / "docker_sharing.txt", False),
        ("https://raw.githubusercontent.com/python/cpython/main/Makefile.pre.in", base / "herramientas" / "python_makefile.txt", False),
        # --- conversacion (Wikipedia simple + ejercicios RL + patrones) ---
        (
            "https://raw.githubusercontent.com/Refefer/simple-wiki-dataset/master/README.md",
            base / "conversacion" / "simplewiki_readme.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/openai/spinningup/master/docs/spinningup/exercise1_1.rst",
            base / "conversacion" / "rl_exercise1.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/openai/spinningup/master/docs/spinningup/exercise1_1_soln.rst",
            base / "conversacion" / "rl_exercise1_soln.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/openai/spinningup/master/docs/spinningup/exercise2_1.rst",
            base / "conversacion" / "rl_exercise2.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/openai/spinningup/master/docs/spinningup/exercise2_2.rst",
            base / "conversacion" / "rl_exercise2_2.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/openai/spinningup/master/docs/spinningup/exercise3_1.rst",
            base / "conversacion" / "rl_exercise3.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/30-seconds/30-seconds-of-python/master/README.md",
            base / "conversacion" / "python_snippets.txt",
            False,
        ),
        ("https://raw.githubusercontent.com/TheAlgorithms/Python/master/README.md", base / "conversacion" / "algorithms_readme.txt", False),
        (
            "https://raw.githubusercontent.com/TheAlgorithms/Python/master/CONTRIBUTING.md",
            base / "conversacion" / "algorithms_contributing.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/donnemartin/system-design-primer/master/README.md",
            base / "conversacion" / "system_design.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/kamranahmedse/developer-roadmap/master/readme.md",
            base / "conversacion" / "dev_roadmap.txt",
            False,
        ),
        # --- proyecto (ciencia/física + electrónica + embebidos + tokenización) ---
        ("https://raw.githubusercontent.com/QuantumBFS/Yao.jl/master/docs/src/man/basics.md", base / "proyecto" / "quantum_basics.txt", False),
        ("https://raw.githubusercontent.com/sympy/sympy/master/README.rst", base / "proyecto" / "sympy_readme.txt", False),
        ("https://raw.githubusercontent.com/sympy/sympy/master/doc/src/tutorial/intro.rst", base / "proyecto" / "sympy_intro.txt", False),
        (
            "https://raw.githubusercontent.com/sympy/sympy/master/doc/src/tutorial/calculus.rst",
            base / "proyecto" / "sympy_calculus.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/sympy/sympy/master/doc/src/tutorial/matrices.rst",
            base / "proyecto" / "sympy_matrices.txt",
            False,
        ),
        ("https://raw.githubusercontent.com/micropython/micropython/master/README.md", base / "proyecto" / "micropython_readme.txt", False),
        ("https://raw.githubusercontent.com/micropython/micropython/master/docs/reference/repl.rst", base / "proyecto" / "micropython_repl.txt", False),
        (
            "https://raw.githubusercontent.com/micropython/micropython/master/docs/reference/isr_rules.rst",
            base / "proyecto" / "micropython_isr.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/micropython/micropython/master/docs/library/machine.rst",
            base / "proyecto" / "micropython_machine.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/micropython/micropython/master/docs/library/machine.Pin.rst",
            base / "proyecto" / "micropython_pin.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/micropython/micropython/master/docs/library/machine.I2C.rst",
            base / "proyecto" / "micropython_i2c.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/micropython/micropython/master/docs/library/machine.SPI.rst",
            base / "proyecto" / "micropython_spi.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/micropython/micropython/master/docs/library/machine.ADC.rst",
            base / "proyecto" / "micropython_adc.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/micropython/micropython/master/docs/library/machine.PWM.rst",
            base / "proyecto" / "micropython_pwm.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/micropython/micropython/master/docs/library/machine.Timer.rst",
            base / "proyecto" / "micropython_timer.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/micropython/micropython/master/docs/reference/constrained.rst",
            base / "proyecto" / "micropython_constrained.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/micropython/micropython/master/docs/reference/speed_python.rst",
            base / "proyecto" / "micropython_speed.txt",
            False,
        ),
        (
            "https://raw.githubusercontent.com/micropython/micropython/master/docs/reference/pyboard.rst",
            base / "proyecto" / "micropython_pyboard.txt",
            False,
        ),
        ("https://raw.githubusercontent.com/karpathy/nn-zero-to-hero/master/README.md", base / "proyecto" / "nn_zero_hero_readme.txt", False),
        ("https://raw.githubusercontent.com/karpathy/minbpe/master/README.md", base / "proyecto" / "minbpe_readme.txt", False),
        ("https://raw.githubusercontent.com/karpathy/minbpe/master/minbpe/base.py", base / "proyecto" / "minbpe_base.txt", False),
        ("https://raw.githubusercontent.com/karpathy/minbpe/master/minbpe/regex.py", base / "proyecto" / "minbpe_regex.txt", False),
        ("https://raw.githubusercontent.com/karpathy/minbpe/master/minbpe/basic.py", base / "proyecto" / "minbpe_basic.txt", False),
        ("https://raw.githubusercontent.com/huggingface/tokenizers/main/README.md", base / "proyecto" / "tokenizers_readme.txt", False),
        (
            "https://raw.githubusercontent.com/huggingface/tokenizers/main/bindings/python/README.md",
            base / "proyecto" / "tokenizers_python.txt",
            False,
        ),
    ]

    downloaded: list[Path] = []

    for url, out_path, _allow_fail in targets:
        try:
            raw = _download(url)
            text = _decode_bytes(raw)

            if url.lower().endswith(".md") or "raw.githubusercontent.com" in url.lower():
                cleaned = _clean_text_block(text)
            else:
                extracted = _extract_text_from_html(text)
                cleaned = _clean_text_block(extracted)

            _write_text(out_path, cleaned)
            downloaded.append(out_path)
        except Exception as e:
            print(f"SKIP: {url}")
            continue

    print("=== VIERNES corpus download stats ===")
    for p in downloaded:
        size_kb = p.stat().st_size / 1024 if p.exists() else 0.0
        print(f"- {p.as_posix()}  ({size_kb:.1f} KB)")
    print(f"Total de archivos descargados: {len(downloaded)}")

    # Run prepare_corpus.py automatically
    prepare_path = project_root / "viernes" / "data" / "scripts" / "prepare_corpus.py"
    cmd = f'python3 "{prepare_path.as_posix()}"'
    print("\n=== Ejecutando prepare_corpus.py ===")
    sys.stdout.flush()
    return os.system(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

