# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from huggingface/transformers: https://github.com/huggingface/transformers/blob/21a2d900eceeded7be9edc445b56877b95eda4ca/setup.py


import re
import shutil
from pathlib import Path

from setuptools import find_packages, setup


# Remove stale rea_rl.egg-info directory to avoid https://github.com/pypa/pip/issues/5466
stale_egg_info = Path(__file__).parent / "rea_rl.egg-info"
if stale_egg_info.exists():
    print(
        (
            "Warning: {} exists.\n\n"
            "If you recently updated rea_rl, this is expected,\n"
            "but it may prevent rea_rl from installing in editable mode.\n\n"
            "This directory is automatically generated by Python's packaging tools.\n"
            "I will remove it now.\n\n"
        ).format(stale_egg_info)
    )
    shutil.rmtree(stale_egg_info)


# IMPORTANT: all dependencies should be listed here with their version requirements, if any.
#   * If a dependency is fast-moving (e.g. trl), pin to the exact version
_deps = [
    "accelerate==1.4.0",
    "bitsandbytes>=0.43.0",
    "datasets>=3.2.0",
    "deepspeed==0.15.4",
    "distilabel[vllm,ray,openai]>=1.5.2",
    "e2b-code-interpreter>=1.0.5",
    "einops>=0.8.0",
    "flake8>=6.0.0",
    "hf_transfer>=0.1.4",
    "huggingface-hub[cli]>=0.19.2,<1.0",
    "isort>=5.12.0",
    "liger_kernel==0.5.3",
    "packaging>=23.0",
    "parameterized>=0.9.0",
    "peft>=0.14.0",
    "pytest",
    "python-dotenv",
    "ruff>=0.9.0",
    "safetensors>=0.3.3",
    "sentencepiece>=0.1.99",
    "torch==2.5.1",
    "transformers==4.49.0",
    "trl==0.16.0",
    "vllm==0.7.2",
    "wandb>=0.19.1",
    "pebble==5.1.1",
    "ijson>=3.4.0",
    "pyparsing==3.2.1",
    "word2number==1.1",
    "math-verify==0.5.2",  # Used for math verification in grpo
]

# this is a lookup table with items like:
#
# tokenizers: "tokenizers==0.9.4"
# packaging: "packaging"
#
# some of the values are versioned whereas others aren't.
deps = {b: a for a, b in (re.findall(r"^(([^!=<>~ \[\]]+)(?:\[[^\]]+\])?(?:[!=<>~ ].*)?$)", x)[0] for x in _deps)}


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


extras = {}
extras["tests"] = deps_list("pytest", "parameterized")
extras["torch"] = deps_list("torch")
extras["quality"] = deps_list("ruff", "isort", "flake8")
extras["code"] = deps_list("e2b-code-interpreter", "python-dotenv")
extras["eval"] = deps_list("pyparsing", "word2number", "math-verify")
extras["dev"] = extras["quality"] + extras["tests"] + extras["eval"]

# core dependencies shared across the whole project - keep this to a bare minimum :)
install_requires = [
    deps["accelerate"],
    deps["bitsandbytes"],
    deps["einops"],
    deps["datasets"],
    deps["deepspeed"],
    deps["hf_transfer"],
    deps["huggingface-hub"],
    deps["liger_kernel"],
    deps["packaging"],  # utilities from PyPA to e.g., compare versions
    deps["safetensors"],
    deps["sentencepiece"],
    deps["transformers"],
    deps["trl"],
    deps["wandb"],
    deps["pebble"],
    deps["peft"],
    deps["vllm"],
    deps["ijson"],
]

setup(
    name="rea-rl",
    version="0.1.0.dev0",  # expected format is one of x.y.z.dev0, or x.y.z.rc1 or x.y.z (no to dashes, yes to dots)
    author="Anonymous",
    author_email="Anonymous",
    description="REA-RL",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="llm inference-time compute reasoning",
    license="Apache",
    package_dir={"": "src"},
    packages=find_packages("src"),
    zip_safe=False,
    extras_require=extras,
    python_requires=">=3.11.0",
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
