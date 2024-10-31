from pathlib import Path

import modal


def find_project_root() -> Path:
    current_dir = Path(__file__).resolve().parent
    while current_dir != current_dir.parent:
        if (current_dir / ".root").exists():
            return current_dir
        current_dir = current_dir.parent
    raise RuntimeError("Could not find project root; no .root file found in parent directories")


REPO_BRANCH = "main"
REMOTE_ROOT = Path("/root/observatory")
PROJECT_ROOT = find_project_root() if modal.is_local() else REMOTE_ROOT


stub = modal.App("monitor-server-test")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .run_commands(
        f"git clone -b {REPO_BRANCH} https://github.com/TransluceAI/observatory.git {REMOTE_ROOT}",
        f"pip install -e {REMOTE_ROOT}/project/monitor",
    )
    .copy_local_file(PROJECT_ROOT / ".env", f"/.env")
    .env({"HF_HOME": "/root/.cache/huggingface"})
)

# Load the llama3.1-8b-instruct model
volume = modal.Volume.from_name("llama3.1-8b-instruct")
model_store_path = "/root/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/"

# Define the GPU configuration
gpu = modal.gpu.A100(count=1)  # type: ignore

# TODO: fix these so the server becomes persistent!
# INTERVENTIONS = modal.Dict.from_name("interventions", create_if_missing=True)
# SESSIONS = modal.Dict.from_name("sessions", create_if_missing=True)
# FILTER_CACHE = modal.Dict.from_name("filter_cache", create_if_missing=True)
# LINTER_CACHE = modal.Dict.from_name("linter_cache", create_if_missing=True)
# MESSAGE_CACHE = modal.Dict.from_name("generation_cache", create_if_missing=True)


@stub.function(
    image=image,
    gpu=gpu,  # type: ignore
    concurrency_limit=1,
    allow_concurrent_inputs=1,
    volumes={model_store_path: volume},
    container_idle_timeout=300,
    keep_warm=1,
)
# @modal.asgi_app(custom_domains=["monitor-backend.transluce.org"])
@modal.asgi_app()  # type: ignore
def app():
    from monitor.server import asgi_app

    return asgi_app
