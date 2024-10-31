import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel


class EnvironmentVariables(BaseModel):
    OPENAI_API_ORG: str | None
    OPENAI_API_KEY: str | None

    ANTHROPIC_API_KEY: str | None

    HF_TOKEN: str | None

    PG_USER: str | None
    PG_PASSWORD: str | None
    PG_HOST: str | None
    PG_PORT: str | None
    PG_DATABASE: str | None

    @classmethod
    def load_from_env(cls):
        env_file = find_dotenv()
        load_dotenv(env_file)

        openai_api_org = os.getenv("OPENAI_API_ORG")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        hf_token = os.getenv("HF_TOKEN")
        pg_user = os.getenv("PG_USER")
        pg_password = os.getenv("PG_PASSWORD")
        pg_host = os.getenv("PG_HOST")
        pg_port = os.getenv("PG_PORT")
        pg_database = os.getenv("PG_DATABASE")

        return cls(
            OPENAI_API_ORG=openai_api_org,
            OPENAI_API_KEY=openai_api_key,
            ANTHROPIC_API_KEY=anthropic_api_key,
            HF_TOKEN=hf_token,
            PG_USER=pg_user,
            PG_PASSWORD=pg_password,
            PG_HOST=pg_host,
            PG_PORT=pg_port,
            PG_DATABASE=pg_database,
        )


def find_dotenv():
    """
    Find the .env file in the project directory. Stops ascending at the project root.
    Raises an error with the list of paths explored if no .env file is found.
    """
    current_dir = Path(__file__).parent.resolve()
    paths_explored: list[str] = []

    while True:
        paths_explored.append(str(current_dir))
        env_file = current_dir / ".env"
        if env_file.is_file():
            return str(env_file)
        if is_project_root(current_dir):
            break
        if current_dir == current_dir.parent:
            break
        current_dir = current_dir.parent

    raise FileNotFoundError(f"No .env file found. Paths explored: {', '.join(paths_explored)}")


def is_project_root(directory: Path):
    return (directory / ".root").exists()


ENV = EnvironmentVariables.load_from_env()
