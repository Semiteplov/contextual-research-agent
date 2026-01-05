import os


def get_pg_dsn(db_name: str) -> str:
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER", "postgres")
    pwd = os.getenv("POSTGRES_PASSWORD", "postgres")

    return f"postgresql://{user}:{pwd}@{host}:{port}/{db_name}"
