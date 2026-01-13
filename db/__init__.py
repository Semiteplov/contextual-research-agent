from common.settings import get_settings


def get_pg_dsn(db_name: str) -> str:
    settings = get_settings()
    host = settings.postgres_host
    port = settings.postgres_port
    user = settings.postgres_user
    pwd = settings.postgres_password.get_secret_value()

    return f"postgresql://{user}:{pwd}@{host}:{port}/{db_name}"
