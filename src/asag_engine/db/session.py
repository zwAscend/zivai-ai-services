import os
from urllib.parse import parse_qsl, urlsplit

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker

load_dotenv()


def _build_sqlalchemy_url_from_zivai_env() -> str:
    jdbc_url = os.getenv("ZIVAI_DB_URL", "").strip()
    if not jdbc_url:
        return ""

    if jdbc_url.startswith("jdbc:"):
        jdbc_url = jdbc_url[len("jdbc:"):]

    parsed = urlsplit(jdbc_url)
    if parsed.scheme not in {"postgresql", "postgres"}:
        raise RuntimeError(
            "ZIVAI_DB_URL must be a PostgreSQL JDBC URL, for example "
            "'jdbc:postgresql://host:5432/zivai'."
        )

    database = parsed.path.lstrip("/")
    if not database:
        raise RuntimeError("ZIVAI_DB_URL is missing the database name.")

    username = os.getenv("ZIVAI_DB_USERNAME", parsed.username or "").strip()
    password = os.getenv("ZIVAI_DB_PASSWORD", parsed.password or "")
    if not username:
        raise RuntimeError("ZIVAI_DB_USERNAME is not set.")
    if not parsed.hostname:
        raise RuntimeError("ZIVAI_DB_URL is missing the database host.")

    return URL.create(
        drivername="postgresql+psycopg2",
        username=username,
        password=password or None,
        host=parsed.hostname,
        port=parsed.port,
        database=database,
        query=dict(parse_qsl(parsed.query, keep_blank_values=True)),
    ).render_as_string(hide_password=False)


def _resolve_database_url() -> str:
    database_url = _build_sqlalchemy_url_from_zivai_env()
    if database_url:
        return database_url

    database_url = os.getenv("DATABASE_URL", "").strip()
    if database_url:
        return database_url

    raise RuntimeError(
        "DATABASE_URL is not set. Either define DATABASE_URL directly or set "
        "ZIVAI_DB_URL, ZIVAI_DB_USERNAME, and ZIVAI_DB_PASSWORD to match core-backend."
    )


DATABASE_URL = _resolve_database_url()

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_session():
    return SessionLocal()
