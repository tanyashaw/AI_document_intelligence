from dotenv import load_dotenv
import os
import warnings

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")

# ── Auth ────────────────────────────────────────────────────────────────
# JWT signing secret. MUST be set to a long random value in production
# (e.g. `openssl rand -hex 32`). Falls back to an insecure dev default so
# the app still boots locally, but warns loudly so nobody ships it as-is.
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not JWT_SECRET_KEY:
    JWT_SECRET_KEY = "insecure-dev-secret-do-not-use-in-production"
    warnings.warn(
        "JWT_SECRET_KEY is not set. Using an insecure default — set a real "
        "secret via the JWT_SECRET_KEY environment variable before deploying.",
        stacklevel=2,
    )

JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

# Hosted embeddings (see app/vectordb/embedder.py) — replaces the local
# sentence-transformers model to keep the app inside Render's 512MB limit.
JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_EMBEDDING_MODEL = os.getenv("JINA_EMBEDDING_MODEL", "jina-embeddings-v3")