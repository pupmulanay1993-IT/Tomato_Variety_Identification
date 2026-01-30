import os
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

try:
    from supabase import create_client
except Exception:
    create_client = None

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

_supabase = None
if create_client and SUPABASE_URL and SUPABASE_KEY:
    try:
        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        _supabase = None


def insert_prediction(record: dict):
    """Insert a prediction record into the `predictions` table.
    Returns the Supabase response or None on failure.
    """
    if not _supabase:
        return None
    try:
        return _supabase.table("predictions").insert(record).execute()
    except Exception:
        return None
