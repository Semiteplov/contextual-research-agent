class DBError(Exception):
    """Base exception for database operations."""


class DBConnectionError(DBError):
    """Failed to establish database connection."""


class QueryError(DBError):
    """Failed to execute database query."""


class IntegrityError(DBError):
    """Data integrity constraint violation."""
