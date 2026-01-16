from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from psycopg2.extensions import connection as PGConnection


class BaseRepository:
    __slots__ = ("_conn",)

    def __init__(self, conn: "PGConnection") -> None:
        self._conn = conn
