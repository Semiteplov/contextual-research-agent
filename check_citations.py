import psycopg2
from contextual_research_agent.common.settings import get_settings

s = get_settings()
conn = psycopg2.connect(
    host=s.postgres.host,
    port=s.postgres.port,
    dbname="arxiv",
    user=s.postgres.user,
    password=s.postgres.password.get_secret_value(),
)
cur = conn.cursor()

cur.execute(
    "SELECT cited_paper_id FROM citation_edges WHERE citing_paper_id = %s LIMIT 5", ("2106.09685",)
)
print("LoRA cites:", cur.fetchall())

cur.execute(
    "SELECT citing_paper_id FROM citation_edges WHERE cited_paper_id = %s LIMIT 10", ("2106.09685",)
)
print("Citing LoRA:", cur.fetchall())

conn.close()
