from database.schema import get_connection

conn = get_connection()

print("monte_carlo_significance columns:")
cols = [r[1] for r in conn.execute("PRAGMA table_info(monte_carlo_significance)").fetchall()]
print(" ", cols)
print()

rows = conn.execute("SELECT * FROM monte_carlo_significance").fetchall()
print(f"Row count: {len(rows)}")
for r in rows:
    print(" ", dict(r))

conn.close()
