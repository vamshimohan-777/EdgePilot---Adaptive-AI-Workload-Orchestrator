import os
import pandas as pd
import sqlite3

# =============================================================================
# EdgePilot — P3: AI Intelligence
# telemetry_db.py — Manages conversion from raw CSV logs to SQLite DB and
#                    generates pandas dataframes for model training.
# =============================================================================

class TelemetryDatabase:
    def __init__(self, csv_path="edgepilot_telemetry.csv", db_path="edgepilot_telemetry.db"):
        self.csv_path = csv_path
        self.db_path = db_path

    def sync_csv_to_db(self):
        """Reads new records from the CSV file and inserts them into SQLite."""
        if not os.path.exists(self.csv_path):
            print(f"[TelemetryDB] CSV file not found: {self.csv_path}")
            return

        df = pd.read_csv(self.csv_path)
        
        conn = sqlite3.connect(self.db_path)
        # Store execution history in table 'execution_history'
        df.to_sql("execution_history", conn, if_exists="replace", index=False)
        conn.close()
        print(f"[TelemetryDB] Synced {len(df)} records to SQLite database at {self.db_path}")

    def load_training_dataset(self):
        """Loads execution telemetry and returns a cleaned pandas DataFrame for training."""
        self.sync_csv_to_db()
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query("SELECT * FROM execution_history", conn)
        except Exception as e:
            print(f"[TelemetryDB] Failed to read from database: {e}")
            df = pd.DataFrame()
        finally:
            conn.close()
        return df

    def get_model_baseline(self, model_id):
        """Returns baseline latency and memory stats for cold-start profiling query."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT AVG(latency_us), AVG(peak_memory_bytes), AVG(energy_estimate_joules)
                FROM execution_history
                WHERE model_id = ?
            """, (model_id,))
            row = cursor.fetchone()
            if row and row[0] is not None:
                return {
                    "avg_latency_us": row[0],
                    "avg_peak_memory_bytes": row[1],
                    "avg_energy_joules": row[2]
                }
        except Exception as e:
            print(f"[TelemetryDB] Failed query: {e}")
        finally:
            conn.close()
        return None
