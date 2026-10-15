from influxdb_client import InfluxDBClient

# ======================================
# InfluxDB Configuration
# ======================================

url = "http://localhost:8086"

token = "6nhTDXteHfWnE_4k7Kx1XQ5PYiwza293nyYfqMw_qQdW3DBtl_GV0txHWf_KYsSM0juKjJWs8cQDeIHgxzBd_g=="

org = "ABC"

bucket = "industrial_data"

# ======================================
# Connect to InfluxDB
# ======================================

client = InfluxDBClient(
    url=url,
    token=token,
    org=org
)

query_api = client.query_api()

# ======================================
# Flux Query
# ======================================

query = f'''
from(bucket: "{bucket}")
  |> range(start: -24h)
'''

# Execute Query
tables = query_api.query(query=query)

print("\n========== INFLUXDB DATA ==========\n")

# Check if data exists
if len(tables) == 0:
    print("No data found in the bucket.")

# Print all records
for table in tables:
    for record in table.records:

        print("--------------------------------------")
        print("Measurement :", record.get_measurement())
        print("Field       :", record.get_field())
        print("Value       :", record.get_value())
        print("Time        :", record.get_time())

client.close()