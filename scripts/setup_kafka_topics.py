"""
scripts/setup_kafka_topics.py
Creates all required Kafka topics with correct configurations.
Run once after docker compose up.
"""

from confluent_kafka.admin import AdminClient, NewTopic
import sys

BOOTSTRAP_SERVERS = "localhost:9092"

TOPICS = [
    NewTopic("transactions",         num_partitions=3, replication_factor=1),
    NewTopic("flagged_transactions",  num_partitions=3, replication_factor=1),
    NewTopic("enriched_transactions", num_partitions=3, replication_factor=1),
    NewTopic("case_reports",          num_partitions=3, replication_factor=1),
]

def create_topics():
    admin = AdminClient({"bootstrap.servers": BOOTSTRAP_SERVERS})
    existing = admin.list_topics(timeout=10).topics.keys()

    to_create = [t for t in TOPICS if t.topic not in existing]
    if not to_create:
        print("All topics already exist.")
        return

    results = admin.create_topics(to_create)
    for topic, future in results.items():
        try:
            future.result()
            print(f"  Created topic: {topic}")
        except Exception as e:
            print(f"  Topic {topic} already exists or error: {e}")

if __name__ == "__main__":
    print("Setting up Kafka topics...")
    create_topics()
    print("Done.")
