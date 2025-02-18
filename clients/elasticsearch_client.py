import logging
import ssl
from elasticsearch import Elasticsearch
from config.config import ELASTIC_ENDPOINT, ELASTIC_API_KEY

def create_es_client():
    logging.info("Creating Elasticsearch client...")
    client = Elasticsearch(
        ELASTIC_ENDPOINT,
        api_key=ELASTIC_API_KEY,
    )
    logging.info("Elasticsearch client created.")
    return client