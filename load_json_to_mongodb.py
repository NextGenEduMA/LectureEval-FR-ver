import json
import pymongo
import os
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017')  # Default to local MongoDB
json_path = 'texts/texts.json'  # Path to your JSON file

def load_json_to_mongodb():
    try:
        # Connect to MongoDB
        client = pymongo.MongoClient(mongo_uri)
        db = client['language_learning']  # Database name
        text_collection = db['texts']  # Collection name
        client.server_info()  # Test connection
        logger.info("Connected to MongoDB successfully")

        # Read JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            text_db = json.load(f)
        
        # Clear existing collection to avoid duplicates (optional)
        text_collection.delete_many({})
        logger.info("Cleared existing texts in MongoDB collection")

        # Insert texts into MongoDB
        text_collection.insert_many(text_db)
        logger.info(f"Loaded {len(text_db)} texts into MongoDB")

        # Verify insertion
        count = text_collection.count_documents({})
        logger.info(f"Total documents in collection: {count}")

    except Exception as e:
        logger.error(f"Failed to load JSON into MongoDB: {e}")
        raise
    finally:
        client.close()
        logger.info("MongoDB connection closed")

if __name__ == '__main__':
    load_json_to_mongodb()