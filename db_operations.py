from pymongo import MongoClient

def insert_document(mongo_uri, database_name, collection_name, data_to_insert):
    """
    This function inserts a document into a MongoDB collection.

    Parameters:
    - mongo_uri: URI for connecting to MongoDB.
    - database_name: Name of the database to use.
    - collection_name: Name of the collection to insert the document into.
    - data_to_insert: A dictionary representing the document to be inserted.

    Returns:
    The ID of the inserted document.
    """
    client = MongoClient(mongo_uri)
    db = client[database_name]
    collection = db[collection_name]
    insert_result = collection.insert_one(data_to_insert)
    return insert_result.inserted_id
