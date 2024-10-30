TOOL_MODEL="claude-3-haiku-20240307"
EMBEDDING_MODEL="text-embedding-3-large"
SEARCHING_TOOL = [
    {
        "name": "retrieval",
        "description": "For any queries regarding civil engineering issues, answer the question in the query field by fetching relevant documents from vectorstore. Keep the user query as it is if it requires retrieval.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query in Korean language against which relevant documents will be retrieved from retrieval tool. Don't change the user query."
                }
            },
            "required": ["query"]
        }
    }
]