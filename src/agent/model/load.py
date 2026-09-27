from langchain_aws import ChatBedrockConverse

MODEL_ID = "openai.gpt-oss-20b-1:0"


def load_model() -> ChatBedrockConverse:
    """Get Bedrock model client using IAM credentials.

    gpt-oss only emits native tool calls through the Converse API; the
    InvokeModel-based ChatBedrock client returns the intended call as plain text.
    """
    return ChatBedrockConverse(model_id=MODEL_ID)
