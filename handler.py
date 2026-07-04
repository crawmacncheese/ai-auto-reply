import json
import logging

from utils.json import to_json_compatible
from models.workflow import State
from nodes.orchestrator import orchestrate

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def handler(event, context):
    """
    Simple Lambda handler function that demonstrates dependencies are loaded
    """
    logger.info(f"Event: {json.dumps(event, indent=2)}")

    try:
        state = json.loads(event["body"])["state"]
        state = State(**state)

        state = orchestrate(state)

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            },
            "body": json.dumps(
                {
                    "state": to_json_compatible(state),
                }
            ),
        }
    except Exception as e:
        logger.error(f"Error: {e}")
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            },
            "body": json.dumps({"error": str(e)}),
        }
