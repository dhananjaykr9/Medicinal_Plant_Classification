# agent.py
# Lightweight Agentic AI for prediction reliability

class PredictionAgent:
    def __init__(self, confidence_threshold=0.70):
        self.confidence_threshold = confidence_threshold

    def decide(self, confidence):
        """
        Decide whether prediction is reliable
        """

        if confidence < self.confidence_threshold:
            return {
                "decision": "REUPLOAD",
                "message": "Prediction confidence is low. Please upload a clearer leaf image."
            }

        return {
            "decision": "APPROVE",
            "message": "Prediction confidence is sufficient. Showing results."
        }
