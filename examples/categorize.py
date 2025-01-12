import dspy
from typing import Literal

lm = dspy.LM("ollama_chat/llama3.1", api_base="http://localhost:11434")
dspy.configure(lm=lm)


class Categorize(dspy.Signature):
    """Classify historic events."""

    event: str = dspy.InputField()
    category: Literal[
        "Wars and Conflicts",
        "Politics and Governance",
        "Science and Innovation",
        "Cultural and Artistic Movements",
        "Exploration and Discovery",
        "Economic Events",
        "Social Movements",
        "Man-Made Disasters and Accidents",
        "Natural Disasters and Climate",
        "Sports and Entertainment",
        "Famous Personalities and Achievements",
    ] = dspy.OutputField()
    confidence: float = dspy.OutputField()


classify = dspy.Predict(Categorize)

# Here is how we call this module
classification = classify(
    event="Secopnd Boer War: In the Battle of Magersfontein, Boer forces defeat a British army, killing more than 200 soldiers."
)

print(classification)
