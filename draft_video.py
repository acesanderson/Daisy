import dspy
import os
from rich.console import Console

api_key = os.getenv("OPENAI_API_KEY")
console = Console(width=100)

# lm = dspy.LM("ollama_chat/llama3.1", api_base="http://localhost:11434", api_key="")
lm = dspy.LM("openai/gpt-4o-mini", api_key="")
dspy.configure(lm=lm)

topic = "Body Language for Executives"


class WriteVideoTranscript(dspy.Signature):
    """Write a video transcript for the intro video to a course on a given topic."""

    topic: str = dspy.InputField(desc="the topic of the course")
    intro_video: str = dspy.OutputField(
        desc="markdown-formatted transcript of an intro video of roughly 250 words"
    )


write_video_transcript = dspy.ChainOfThought(WriteVideoTranscript)
intro_video = write_video_transcript(topic=topic)
console.print(intro_video.intro_video)
