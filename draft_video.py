import dspy
import os
from rich.console import Console
from Kramer.database.MongoDB_CRUD import get_all_courses_sync

# Get our training set: course titles and first video transcript
print("Getting courses...")
courses = get_all_courses_sync()
trainset = []
for course in courses:
    try:
        topic = course.course_title
        intro_video = course.sections[0].entries[0].transcript
        trainset.append(dspy.Example(topic=topic, intro_video=intro_video))
    except:
        pass
print(f"Got {len(trainset)} courses.")

# Set up the console for pretty printing
console = Console(width=100)
# Set up the OpenAI API key and instantiate the GPT-4o model
api_key = os.getenv("OPENAI_API_KEY")
# lm = dspy.LM("ollama_chat/llama3.1", api_base="http://localhost:11434", api_key="")
lm = dspy.LM("ollama_chat/llama3.1", api_base="http://localhost:11434", api_key="")
# lm = dspy.LM("openai/gpt-4o-mini", api_key="")
dspy.configure(lm=lm)

topic = "ChatGPT for beginners"


class WriteVideoTranscript(dspy.Signature):
    """Write a video transcript for the intro video to a course on a given topic."""

    topic: str = dspy.InputField(desc="the topic of the course")
    intro_video: str = dspy.OutputField(
        desc="markdown-formatted transcript of an intro video of roughly 250 words"
    )


write_video_transcript = dspy.ChainOfThought(WriteVideoTranscript)
intro_video = write_video_transcript(topic=topic)
console.print(intro_video.intro_video)


class Assess(dspy.Signature):
    """Assess the quality of a video transcript along the specified dimension."""

    assessed_transcript = dspy.InputField()
    assessment_topic = dspy.InputField()
    assessment_intro_video: bool = dspy.OutputField()


def metric(gold, pred, trace=None):
    """
    Evaluation metric we will use to train the dspy model.
    """
    topic, intro_video, transcript = gold.topic, gold.intro_video, pred.intro_video

    comprehensive = (
        f"Does the assessed transcript cover the most import topics related to {topic}?"
    )
    similar = f"Does the assessed transcript match the tone and overall instructional approach of this example transcript on the same topic?\n\n{intro_video}"

    comprehensive = dspy.Predict(Assess)(
        assessed_transcript=transcript, assessment_topic=comprehensive
    )
    similar = dspy.Predict(Assess)(
        assessed_transcript=transcript, assessment_topic=similar
    )

    comprehensive, similar = [m.assessment_transcript for m in [comprehensive, similar]]
    score = (
        (comprehensive + similar)
        if comprehensive and (200 < len(transcript.split()) < 300)
        else 0
    )

    if trace is not None:
        return score >= 2
    return score / 2.0


trainset = [x.with_inputs("topic", "intro_video") for x in trainset]
scores = []
for index, x in enumerate(trainset[:10]):
    print(f"Example {index + 1} of {len(trainset)}")
    pred = write_video_transcript(**x.inputs())
    score = metric(x, pred)
    scores.append(score)
