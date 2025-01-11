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
# lm = dspy.LM("ollama_chat/llama3.1", api_base="http://localhost:11434", api_key="")
lm = dspy.LM("openai/gpt-4o-mini", api_key=api_key)
dspy.configure(lm=lm)


class WriteVideoTranscript(dspy.Signature):
    """Write a video transcript for the intro video to a course on a given topic."""

    topic: str = dspy.InputField(desc="the topic of the course")
    intro_video: str = dspy.OutputField(
        desc="markdown-formatted transcript of an intro video of roughly 250 words"
    )


write_video_transcript = dspy.ChainOfThought(WriteVideoTranscript)

# Generate a video transcript; note this is unoptimized and uses bare syntax
# topic = "ChatGPT for beginners"
# intro_video = write_video_transcript(topic=topic)
# console.print(intro_video.intro_video)


# class Assess(dspy.Signature):
#     """Assess the quality of a video transcript along the specified dimension."""
#
#     assessed_transcript = dspy.InputField()
#     assessment_topic = dspy.InputField()
#     assessment_intro_video: bool = dspy.OutputField()


# RERANKER CODE
# def rerank_options(options: list[tuple], query: str, k: int = 5) -> list[tuple]:
#     """
#     Reranking magic.
#     """
#     reranker = FlagReranker(
#         "BAAI/bge-reranker-large", use_fp16=True
#     )  # Setting use_fp16 to True speeds up computation with a slight performance degradation
#     ranked_results: list[tuple] = []
#     for option in options:
#         course = option[0]  # This is "id" from the Chroma output.
#         TOC = option[1]  # This is "document" from the Chroma output.
#         score = reranker.compute_score([query, TOC])
#         ranked_results.append((course, score))
#     # sort ranked_results by highest score
#     ranked_results.sort(key=lambda x: x[1], reverse=True)
#     # Return the five best.
#     return ranked_results[:k]


def metric(gold, pred, trace=False):
    topic, intro_video, transcript = gold.topic, gold.intro_video, pred.intro_video
    reranker = FlagReranker(
        "BAAI/bge-reranker-large", use_fp16=True
    )  # Setting use_fp16 to True speeds up computation with a slight performance degradation
    score = reranker.compute_score(


def metric(gold, pred, trace=False):
    """
    Evaluation metric we will use to train the dspy model.
    'trace' = True will return a boolean indicating whether the prediction is correct.
    'trace' = False will return a float score between 0 and 1.
    """
    topic, intro_video, transcript = gold.topic, gold.intro_video, pred.intro_video

    similar = dspy.Predict(Assess)(
        assessed_transcript=transcript, assessment_topic=similar
    )

    similar = [m.assessment_intro_video for m in [similar]]

    score = similar if (200 < len(transcript.split()) < 300) else 0

    if trace is not None:
        return score >= 2
    return score / 2.0


# def metric(gold, pred, trace=False):
#     """
#     Evaluation metric we will use to train the dspy model.
#     'trace' = True will return a boolean indicating whether the prediction is correct.
#     'trace' = False will return a float score between 0 and 1.
#     """
#     topic, intro_video, transcript = gold.topic, gold.intro_video, pred.intro_video
#
#     comprehensive = (
#         f"Does the assessed transcript cover the most import topics related to {topic}?"
#     )
#     similar = f"Does the assessed transcript match the tone and overall instructional approach of this example transcript on the same topic?\n\n{intro_video}"
#
#     comprehensive = dspy.Predict(Assess)(
#         assessed_transcript=transcript, assessment_topic=comprehensive
#     )
#     similar = dspy.Predict(Assess)(
#         assessed_transcript=transcript, assessment_topic=similar
#     )
#
#     comprehensive, similar = [
#         m.assessment_intro_video for m in [comprehensive, similar]
#     ]
#     score = (
#         (comprehensive + similar)
#         if comprehensive and (200 < len(transcript.split()) < 300)
#         else 0
#     )
#
#     if trace is not None:
#         return score >= 2
#     return score / 2.0

trainset = [x.with_inputs("topic", "intro_video") for x in trainset]
trainset = trainset[:500]
# scores = []
# for index, x in enumerate(trainset[:200]):
#     print(f"Example {index + 1} of {len(trainset)}")
#     pred = write_video_transcript(**x.inputs())
#     score = metric(x, pred)
#     scores.append(score)

# Miprov2: https://github.com/stanfordnlp/dspy/blob/main/docs/docs/deep-dive/optimizers/miprov2.md
# Initialize the model
teleprompter = dspy.MIPROv2(metric=metric, num_threads=24, verbose=True)
# Train the model
optimized_program = teleprompter.compile(write_video_transcript, trainset=trainset)

# Save the optimized program to disk
optimized_program.save("draft_video.pkl")
optimized_program.save("draft_video.json")


# write_video_transcript.load("draft_video.pkl")
