from Kramer.database.MongoDB_CRUD import get_all_courses_sync
from rich.console import Console
import chromadb
import dspy
import os


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
trainset = [x.with_inputs("topic", "intro_video") for x in trainset]
print(len(trainset))


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


def get_similarity_score(original_text: str, generated_text: str):
    """Get the similarity score between two texts."""
    if not isinstance(original_text, str) or not isinstance(generated_text, str):
        return 0.0  # Return a default poor score instead of None

    client = chromadb.Client()

    # Clean up previous collection if exists
    try:
        client.delete_collection("similarity_test")
    except:
        pass

    collection = client.create_collection("similarity_test")

    try:
        collection.add(
            documents=[original_text, generated_text], ids=["original", "generated"]
        )

        results = collection.query(
            query_texts=[original_text], n_results=2, include=["distances"]
        )

        similarity_score = results["distances"][0][1]
        return float(similarity_score)
    except Exception as e:
        print(f"Error in similarity calculation: {e}")
        return 0.0  # Return a default poor score instead of None
    finally:
        # Clean up
        try:
            client.delete_collection("similarity_test")
        except:
            pass


write_video_transcript = dspy.ChainOfThought(WriteVideoTranscript)


# Plot the data
# import pandas as pd
# import matplotlib.pyplot as plt
# scores = [s for s in scores if s is not None]
# s=pd.Series(scores)
# mean = sum(scores) / len(scores)
# print(mean)
# # Histogram
# s.hist(bins=30)  # you can adjust number of bins as needed
# plt.title('Distribution of Similarity Scores')
# plt.xlabel('Similarity Score')
# plt.ylabel('Frequency')
# plt.show()
# # plt.scatter(s.index, s.values, alpha = .1)
# s.describe()
# s.to_csv("scores_500.csv", index = False)
"""
count    498.000000
mean       1.719879
std        0.163160
min        0.548691
25%        1.622713
50%        1.728114
75%        1.837510
max        2.147124
"""


def metric(example, pred, trace=None):
    topic, intro_video, transcript = (
        example.topic,
        example.intro_video,
        pred.intro_video,
    )

    similarity_score = get_similarity_score(intro_video, transcript)
    try:
        verdict = similarity_score >= 1.9
        return verdict
    except:
        return None


scores = []
n = 200
for index, x in enumerate(trainset[:n]):
    print(f"Evaluating {index+1} out of {n}")
    pred = write_video_transcript(**x.inputs())
    score = metric(x, pred)
    scores.append(score)


import matplotlib.pyplot as plt
import numpy as np


def visualize_bool_list(bool_list):
    # Calculate proportions
    total = len(bool_list)
    true_count = sum(bool_list)
    proportions = [true_count / total, (total - true_count) / total]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 2))

    # Create horizontal bar
    ax.barh(0, proportions[0], color="#2ecc71", label=f"True ({true_count})")
    ax.barh(
        0,
        proportions[1],
        left=proportions[0],
        color="#e74c3c",
        label=f"False ({total-true_count})",
    )

    # Customize appearance
    ax.set_yticks([])  # Remove y-axis ticks
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_xlim(0, 1)  # Set x-axis limits

    # Remove all spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add percentage labels
    ax.text(
        proportions[0] / 2,
        0,
        f"TRUE ({true_count}/{total})",
        ha="center",
        va="center",
        color="white",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        proportions[0] + proportions[1] / 2,
        0,
        f"FALSE ({total-true_count}/{total})",
        ha="center",
        va="center",
        color="white",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.show()


# Example usage:
# scores = [s for s in scores if s is not None]
# bool_list = scores
# visualize_bool_list(bool_list)
#

teleprompter = dspy.MIPROv2(
    metric=metric,
    num_threads=24,
    verbose=True,
    max_bootstrapped_demos=4,
    max_labeled_demos=16,
)
# Train the model
optimized_program = teleprompter.compile(write_video_transcript, trainset=trainset)


# Save the optimized program to disk
optimizeda_program.save("intro_video.json")
optimized_program.save("intro_video.pkl")
