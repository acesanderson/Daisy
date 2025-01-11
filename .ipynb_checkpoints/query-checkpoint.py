#!/usr/bin/env python
# coding: utf-8


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
    """
    Get the similarity score between two texts.
    """
    if not isinstance(original_text, str):
        return None
    if not isinstance(generated_text, str):
        return None
    # Initialize ChromaDB client
    client = chromadb.Client()

    # Create a collection
    try:
        collection = client.get_collection("similarity_test")
    except chromadb.errors.InvalidCollectionException:
        collection = client.create_collection("similarity_test")

    # Add your documents
    try:
        collection.add(
            documents=[original_text, generated_text], ids=["original", "generated"]
        )
    except:
        print("Error adding document for whatever reason.")

    # Query to get similarity
    results = collection.query(
        query_texts=[original_text], n_results=2, include=["distances"]
    )

    # The distances in the results represent similarity scores
    # Lower distance = higher similarity
    similarity_scores = results["distances"][0][1]
    return similarity_scores


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
    return similarity_score >= 1.8


teleprompter = dspy.MIPROv2(metric=metric, num_threads=24, verbose=True)
# Train the model
optimized_program = teleprompter.compile(write_video_transcript, trainset=trainset)

# Save the optimized program to disk
optimized_program.save("intro_video.json")
optimized_program.save("intro_video.pkl")
