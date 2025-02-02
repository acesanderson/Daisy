from Kramer.database.MongoDB_CRUD import get_all_courses_sync
from rich.console import Console
from chromadb.utils import embedding_functions
import dspy
from dspy import Example
import os
import numpy as np


def get_trainset() -> list[Example]:
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
    return trainset


# Set up the console for pretty printing
console = Console(width=100)
# Set up the OpenAI API key and instantiate the GPT-4o model
api_key = os.getenv("OPENAI_API_KEY")
# lm = dspy.LM("ollama_chat/llama3.1", api_base="http://localhost:11434", api_key="")
# lm = dspy.LM("ollama_chat/llama3.1", api_base="http://localhost:11434", api_key="")
# lm = dspy.LM("openai/gpt-4o-mini", api_key=api_key)
lm = dspy.LM("openai/gpt-3.5-turbo-0125", api_key=api_key)
dspy.configure(lm=lm)


class WriteVideoTranscript(dspy.Signature):
    """Write a video transcript for the intro video to a course on a given topic."""

    topic: str = dspy.InputField(desc="the topic of the course")
    intro_video: str = dspy.OutputField(
        desc="markdown-formatted transcript of an intro video of roughly 250 words"
    )


def get_similarity_score(original_text: str, generated_text: str):
    """Get the similarity score between two texts using direct embedding comparison."""
    if not isinstance(original_text, str) or not isinstance(generated_text, str):
        return 0.0
    embedding_model = embedding_functions.DefaultEmbeddingFunction()
    try:
        # Generate embeddings directly
        original_embedding = embedding_model([original_text])[0]
        generated_embedding = embedding_model([generated_text])[0]

        # Calculate cosine similarity
        similarity = np.dot(original_embedding, generated_embedding) / (
            np.linalg.norm(original_embedding) * np.linalg.norm(generated_embedding)
        )

        # Convert similarity to a distance (1 - similarity) to match ChromaDB's distance format
        distance = 1 - similarity
        return float(distance)
    except Exception as e:
        print(f"Error in similarity calculation: {e}")
        return 0.0


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


if __name__ == "__main__":
    write_video_transcript = dspy.ChainOfThought(WriteVideoTranscript)
    scores = []
    n = 200
    trainset = get_trainset()
    for index, x in enumerate(trainset[:n]):
        print(f"Evaluating {index+1} out of {n}")
        pred = write_video_transcript(**x.inputs())
        score = metric(x, pred)
        scores.append(score)
    exit()
    n = 100
    teleprompter = dspy.MIPROv2(
        metric=metric,
        num_threads=24,
        verbose=True,
        max_bootstrapped_demos=4,
        max_labeled_demos=16,
    )
    # Train the model
    optimized_program = teleprompter.compile(
        write_video_transcript, trainset=trainset[:n]
    )

    # Save the optimized program to disk
    optimized_program.save("intro_video.json")
    optimized_program.save("intro_video.pkl")
