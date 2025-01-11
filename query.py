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

# Set up the console for pretty printing
console = Console(width=100)
# Set up the OpenAI API key and instantiate the GPT-4o model
# api_key = os.getenv("OPENAI_API_KEY")
lm = dspy.LM("ollama_chat/llama3.1", api_base="http://localhost:11434", api_key="")
# lm = dspy.LM("ollama_chat/llama3.1", api_base="http://localhost:11434", api_key="")
# lm = dspy.LM("openai/gpt-4o-mini", api_key=api_key)
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
    # Initialize ChromaDB client
    client = chromadb.Client()

    # Create a collection
    try:
        collection = client.get_collection("similarity_test")
    except chromadb.errors.InvalidCollectionException:
        collection = client.create_collection("similarity_test")

    # Add your documents
    collection.add(
        documents=[original_text, generated_text], ids=["original", "generated"]
    )

    # Query to get similarity
    results = collection.query(
        query_texts=[original_text], n_results=2, include=["distances"]
    )

    # The distances in the results represent similarity scores
    # Lower distance = higher similarity
    similarity_scores = results["distances"][0][1]
    return similarity_scores


# write_video_transcript = dspy.ChainOfThought(WriteVideoTranscript)
if __name__ == "__main__":
    write_video_transcript = dspy.ChainOfThought(WriteVideoTranscript)
    n = 100
    for index, example in enumerate(trainset[:n]):
        print(f"Processing example {index + 1} of {n} ")
        topic = example.topic
        intro_video = example.intro_video
        generated_intro_video = write_video_transcript(topic=topic)
        similarity_score = get_similarity_score(
            intro_video, generated_intro_video.intro_video
        )
        print(f"Topic: {topic}")
        print(f"Similarity Score: {similarity_score}")
        # print(f"Original Intro Video: {intro_video}")
        # print(f"Generated Intro Video: {generated_intro_video.intro_video}")
        print("\n")
