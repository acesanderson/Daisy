import os
import dspy
from dspy.datasets import HotPotQA

openai_api_key = os.getenv("OPENAI_API_KEY")

lm = dspy.LM("openai/gpt-4o-mini", api_key=openai_api_key)
dspy.configure(lm=lm)

# math = dspy.ChainOfThought("question -> answer: float")
# p = math(
#     question="Two dice are tossed. What is the probability that the sum equals two?"
# )


# def search_wikipedia(query: str) -> list[str]:
#     results = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")(
#         query, k=3
#     )
#     return [x["text"] for x in results]
#
#
# rag = dspy.ChainOfThought("context, question -> response")
#
# question = "What's the name of the castle that David Gregory inherited?"
# p = rag(context=search_wikipedia(question), question=question)
# print(p)
"""
instructions='Given the `question`, your task is to produce the `answer`. You will interleave your reasoning (Thought), select│
   29 #  │ the appropriate tool (next_tool_name), and provide the necessary arguments (next_tool_args) to gather information. Use the follow│
   28 #  │ing tools: (1) `search_wikipedia`, which requires a JSON formatted query, and (2) `finish`, which indicates that the final outputs│
   27 #  │ are ready. Start by analyzing the question to determine the best approach for retrieving the answer.'                            │
   26 #  │    q
"""

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))


def search_wikipedia(query: str) -> list[str]:
    results = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")(
        query, k=3
    )
    return [x["text"] for x in results]


trainset = [
    x.with_inputs("question") for x in HotPotQA(train_seed=2024, train_size=500).train
]
react = dspy.ReAct("question -> answer", tools=[search_wikipedia])

tp = dspy.MIPROv2(metric=dspy.evaluate.answer_exact_match, auto="light", num_threads=24)
optimized_react = tp.compile(react, trainset=trainset)
