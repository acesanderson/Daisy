"""
Boilerplate from DSPY official documentation. This uses multiple stages to write an article from a given topic.
This analogizes to our course transcript generating workflow.
```mermaid
graph LR;
    A[Topic] --> B[Course TOC]
    B --> C[Course Description]
    A --> D[Verbose TOC]
    B --> D
    C --> D
    A --> E[Transcript loop]
    B --> E
    C --> E
    D --> E
    E --> F[Video transcript]
    F --> E
    F --> G[Complete Video Course]
```
"""

import dspy
import os
from rich.console import Console
from rich.markdown import Markdown

api_key = os.getenv("OPENAI_API_KEY")
console = Console(width=100)

# lm = dspy.LM("ollama_chat/llama3.1", api_base="http://localhost:11434", api_key="")
lm = dspy.LM("openai/gpt-4o-mini", api_key="")
dspy.configure(lm=lm)


class Outline(dspy.Signature):
    """Outline a thorough overview of a video course."""

    topic: str = dspy.InputField()
    title: str = dspy.OutputField()
    sections: list[str] = dspy.OutputField()
    section_subheadings: dict[str, list[str]] = dspy.OutputField(
        desc="mapping from section headings to subheadings"
    )


class DraftSection(dspy.Signature):
    """Draft a top-level section of a video."""

    topic: str = dspy.InputField()
    section_heading: str = dspy.InputField()
    section_subheadings: list[str] = dspy.InputField()
    content: str = dspy.OutputField(desc="markdown-formatted section")


class DraftCourse(dspy.Module):
    def __init__(self):
        self.build_outline = dspy.ChainOfThought(Outline)
        self.draft_section = dspy.ChainOfThought(DraftSection)

    def forward(self, topic):
        outline = self.build_outline(topic=topic)
        sections = []
        for heading, subheadings in outline.section_subheadings.items():
            section, subheadings = f"## {heading}", [
                f"### {subheading}" for subheading in subheadings
            ]
            section = self.draft_section(
                topic=outline.title,
                section_heading=section,
                section_subheadings=subheadings,
            )
            sections.append(section.content)
        return dspy.Prediction(title=outline.title, sections=sections)


draft_course = DraftCourse()
# course = draft_course(topic="Intermediate Python for Data Scientists")
course = draft_course(topic="Body Language for Executives")
console.print(Markdown("\n".join(course.sections)))
