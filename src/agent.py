import os
from typing import Any

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI


SYSTEM_PROMPT = """
You are a research assistant for sepsis analysis.

Audience:
- Scientific researchers, clinicians, epidemiologists, bioinformaticians.

Scope:
- Explain sepsis-related concepts, biomarkers, study designs, immune pathways,
  trial endpoints, risk models, diagnostics, omics analysis, and literature-style reasoning.
- You may interpret user-provided images only as research material.
- Do not provide patient-specific diagnosis, triage, or treatment instructions.

Evidence behavior:
- When an uploaded image is relevant, explicitly reference it as visual evidence.
- When evidence is uncertain, state the uncertainty.
- Do not fabricate citations, image findings, or statistical results.

Safety:
- This backend is for research support only.
- It is not a clinical decision support system.
"""


@tool
def sepsis_domain_notes(topic: str) -> str:
    """
    Return concise background notes for common sepsis research topics.
    Use this for definitions, mechanisms, and study-analysis framing.
    """
    topic_l = topic.lower()

    if "sofa" in topic_l:
        return (
            "SOFA is commonly used to describe organ dysfunction in sepsis research. "
            "It summarizes respiratory, coagulation, liver, cardiovascular, CNS, and renal components."
        )

    if "lactate" in topic_l:
        return (
            "Lactate is often analyzed as a marker of tissue hypoperfusion and illness severity, "
            "but it is not specific to sepsis and should be interpreted with clinical context."
        )

    if "cytokine" in topic_l or "immune" in topic_l:
        return (
            "Sepsis involves dysregulated host response, including inflammatory activation, "
            "endothelial dysfunction, immune suppression, and metabolic reprogramming."
        )

    return (
        "For sepsis research, distinguish clinical phenotype, data source, cohort definition, "
        "endpoint, confounding structure, and measurement timing before drawing conclusions."
    )


@tool
def suggest_analysis_plan(question: str) -> str:
    """
    Suggest a research-analysis plan for a sepsis-related question.
    """
    return (
        "Suggested structure: define cohort inclusion/exclusion criteria; specify sepsis definition; "
        "define exposure, comparator, endpoint, and time zero; inspect missingness; choose statistical "
        "or ML model; validate internally; assess calibration, discrimination, sensitivity analyses, "
        "and biological plausibility."
    )


def build_agent():
    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    llm = ChatOpenAI(
        model=model_name,
        temperature=0.2,
    )

    return create_agent(
        model=llm,
        tools=[sepsis_domain_notes, suggest_analysis_plan],
        system_prompt=SYSTEM_PROMPT,
    )


def build_multimodal_user_message(message: str, uploaded_images: list[dict]) -> dict[str, Any]:
    """
    Uses OpenAI-style multimodal content blocks.
    The model receives base64 data URLs.
    The frontend receives normal static URLs in the API response.
    """
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": message,
        }
    ]

    for image in uploaded_images:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image["data_url"],
                },
            }
        )

    return {
        "role": "user",
        "content": content,
    }


async def ask_agent(message: str, uploaded_images: list[dict]) -> str:
    agent = build_agent()

    user_message = build_multimodal_user_message(
        message=message,
        uploaded_images=uploaded_images,
    )

    result = await agent.ainvoke(
        {
            "messages": [user_message],
        }
    )

    messages = result.get("messages", [])
    if not messages:
        return "No response was generated."

    final_message = messages[-1]

    content = getattr(final_message, "content", final_message)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return "\n".join(text_parts).strip() or "No text response was generated."

    return str(content)