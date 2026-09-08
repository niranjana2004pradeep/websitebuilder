import os
import uuid
import zipfile
import logging
import tempfile

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("website-gen-structured")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY must be set in the environment (.env). Get a free key at console.groq.com")

# Structured output is enforced by the Pydantic models and the
# with_structured_output() Runnable wrappers below.
llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.3)

app = FastAPI(title="AI Website Generation Pipeline - Structured Output (LangChain)")

WORK_DIR = os.path.join(tempfile.gettempdir(), "website_gen_structured")
os.makedirs(WORK_DIR, exist_ok=True)

MAX_SECTIONS = 5

# How much accumulated history to feed back into each new prompt. Even with
# structured (more compact) output, this is capped defensively so a long
# site can't silently regrow the same rate-limit problem hit earlier.
MAX_CONTEXT_FRAGMENTS = 3
MAX_CONTEXT_CHARS = 2000


# ---------------------------------------------------------------------------
# In-memory project state
#
# css_variables is the structured "delta" store — the shared design-system
# facts that must stay consistent — separate from the raw code fragments.
# ---------------------------------------------------------------------------

PROJECTS: dict[str, "ProjectState"] = {}


class ProjectState(BaseModel):
    html_fragments: list[str] = Field(default_factory=list)
    css_fragments: list[str] = Field(default_factory=list)
    js_fragments: list[str] = Field(default_factory=list)
    css_variables: dict[str, str] = Field(default_factory=dict)
    section_order: list[str] = Field(default_factory=list)


def build_capped_context(state: "ProjectState") -> str:
    """Only the most recent fragments, hard-capped in length — prevents
    unbounded prompt growth as more sections get generated."""
    all_fragments = state.html_fragments + state.css_fragments + state.js_fragments
    if not all_fragments:
        return "(nothing generated yet)"
    recent = all_fragments[-MAX_CONTEXT_FRAGMENTS:]
    joined = "\n\n".join(recent)
    return joined[-MAX_CONTEXT_CHARS:]


# ---------------------------------------------------------------------------
# Structured schemas
# ---------------------------------------------------------------------------

class SectionPlan(BaseModel):
    section_type: str
    content_brief: str


class SitePlan(BaseModel):
    sections: list[SectionPlan]


class SectionOutput(BaseModel):
    html: str = Field(description="HTML code for the section only. Give the outer element an id matching its own section_type.")
    css: str = Field(description="CSS rules for this section only. Do NOT declare :root or any CSS custom properties here — report new ones via new_css_variables instead.")
    js: str = Field(default="", description="JavaScript code, or an empty string if none is needed")
    new_css_variables: dict[str, str] = Field(
        default_factory=dict,
        description="Any NEW CSS custom properties this section introduces, e.g. {'--accent': '#ffcc00'}. Leave empty if reusing existing ones from context.",
    )


class ElementOutput(BaseModel):
    code: str = Field(description="Only the requested HTML, CSS, or JavaScript code — no explanation, no Markdown fences.")
    new_css_variables: dict[str, str] = Field(
        default_factory=dict,
        description="Any NEW CSS custom properties this element introduces, if applicable. Leave empty otherwise.",
    )


# ---------------------------------------------------------------------------
# LangChain Runnable chains (LCEL: prompt | llm)
# ---------------------------------------------------------------------------

PLANNING_SYSTEM_PROMPT = """You are a website architect.

Break the user's website description into an ordered list of sections.

Rules:
- The first section must be navbar.
- The last section must be footer.
- Choose the sections that match the user's description.
- Give every section a useful content brief.
- Return the result using the required structured fields.
"""

planning_prompt = ChatPromptTemplate.from_messages([
    ("system", PLANNING_SYSTEM_PROMPT),
    ("human", "{description}"),
])
structured_planner = llm.with_structured_output(SitePlan)
planning_chain = planning_prompt | structured_planner


SECTION_SYSTEM_PROMPT = """You are an expert front-end developer.

Generate one complete website section.

Full planned site structure (in order) — use these EXACT section_type values as
element ids and as anchor targets for any internal links (e.g. a link to the
"menu" section must use href="#menu", and the element for that section must
have id="menu"):
{full_site_plan}

Design system already established (reuse these variables via var(--name);
do not redeclare :root or invent duplicates — only report genuinely NEW
variables via the new_css_variables field):
{existing_css_variables}

Context — recent code generated so far in this project (for tone/style consistency):
{previous_context}

Return:
- html: the section's HTML only, with id="{section_type}" on its outer element
- css: the section's CSS only (no :root, no variable declarations)
- js: JavaScript only when needed, otherwise an empty string
- new_css_variables: only variables this section is introducing for the first time

Do not include explanations or Markdown code fences.
"""

section_prompt = ChatPromptTemplate.from_messages([
    ("system", SECTION_SYSTEM_PROMPT),
    ("human", "Section to generate: {section_type}\n\nContent brief: {content_brief}"),
])
structured_section_llm = llm.with_structured_output(SectionOutput)
section_chain = section_prompt | structured_section_llm


ELEMENT_SYSTEM_PROMPT = """You are an expert front-end developer. Generate a single {element_type} \
element for a website based on the instruction below.

Design system already established (reuse these variables via var(--name);
only report genuinely NEW variables via new_css_variables):
{existing_css_variables}

Context — recent code generated so far in this project (for consistency):
{previous_context}

Instruction: {instruction}

Existing code to modify/extend (if any): {existing_code}

Return only the requested {element_type} code — no explanation, no Markdown fences.
"""

element_prompt = ChatPromptTemplate.from_messages([
    ("system", ELEMENT_SYSTEM_PROMPT),
])
structured_element_llm = llm.with_structured_output(ElementOutput)
element_chain = element_prompt | structured_element_llm


# ---------------------------------------------------------------------------
# Shared assembly helpers
# ---------------------------------------------------------------------------

def _assemble_html(html_fragments: list[str]) -> str:
    body = "\n\n".join(html_fragments)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Generated Website</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
{body}
<script src="script.js"></script>
</body>
</html>
"""


def assemble_site(state: "ProjectState") -> tuple[str, str, str]:
    html_content = _assemble_html(state.html_fragments)

    root_vars = "\n".join(f"  {k}: {v};" for k, v in state.css_variables.items())
    root_block = f":root {{\n{root_vars}\n}}\n\n" if state.css_variables else ""
    css_content = root_block + "\n\n".join(state.css_fragments)

    js_content = "\n\n".join(state.js_fragments) if state.js_fragments else "// no JS generated"

    return html_content, css_content, js_content


def _write_files(project_id: str, html: str, css: str, js: str) -> None:
    project_dir = os.path.join(WORK_DIR, project_id)
    os.makedirs(project_dir, exist_ok=True)
    with open(os.path.join(project_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    with open(os.path.join(project_dir, "styles.css"), "w", encoding="utf-8") as f:
        f.write(css)
    with open(os.path.join(project_dir, "script.js"), "w", encoding="utf-8") as f:
        f.write(js)


# ---------------------------------------------------------------------------
# MODE A — one description -> planned, generated site (structured)
# ---------------------------------------------------------------------------

class SiteGenerationRequest(BaseModel):
    description: str


class SiteGenerationResponse(BaseModel):
    project_id: str
    sections_generated: list[str]
    html: str
    css: str
    js: str


@app.post("/generate-site", response_model=SiteGenerationResponse)
def generate_site(req: SiteGenerationRequest):
    if not req.description.strip():
        raise HTTPException(status_code=400, detail="Description must not be empty.")

    project_id = str(uuid.uuid4())
    state = ProjectState()
    PROJECTS[project_id] = state

    try:
        plan_result = planning_chain.invoke({"description": req.description})
    except Exception as error:
        logger.exception("Structured planning failed")
        raise HTTPException(status_code=502, detail=f"Website planning failed: {error}")

    sections = plan_result.sections[:MAX_SECTIONS]

    if not sections:
        raise HTTPException(status_code=502, detail="The model returned an empty section plan.")

    # Built ONCE, from the full plan — this is what lets the navbar (generated
    # first) already know the ids of sections that don't exist yet.
    full_site_plan = "\n".join(f"- {s.section_type}: {s.content_brief}" for s in sections)

    for section in sections:
        previous_context = build_capped_context(state)

        try:
            result = section_chain.invoke({
                "section_type": section.section_type,
                "content_brief": section.content_brief,
                "full_site_plan": full_site_plan,
                "existing_css_variables": state.css_variables or "(none yet)",
                "previous_context": previous_context,
            })
        except Exception as error:
            logger.exception("Structured generation failed for section %s", section.section_type)
            raise HTTPException(
                status_code=502,
                detail=f"Generation failed for section '{section.section_type}': {error}",
            )

        if not result.html.strip():
            raise HTTPException(
                status_code=502,
                detail=f"Model returned empty HTML for section '{section.section_type}'.",
            )

        state.html_fragments.append(result.html)
        if result.css.strip():
            state.css_fragments.append(result.css)
        if result.js.strip():
            state.js_fragments.append(result.js)
        state.css_variables.update(result.new_css_variables or {})

        state.section_order.append(section.section_type)

    html_content, css_content, js_content = assemble_site(state)
    _write_files(project_id, html_content, css_content, js_content)

    return SiteGenerationResponse(
        project_id=project_id,
        sections_generated=state.section_order,
        html=html_content,
        css=css_content,
        js=js_content,
    )


# ---------------------------------------------------------------------------
# MODE B — manual, one element at a time (structured)
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    project_id: str
    element_type: str  # "html" | "css" | "js"
    instruction: str
    existing_code: str | None = None


class GenerateResponse(BaseModel):
    project_id: str
    element_type: str
    code: str


@app.post("/session/start")
def start_session():
    project_id = str(uuid.uuid4())
    PROJECTS[project_id] = ProjectState()
    return {"project_id": project_id}


@app.get("/session/{project_id}/state")
def get_state(project_id: str):
    state = PROJECTS.get(project_id)
    if not state:
        raise HTTPException(status_code=404, detail="Project not found.")
    return state


@app.post("/generate-element", response_model=GenerateResponse)
def generate_element(req: GenerateRequest):
    state = PROJECTS.get(req.project_id)
    if not state:
        raise HTTPException(status_code=404, detail="Project not found. Call /session/start first.")

    if req.element_type not in ("html", "css", "js"):
        raise HTTPException(status_code=400, detail="element_type must be 'html', 'css', or 'js'.")

    if not req.instruction.strip():
        raise HTTPException(status_code=400, detail="Instruction must not be empty.")

    previous_context = build_capped_context(state)

    try:
        result = element_chain.invoke({
            "element_type": req.element_type,
            "instruction": req.instruction,
            "existing_code": req.existing_code or "(none provided)",
            "existing_css_variables": state.css_variables or "(none yet)",
            "previous_context": previous_context,
        })
    except Exception as error:
        logger.exception("Structured element generation failed")
        raise HTTPException(status_code=502, detail=f"Generation failed: {error}")

    if not result.code.strip():
        raise HTTPException(status_code=502, detail="Model returned empty code.")

    if req.element_type == "html":
        state.html_fragments.append(result.code)
    elif req.element_type == "css":
        state.css_fragments.append(result.code)
    else:
        state.js_fragments.append(result.code)
    state.css_variables.update(result.new_css_variables or {})

    return GenerateResponse(
        project_id=req.project_id,
        element_type=req.element_type,
        code=result.code,
    )


# ---------------------------------------------------------------------------
# Finalize (zip download)
# ---------------------------------------------------------------------------

@app.post("/finalize/{project_id}")
def finalize_site(project_id: str):
    state = PROJECTS.get(project_id)
    if state and state.html_fragments:
        html_content, css_content, js_content = assemble_site(state)
        _write_files(project_id, html_content, css_content, js_content)

    project_dir = os.path.join(WORK_DIR, project_id)
    if not os.path.exists(os.path.join(project_dir, "index.html")):
        raise HTTPException(status_code=400, detail="Nothing to finalize yet — call /generate-site first.")

    zip_path = os.path.join(WORK_DIR, f"{project_id}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename in ("index.html", "styles.css", "script.js"):
            zf.write(os.path.join(project_dir, filename), arcname=filename)

    return FileResponse(zip_path, filename="generated_website.zip", media_type="application/zip")