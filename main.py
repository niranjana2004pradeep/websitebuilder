
import os
import re
import uuid
import zipfile
import logging
 
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
 
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
 
load_dotenv()
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("website-gen-unstructured")
 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL","openai/gpt-oss-120b")
 
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY must be set in the environment (.env). Get a free key at console.groq.com")
 
# No response_format / json mode / schema of any kind — the model can
# respond however it likes. This is the UNSTRUCTURED half of the comparison.
llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.3)
 
app = FastAPI(title="AI Website Generation Pipeline — Unstructured Output (LangChain)")
 
WORK_DIR = "/tmp/website_gen_unstructured"
os.makedirs(WORK_DIR, exist_ok=True)
 
MAX_SECTIONS = 5
 
 
# ---------------------------------------------------------------------------
# In-memory project state
#
# IMPORTANT DIFFERENCE from the structured version: without JSON output, we
# CANNOT reliably extract "just the new CSS variables" or "just the new class
# names" from a model's free-text response — there's no guaranteed field to
# read. So instead of tracking clean structured deltas, this version falls
# back to the naive approach: keep the full raw text of everything generated
# so far, and paste it all back into the next prompt as context. This is
# exactly the expensive, unreliable fallback we deliberately avoided in the
# structured version — kept here on purpose so the contrast is visible.
# ---------------------------------------------------------------------------
 
PROJECTS: dict[str, "ProjectState"] = {}
 
 
class ProjectState(BaseModel):
    raw_fragments: list[str] = Field(default_factory=list)  # full raw model output, in order
    section_order: list[str] = Field(default_factory=list)
 
 
# ---------------------------------------------------------------------------
# Naive extraction helpers (regex-based — fragile by nature, on purpose)
# ---------------------------------------------------------------------------
 
def extract_code_block(text: str, language: str) -> str:
    """Best-effort extraction of a ```language ... ``` fenced block.
    Returns an empty string if the model didn't format its response this way
    — which is common, because nothing FORCES it to. This function is the
    manual, error-prone stand-in for what a schema + parser would guarantee."""
    pattern = rf"```{language}\s*(.*?)```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""
 
 
def parse_numbered_sections(text: str) -> list[dict]:
    """Naive parsing of a numbered list like:
         1. navbar: A responsive navbar with...
         2. hero: A bold headline...
    Returns [] if the model didn't follow this format closely enough to match
    — again, nothing enforces that it will."""
    pattern = r"^\s*\d+\.\s*([a-zA-Z_\-]+)\s*[:\-]\s*(.+)$"
    matches = re.findall(pattern, text, re.MULTILINE)
    return [{"section_type": m[0].strip().lower(), "content_brief": m[1].strip()} for m in matches]
 
 
# ---------------------------------------------------------------------------
# LangChain Runnable chains (LCEL: prompt | llm)
# ---------------------------------------------------------------------------
 
PLANNING_SYSTEM_PROMPT = """You are a website architect. A non-technical user will describe a website \
they want. Break their description into an ordered list of sections needed for a complete website.
 
Always list a "navbar" first and a "footer" last. Respond as a numbered plain-text list, ONE section \
per line, in exactly this format (nothing else, no extra commentary):
 
1. section_type: content brief for that section
2. section_type: content brief for that section
...
"""
 
planning_prompt = ChatPromptTemplate.from_messages([
    ("system", PLANNING_SYSTEM_PROMPT),
    ("human", "{description}"),
])
planning_chain = planning_prompt | llm  # Runnable composition (LCEL)
 
 
SECTION_SYSTEM_PROMPT = """You are an expert front-end developer. Generate ONE section of a website: \
its HTML, its CSS, and JS only if genuinely needed.
 
Context — everything generated so far in this project (for consistency; reuse existing colors, class \
names, and function names where relevant instead of inventing new ones):
{previous_context}
 
Present your answer using fenced code blocks labeled by language, like this:
 
```html
<!-- the section's HTML -->
```
 
```css
/* the section's CSS */
```
 
```js
// JS for this section, omit this block entirely if none is needed
```
 
You may briefly explain your choices before or after the code blocks if you want to.
"""
 
section_prompt = ChatPromptTemplate.from_messages([
    ("system", SECTION_SYSTEM_PROMPT),
    ("human", "Section to generate: {section_type}\n\nContent brief: {content_brief}"),
])
section_chain = section_prompt | llm  # Runnable composition (LCEL)
 
 
ELEMENT_SYSTEM_PROMPT = """You are an expert front-end developer. Generate a single {element_type} \
element for a website based on the instruction below.
 
Context — everything generated so far in this project (for consistency):
{previous_context}
 
Present your answer in a single fenced code block labeled with the language, e.g.:
```{element_type}
...code...
```
You may briefly explain your choices before or after the code block if you want to.
"""
 
element_prompt = ChatPromptTemplate.from_messages([
    ("system", ELEMENT_SYSTEM_PROMPT),
    ("human", "Instruction: {instruction}\n\nExisting code to modify/extend (if any): {existing_code}"),
])
element_chain = element_prompt | llm  # Runnable composition (LCEL)
 
 
# ---------------------------------------------------------------------------
# MODE A — one description -> planned, generated site (unstructured)
# ---------------------------------------------------------------------------
 
class SiteGenerationRequest(BaseModel):
    description: str
 
 
class SiteGenerationResponse(BaseModel):
    project_id: str
    sections_generated: list[str]
    raw_model_outputs: list[str]  # exposed deliberately so you can SEE the messiness
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
 
    plan_result = planning_chain.invoke({"description": req.description})
    plan_text = plan_result.content
    sections = parse_numbered_sections(plan_text)
 
    if not sections:
        # This is the unstructured failure mode made visible: the model's
        # text didn't match our expected numbered-list pattern, so there is
        # NOTHING reliable to extract. A schema would have prevented this.
        raise HTTPException(
            status_code=502,
            detail=f"Could not parse a section plan from the model's response. Raw output was: {plan_text[:500]}",
        )
 
    sections = sections[:MAX_SECTIONS]
 
    html_fragments, css_fragments, js_fragments = [], [], []
 
    for section in sections:
        previous_context = "\n\n---\n\n".join(state.raw_fragments) if state.raw_fragments else "(nothing generated yet)"
 
        result = section_chain.invoke({
            "section_type": section["section_type"],
            "content_brief": section["content_brief"],
            "previous_context": previous_context,
        })
        raw_text = result.content
        state.raw_fragments.append(raw_text)
        state.section_order.append(section["section_type"])
 
        html_part = extract_code_block(raw_text, "html")
        css_part = extract_code_block(raw_text, "css")
        js_part = extract_code_block(raw_text, "js")
 
        if html_part:
            html_fragments.append(html_part)
        else:
            logger.warning(f"No HTML block found for section '{section['section_type']}' — model output didn't match expected format.")
        if css_part:
            css_fragments.append(css_part)
        if js_part:
            js_fragments.append(js_part)
 
    html_content = _assemble_html(html_fragments)
    css_content = "\n\n".join(css_fragments)
    js_content = "\n\n".join(js_fragments) if js_fragments else "// no JS generated"
 
    _write_files(project_id, html_content, css_content, js_content)
 
    return SiteGenerationResponse(
        project_id=project_id,
        sections_generated=state.section_order,
        raw_model_outputs=state.raw_fragments,
        html=html_content,
        css=css_content,
        js=js_content,
    )
 
 
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
# MODE B — manual, one element at a time (unstructured)
# ---------------------------------------------------------------------------
 
class GenerateRequest(BaseModel):
    project_id: str
    element_type: str  # "html" | "css" | "js"
    instruction: str
    existing_code: str | None = None
 
 
class GenerateResponse(BaseModel):
    project_id: str
    element_type: str
    raw_model_output: str   # the full, unstructured response — shown as-is
    extracted_code: str     # best-effort regex extraction from it
 
 
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
 
    previous_context = "\n\n---\n\n".join(state.raw_fragments) if state.raw_fragments else "(nothing generated yet)"
 
    result = element_chain.invoke({
        "element_type": req.element_type,
        "instruction": req.instruction,
        "existing_code": req.existing_code or "(none provided)",
        "previous_context": previous_context,
    })
    raw_text = result.content
    extracted = extract_code_block(raw_text, req.element_type)
 
    state.raw_fragments.append(raw_text)
 
    return GenerateResponse(
        project_id=req.project_id,
        element_type=req.element_type,
        raw_model_output=raw_text,
        extracted_code=extracted,
    )
 
 
# ---------------------------------------------------------------------------
# Finalize (zip download)
# ---------------------------------------------------------------------------
 
@app.post("/finalize/{project_id}")
def finalize_site(project_id: str):
    project_dir = os.path.join(WORK_DIR, project_id)
    if not os.path.exists(os.path.join(project_dir, "index.html")):
        raise HTTPException(status_code=400, detail="Nothing to finalize yet — call /generate-site first.")
 
    zip_path = os.path.join(WORK_DIR, f"{project_id}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename in ("index.html", "styles.css", "script.js"):
            zf.write(os.path.join(project_dir, filename), arcname=filename)
 
    return FileResponse(zip_path, filename="generated_website.zip", media_type="application/zip")