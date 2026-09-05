import os
import json
import uuid
import shutil
import zipfile
import logging
 
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from groq import Groq
 
load_dotenv()
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("website-gen")
 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
 
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY must be set in the environment (.env). Get a free key at console.groq.com")
 
client = Groq(api_key=GROQ_API_KEY)
 
app = FastAPI(title="AI Website Generation Pipeline (MVP)")
 
WORK_DIR = "/tmp/website_gen"
os.makedirs(WORK_DIR, exist_ok=True)



PROJECTS: dict[str, "ProjectState"] = {}
 
 
class ProjectState(BaseModel):
    css_variables: dict[str, str] = Field(default_factory=dict)
    classes_used: list[str] = Field(default_factory=list)
    js_functions: list[str] = Field(default_factory=list)
 
    html_fragments: list[str] = Field(default_factory=list)
    css_fragments: list[str] = Field(default_factory=list)
    js_fragments: list[str] = Field(default_factory=list)
 
 
class GenerateRequest(BaseModel):
    project_id: str
    element_type: str  # "html" | "css" | "js"
    instruction: str
    existing_code: str | None = None  # user-supplied code to modify/extend
 
 
class GenerateResponse(BaseModel):
    project_id: str
    element_type: str
    code: str
    design_state: ProjectState
 
 
# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------
 
SYSTEM_PROMPT = """You are an expert front-end developer generating a single element \
of a real, production-quality website. Follow these rules strictly:
 
1. Generate clean, semantic, well-formatted code for ONLY the requested element type.
2. Reuse the existing CSS variables, class/id names, and JS function names given to \
you under "Current design system state" wherever applicable — do NOT invent new \
names for things that already exist. Only introduce new variables/classes/functions \
when the instruction genuinely requires something new.
3. If existing code is provided, modify/extend it according to the instruction \
rather than starting over.
4. Return ONLY a JSON object, no prose, no markdown fences, with EXACTLY these keys:
   {
     "code": "<the generated code as a string>",
     "new_css_variables": {"--example": "value"},
     "new_classes": ["example-class"],
     "new_js_functions": ["exampleFunction"]
   }
   Use empty objects/arrays for any category with nothing new to report.
"""
 
 
def build_user_prompt(req: GenerateRequest, state: ProjectState) -> str:
    return f"""Element type to generate: {req.element_type}
 
Instruction: {req.instruction}
 
Existing code for this element (modify/extend this if provided, else write fresh):
{req.existing_code or "(none provided)"}
 
Current design system state (reuse these, don't duplicate with new names):
- CSS variables already defined: {json.dumps(state.css_variables)}
- Classes/ids already used: {json.dumps(state.classes_used)}
- JS functions already defined: {json.dumps(state.js_functions)}
"""
 
 
def call_model(system_prompt: str, user_prompt: str) -> dict:
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.error(f"Model did not return valid JSON: {raw[:300]}")
        raise HTTPException(status_code=502, detail="Model returned malformed output. Try again.")


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
 
    user_prompt = build_user_prompt(req, state)
 
    try:
        result = call_model(SYSTEM_PROMPT, user_prompt)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Groq call failed: {e}")
        raise HTTPException(status_code=502, detail="Generation failed. Try again.")
 
    code = result.get("code", "")
    if not code.strip():
        raise HTTPException(status_code=502, detail="Model returned empty code.")
 
    # Merge deltas into the running design state so the NEXT call sees them
    state.css_variables.update(result.get("new_css_variables", {}) or {})
    for c in result.get("new_classes", []) or []:
        if c not in state.classes_used:
            state.classes_used.append(c)
    for f in result.get("new_js_functions", []) or []:
        if f not in state.js_functions:
            state.js_functions.append(f)
 
    # Store the fragment itself for final assembly
    if req.element_type == "html":
        state.html_fragments.append(code)
    elif req.element_type == "css":
        state.css_fragments.append(code)
    else:
        state.js_fragments.append(code)
 
    return GenerateResponse(
        project_id=req.project_id,
        element_type=req.element_type,
        code=code,
        design_state=state,
    )
 
 
@app.post("/finalize/{project_id}")
def finalize_site(project_id: str):
    state = PROJECTS.get(project_id)
    if not state:
        raise HTTPException(status_code=404, detail="Project not found.")
 
    if not state.html_fragments:
        raise HTTPException(status_code=400, detail="No HTML elements generated yet — nothing to finalize.")
 
    project_dir = os.path.join(WORK_DIR, project_id)
    os.makedirs(project_dir, exist_ok=True)
 
    # --- styles.css: CSS variables block + all generated CSS fragments ---
    root_vars = "\n".join(f"  {k}: {v};" for k, v in state.css_variables.items())
    css_content = f":root {{\n{root_vars}\n}}\n\n" + "\n\n".join(state.css_fragments)
    with open(os.path.join(project_dir, "styles.css"), "w", encoding="utf-8") as f:
        f.write(css_content)
 
    # --- script.js: all generated JS fragments concatenated ---
    js_content = "\n\n".join(state.js_fragments) if state.js_fragments else "// no JS generated"
    with open(os.path.join(project_dir, "script.js"), "w", encoding="utf-8") as f:
        f.write(js_content)
 
    # --- index.html: skeleton wrapping all generated HTML fragments in order ---
    body_content = "\n\n".join(state.html_fragments)
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Generated Website</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
{body_content}
<script src="script.js"></script>
</body>
</html>
"""
    with open(os.path.join(project_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_content)
 
    # --- zip it up ---
    zip_path = os.path.join(WORK_DIR, f"{project_id}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename in ("index.html", "styles.css", "script.js"):
            zf.write(os.path.join(project_dir, filename), arcname=filename)
 
    return FileResponse(zip_path, filename="generated_website.zip", media_type="application/zip")