"""AI Cover-Letter Generator — FastAPI + бесплатный LLM (Groq).

Вставляешь вакансию + своё резюме → LLM пишет сопроводительное письмо под неё
(EN/RU), подсвечивает совпадающие навыки, при желании вставляет нужное слово,
и не выдумывает фактов сверх резюме.
"""

import json
import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import llm

app = FastAPI(title="AI Cover-Letter Generator")

HTML = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")

SYSTEM = (
    "You are a professional career assistant that writes concise, tailored cover letters. "
    "You never invent facts, skills or experience not supported by the candidate's CV. "
    "You output STRICT JSON only, with no markdown fences."
)


class GenReq(BaseModel):
    vacancy: str
    cv: str
    lang: str = "en"
    magic_word: str = ""


def _prompt(r: GenReq) -> str:
    lang = "Russian" if r.lang == "ru" else "English"
    magic = (f"\nIMPORTANT: include this exact word naturally somewhere in the letter: "
             f"\"{r.magic_word.strip()}\".") if r.magic_word.strip() else ""
    return (
        f"Write a cover letter in {lang}, based ONLY on the candidate's CV.\n"
        f"Length 150-220 words, professional, specific to the role, highlight overlapping "
        f"skills, no fabrication.{magic}\n\n"
        f"CANDIDATE CV:\n{r.cv.strip()}\n\n"
        f"JOB POSTING:\n{r.vacancy.strip()}\n\n"
        'Return STRICT JSON: {"letter": "<full cover letter text>", '
        '"match_summary": "<1-2 sentences on why the candidate fits>"}'
    )


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.post("/api/generate")
def generate(r: GenReq):
    if not r.vacancy.strip() or not r.cv.strip():
        return {"error": "Заполни и вакансию, и резюме."}
    if not llm.available():
        return {"error": "Нет GROQ_API_KEY в .env"}
    try:
        content, usage = llm.chat(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": _prompt(r)}],
            max_tokens=1400, temperature=0.5,
        )
        m = re.search(r"\{.*\}", content, re.S)
        data = json.loads(m.group(0)) if m else {"letter": content, "match_summary": ""}
        return {
            "letter": data.get("letter", "").strip(),
            "match_summary": data.get("match_summary", "").strip(),
            "tokens": usage.get("total_tokens"),
        }
    except Exception as e:
        return {"error": f"Ошибка генерации: {e}"}
