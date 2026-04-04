from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError, AuthenticationError, BadRequestError
from openai import APIStatusError
from enum import Enum
from typing import Optional
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()


# ── Enums ──────────────────────────────────────────────────────────────────────

class ContentType(str, Enum):
    welcome      = "welcome"
    poem         = "poem"
    haiku        = "haiku"
    joke         = "joke"
    facts        = "facts"
    motivational = "motivational"

class Tone(str, Enum):
    enthusiastic  = "enthusiastic"
    professional  = "professional"
    sarcastic     = "sarcastic"
    pirate        = "pirate"
    shakespearean = "shakespearean"
    gen_z         = "gen_z"

class Length(str, Enum):
    short  = "short"
    medium = "medium"
    long   = "long"

class Language(str, Enum):
    english    = "English"
    spanish    = "Spanish"
    french     = "French"
    german     = "German"
    japanese   = "Japanese"
    portuguese = "Portuguese"
    turkish    = "Turkish"

class Model(str, Enum):
    gpt_4o_mini = "gpt-4o-mini"
    gpt_4o      = "gpt-4o"
    gpt_4_turbo = "gpt-4-turbo"
    gpt_o1      = "o1"
    gpt_o3_mini = "o3-mini"

MODEL_META = {
    Model.gpt_4o_mini: {"label": "GPT-4o mini",  "badge": "⚡ Fast & cheap", "color": "#22d3ee", "note": "Best for quick tasks. Low latency, low cost."},
    Model.gpt_4o:      {"label": "GPT-4o",        "badge": "🌟 Flagship",    "color": "#818cf8", "note": "Multimodal flagship. Strong reasoning and instruction following."},
    Model.gpt_4_turbo: {"label": "GPT-4 Turbo",   "badge": "🔬 128k ctx",   "color": "#fb923c", "note": "Large context window. Good for long documents."},
    Model.gpt_o1:      {"label": "o1",              "badge": "🧠 Reasoning",  "color": "#34d399", "note": "Full reasoning model. Thinks before it answers."},
    Model.gpt_o3_mini: {"label": "o3-mini",        "badge": "🚀 Latest",     "color": "#f472b6", "note": "Newest reasoning model. Highest accuracy on hard tasks."},
}

REASONING_MODELS = {Model.gpt_o1, Model.gpt_o3_mini}


# ── Error types ────────────────────────────────────────────────────────────────

class APIError:
    """Structured error returned instead of a reply on failure."""
    def __init__(self, kind: str, message: str, suggestion: str, retryable: bool = False, status_code: int = None):
        self.kind        = kind
        self.message     = message
        self.suggestion  = suggestion
        self.retryable   = retryable
        self.status_code = status_code

    def to_html(self, accent: str) -> str:
        retry_badge = (
            '<span style="background:#fbbf2422;color:#fbbf24;border-radius:99px;'
            'padding:2px 8px;font-size:0.7rem;margin-left:8px;">⟳ retryable</span>'
            if self.retryable else ""
        )
        status_bit = f' <span style="color:#475569;">(HTTP {self.status_code})</span>' if self.status_code else ""
        return f"""
        <div style="background:#1e1010;border:1px solid #ef444466;border-radius:12px;
                    padding:1.5rem 2rem;max-width:680px;width:100%;">
          <div style="display:flex;align-items:center;margin-bottom:0.75rem;">
            <span style="color:#ef4444;font-weight:700;font-size:1rem;">
              ⚠ {self.kind}{status_bit}
            </span>
            {retry_badge}
          </div>
          <p style="color:#fca5a5;margin:0 0 0.75rem;line-height:1.6;">{self.message}</p>
          <p style="color:#94a3b8;font-size:0.85rem;margin:0;">
            💡 {self.suggestion}
          </p>
        </div>"""


# ── Prompt helpers ─────────────────────────────────────────────────────────────

BASE_PROMPTS = {
    ContentType.welcome:      "Write an announcement welcoming visitors to a web app that just launched in production.",
    ContentType.poem:         "Write a poem about a web app going live in production.",
    ContentType.haiku:        "Write a haiku (5-7-5) about deploying software to production. Output only the haiku.",
    ContentType.joke:         "Tell a clever, clean joke that a software developer would appreciate. Output only the joke.",
    ContentType.facts:        "Share surprising little-known facts about the history of the internet or web development.",
    ContentType.motivational: "Write a motivational message for a developer who just shipped their first production deployment.",
}
LENGTH_INSTRUCTIONS = {
    Length.short:  "Be very brief — 1 to 2 sentences or lines maximum.",
    Length.medium: "Keep it moderate — a short paragraph or 4–6 lines.",
    Length.long:   "Be expansive and detailed — at least 3 paragraphs or 10+ lines.",
}
TONE_INSTRUCTIONS = {
    Tone.enthusiastic:   "Use an upbeat, energetic, exclamation-filled tone.",
    Tone.professional:   "Use a calm, formal, professional tone.",
    Tone.sarcastic:      "Use heavy sarcasm and dry wit throughout.",
    Tone.pirate:         "Write entirely in pirate speak (arr, matey, ye, etc.).",
    Tone.shakespearean:  "Write in the style of Shakespeare — archaic English, flowery language.",
    Tone.gen_z:          "Write in Gen Z slang — lowercased, casual, lots of 'fr', 'no cap', 'bussin', etc.",
}


def build_prompt(content_type, tone, length, language, topic, audience) -> str:
    base = BASE_PROMPTS[content_type]
    if topic:
        base = f"{base} The subject or theme to focus on is: {topic}."
    parts = [base, TONE_INSTRUCTIONS[tone], LENGTH_INSTRUCTIONS[length]]
    if audience:
        parts.append(f"Tailor the content specifically for this audience: {audience}.")
    if language != Language.english:
        parts.append(f"Write your entire response in {language.value}.")
    return " ".join(parts)


# ── API call with full error handling ─────────────────────────────────────────

def call_model(client: OpenAI, model: Model, prompt: str) -> tuple[str | APIError, float, int, int]:
    """
    Returns (reply_or_error, latency, prompt_tokens, completion_tokens).
    On failure, reply_or_error is an APIError instance; tokens will be 0.
    """
    kwargs = dict(model=model.value, messages=[{"role": "user", "content": prompt}])
    if model not in REASONING_MODELS:
        kwargs["temperature"] = 0.8

    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(**kwargs)
        latency = time.perf_counter() - t0
        logger.info("OK  model=%s latency=%.2fs tokens=%d+%d",
                    model.value, latency, resp.usage.prompt_tokens, resp.usage.completion_tokens)
        return resp.choices[0].message.content, latency, resp.usage.prompt_tokens, resp.usage.completion_tokens

    except AuthenticationError as e:
        latency = time.perf_counter() - t0
        logger.error("AUTH model=%s status=%s", model.value, e.status_code)
        return APIError(
            kind="Authentication Failed",
            message="Your OpenAI API key is missing, revoked, or invalid.",
            suggestion="Check that OPENAI_API_KEY is set correctly in your environment and hasn't expired.",
            status_code=e.status_code,
        ), latency, 0, 0

    except RateLimitError as e:
        latency = time.perf_counter() - t0
        logger.warning("RATE_LIMIT model=%s status=%s", model.value, e.status_code)
        return APIError(
            kind="Rate Limit Exceeded",
            message="You've hit OpenAI's request or token rate limit for this model.",
            suggestion="Wait a moment and try again, or switch to a model with a higher quota (e.g. gpt-4o-mini).",
            retryable=True,
            status_code=e.status_code,
        ), latency, 0, 0

    except APITimeoutError:
        latency = time.perf_counter() - t0
        logger.warning("TIMEOUT model=%s after=%.2fs", model.value, latency)
        return APIError(
            kind="Request Timed Out",
            message=f"The model didn't respond within the timeout window ({latency:.1f}s elapsed).",
            suggestion="Try a shorter length or a faster model like gpt-4o-mini. Reasoning models (o1/o3) are slower by design.",
            retryable=True,
        ), latency, 0, 0

    except APIConnectionError as e:
        latency = time.perf_counter() - t0
        logger.error("CONNECTION model=%s err=%s", model.value, e)
        return APIError(
            kind="Connection Error",
            message="Could not reach the OpenAI API. This is usually a network issue.",
            suggestion="Check your internet connection, firewall rules, or proxy settings. OpenAI's status page at status.openai.com may show an outage.",
            retryable=True,
        ), latency, 0, 0

    except BadRequestError as e:
        latency = time.perf_counter() - t0
        logger.error("BAD_REQUEST model=%s status=%s body=%s", model.value, e.status_code, e.body)
        return APIError(
            kind="Bad Request",
            message=f"OpenAI rejected the request: {e.message}",
            suggestion="This model may not support the chosen parameters. o1/o3 models don't allow system messages or temperature. Try a different model.",
            status_code=e.status_code,
        ), latency, 0, 0

    except APIStatusError as e:
        # Catch-all for 5xx server errors and anything else with an HTTP status
        latency = time.perf_counter() - t0
        logger.error("API_STATUS model=%s status=%s", model.value, e.status_code)
        return APIError(
            kind=f"API Error",
            message=f"OpenAI returned an unexpected error: {e.message}",
            suggestion="This is likely a temporary issue on OpenAI's side. Check status.openai.com and retry shortly.",
            retryable=e.status_code >= 500,
            status_code=e.status_code,
        ), latency, 0, 0

    except Exception as e:
        latency = time.perf_counter() - t0
        logger.exception("UNEXPECTED model=%s err=%s", model.value, e)
        return APIError(
            kind="Unexpected Error",
            message=f"An unhandled error occurred: {type(e).__name__}: {e}",
            suggestion="This is a bug. Check the server logs for a full traceback.",
        ), latency, 0, 0


# ── Visual config ──────────────────────────────────────────────────────────────

CONTENT_STYLES = {
    ContentType.welcome:      ("🚀 We're Live!",          "#0f172a", "#38bdf8"),
    ContentType.poem:         ("📜 A Poem for the Deploy", "#1a1a2e", "#c084fc"),
    ContentType.haiku:        ("🌸 Haiku",                 "#0d1b2a", "#6ee7b7"),
    ContentType.joke:         ("😄 Dev Joke",              "#1c1917", "#fbbf24"),
    ContentType.facts:        ("🧠 Did You Know?",         "#0f172a", "#f472b6"),
    ContentType.motivational: ("💪 You Did It!",           "#14532d", "#86efac"),
}


# ── Route ──────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def generate_content(
    type:     ContentType   = Query(default=ContentType.welcome,   description="Type of content"),
    tone:     Tone          = Query(default=Tone.enthusiastic,     description="Writing tone"),
    length:   Length        = Query(default=Length.medium,         description="Response length"),
    language: Language      = Query(default=Language.english,      description="Output language"),
    model:    Model         = Query(default=Model.gpt_4o_mini,     description="OpenAI model to use"),
    compare:  bool          = Query(default=False,                 description="Run all models side-by-side"),
    topic:    Optional[str] = Query(default=None, max_length=120,  description="Custom topic/subject"),
    audience: Optional[str] = Query(default=None, max_length=120,  description="Target audience"),
):
    client = OpenAI()
    prompt = build_prompt(type, tone, length, language, topic, audience)
    title, bg, accent = CONTENT_STYLES[type]

    def pill_links(enum_cls, param, current):
        base_params = f"type={type.value}&tone={tone.value}&length={length.value}&language={language.value}&model={model.value}&compare={str(compare).lower()}"
        return " ".join(
            f'<a href="?{base_params.replace(f"{param}={current.value}", f"{param}={v.value}")}"'
            f' style="padding:4px 10px;border-radius:99px;'
            f'background:{""+accent+"33" if v==current else "rgba(255,255,255,0.07)"};'
            f'color:{accent};text-decoration:none;font-size:0.8rem;'
            f'border:1px solid {""+accent if v==current else "transparent"};">'
            f'{v.value}</a>'
            for v in enum_cls
        )

    def stat_chip(label, value):
        return (f'<span style="background:rgba(255,255,255,0.07);border-radius:6px;'
                f'padding:2px 8px;font-size:0.75rem;color:#94a3b8;">'
                f'{label}: <b style="color:#f1f5f9;">{value}</b></span>')

    def render_result(result, latency, p_tok, c_tok, m: Model, is_primary=False):
        meta   = MODEL_META[m]
        border = meta["color"] if is_primary else "rgba(255,255,255,0.1)"
        if isinstance(result, APIError):
            body = result.to_html(accent)
        else:
            body = f'<div style="line-height:1.8;font-size:0.95rem;">{result.replace(chr(10), "<br/>")}</div>'

        chips = "".join([
            stat_chip("⏱", f"{latency:.2f}s"),
            "" if isinstance(result, APIError) else stat_chip("↑", str(p_tok)),
            "" if isinstance(result, APIError) else stat_chip("↓", str(c_tok)),
        ])
        return f"""
        <div style="background:rgba(255,255,255,0.04);border:1px solid {border};
                    border-radius:12px;padding:1.5rem;flex:1;min-width:260px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">
            <span style="color:{meta['color']};font-weight:700;">{meta['label']}</span>
            <span style="background:{meta['color']}22;color:{meta['color']};border-radius:99px;
                         padding:2px 8px;font-size:0.7rem;">{meta['badge']}</span>
          </div>
          <p style="color:#64748b;font-size:0.75rem;margin:0 0 1rem;">{meta['note']}</p>
          {body}
          <div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:1rem;">{chips}</div>
        </div>"""

    # ── single model ───────────────────────────────────────────────────────────
    if not compare:
        result, latency, p_tok, c_tok = call_model(client, model, prompt)
        meta = MODEL_META[model]

        if isinstance(result, APIError):
            content_html = result.to_html(accent)
        else:
            content_html = f'<div class="card">{result.replace(chr(10), "<br/>")}</div>'

        header_chips = "".join([
            stat_chip("⏱", f"{latency:.2f}s"),
            *([] if isinstance(result, APIError) else [stat_chip("↑ in", str(p_tok)), stat_chip("↓ out", str(c_tok))]),
        ])

        content_section = f"""
        <div style="margin-bottom:0.75rem;display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
          <span style="color:{meta['color']};font-weight:700;">{meta['label']}</span>
          <span style="background:{meta['color']}22;color:{meta['color']};border-radius:99px;
                       padding:2px 8px;font-size:0.7rem;">{meta['badge']}</span>
          {header_chips}
        </div>
        {content_html}"""
        compare_toggle = f'<a href="?type={type.value}&tone={tone.value}&length={length.value}&language={language.value}&model={model.value}&compare=true" style="color:{accent};font-size:0.85rem;">⚖️ Compare all models</a>'

    # ── side-by-side compare ───────────────────────────────────────────────────
    else:
        cards = [render_result(*call_model(client, m, prompt), m, m == model) for m in Model]
        content_section = f"""
        <div style="display:flex;flex-wrap:wrap;gap:1rem;width:100%;max-width:1100px;margin-bottom:1.5rem;">
          {"".join(cards)}
        </div>"""
        compare_toggle = f'<a href="?type={type.value}&tone={tone.value}&length={length.value}&language={language.value}&model={model.value}&compare=false" style="color:{accent};font-size:0.85rem;">◀ Single model view</a>'

    model_pills = " ".join(
        f'<a href="?type={type.value}&tone={tone.value}&length={length.value}&language={language.value}&model={m.value}&compare={str(compare).lower()}"'
        f' style="padding:5px 12px;border-radius:99px;display:inline-flex;align-items:center;gap:5px;'
        f'background:{""+MODEL_META[m]["color"]+"33" if m==model else "rgba(255,255,255,0.06)"};'
        f'color:{MODEL_META[m]["color"]};text-decoration:none;font-size:0.8rem;'
        f'border:1px solid {""+MODEL_META[m]["color"] if m==model else "transparent"};">'
        f'{MODEL_META[m]["badge"]} {MODEL_META[m]["label"]}</a>'
        for m in Model
    )

    html = f"""
<html>
<head>
  <title>{title}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      background: {bg};
      color: #f1f5f9;
      font-family: 'Segoe UI', sans-serif;
      min-height: 100vh;
      margin: 0;
      padding: 2rem 1rem;
      display: flex;
      flex-direction: column;
      align-items: center;
    }}
    h1 {{ color: {accent}; font-size: 2rem; margin-bottom: 0.25rem; text-align:center; }}
    .subtitle {{ color: #94a3b8; font-size: 0.85rem; margin-bottom: 1.5rem; text-align:center; }}
    .card {{
      background: rgba(255,255,255,0.05);
      border: 1px solid {accent}44;
      border-radius: 12px;
      padding: 2rem 2.5rem;
      max-width: 680px;
      width: 100%;
      line-height: 1.8;
      font-size: 1.05rem;
      margin-bottom: 1.5rem;
    }}
    .controls {{ max-width:680px;width:100%;display:flex;flex-direction:column;gap:0.75rem;margin-bottom:1rem; }}
    .control-row {{ display:flex;flex-wrap:wrap;align-items:center;gap:6px; }}
    .control-label {{ color:#94a3b8;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;width:72px;flex-shrink:0; }}
    .freeform input {{
      background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.15);
      border-radius:8px;color:#f1f5f9;font-size:0.85rem;padding:4px 10px;outline:none;width:200px;
    }}
    .freeform input:focus {{ border-color:{accent}; }}
    .go-btn {{ background:{accent};color:{bg};border:none;border-radius:8px;padding:5px 14px;font-size:0.85rem;font-weight:600;cursor:pointer; }}
    .model-row {{ max-width:1100px;width:100%;display:flex;flex-wrap:wrap;gap:8px;margin-bottom:1.5rem; }}
    .prompt-debug {{
      max-width:680px;width:100%;background:rgba(0,0,0,0.3);border-radius:8px;
      padding:0.75rem 1rem;font-size:0.75rem;color:#64748b;margin-top:0.5rem;
    }}
    .prompt-debug summary {{ cursor:pointer;color:#475569; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p class="subtitle">model: <b>{MODEL_META[model]['label']}</b> &nbsp;·&nbsp;
     tone: <b>{tone}</b> &nbsp;·&nbsp; length: <b>{length}</b> &nbsp;·&nbsp; language: <b>{language}</b></p>

  <div class="model-row">{model_pills}</div>
  <div style="margin-bottom:1.5rem;">{compare_toggle}</div>

  {content_section}

  <form class="controls" method="get">
    <div class="control-row"><span class="control-label">Type</span>{pill_links(ContentType,"type",type)}</div>
    <div class="control-row"><span class="control-label">Tone</span>{pill_links(Tone,"tone",tone)}</div>
    <div class="control-row"><span class="control-label">Length</span>{pill_links(Length,"length",length)}</div>
    <div class="control-row"><span class="control-label">Language</span>{pill_links(Language,"language",language)}</div>
    <div class="control-row freeform">
      <span class="control-label">Topic</span>
      <input type="hidden" name="type"     value="{type.value}">
      <input type="hidden" name="tone"     value="{tone.value}">
      <input type="hidden" name="length"   value="{length.value}">
      <input type="hidden" name="language" value="{language.value}">
      <input type="hidden" name="model"    value="{model.value}">
      <input type="hidden" name="compare"  value="{str(compare).lower()}">
      <input name="topic"    placeholder="e.g. machine learning, cats…" value="{topic or ''}">
      <input name="audience" placeholder="e.g. junior devs, executives…" value="{audience or ''}">
      <button class="go-btn" type="submit">Go ↩</button>
    </div>
  </form>

  <details class="prompt-debug">
    <summary>🔍 View generated prompt</summary><br/>{prompt}
  </details>
</body>
</html>
"""
    return html