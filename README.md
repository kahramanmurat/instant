# Instant2

AI-powered content generator built with FastAPI and OpenAI, deployed on Vercel.

Generate welcome messages, poems, haikus, jokes, fun facts, and motivational quotes -- with full control over tone, length, language, and model.

**Live:** [instant2-roan.vercel.app](https://instant2-roan.vercel.app)

## Features

- **6 content types:** welcome, poem, haiku, joke, facts, motivational
- **6 tones:** enthusiastic, professional, sarcastic, pirate, shakespearean, gen_z
- **3 lengths:** short, medium, long
- **7 languages:** English, Spanish, French, German, Japanese, Portuguese, Turkish
- **5 OpenAI models:** GPT-5.4 nano, GPT-5.4 mini, GPT-5.4, o1, o3-mini
- **Compare mode:** run all models side-by-side on the same prompt
- **Custom topic and audience** via free-text inputs
- **Token stats and latency** displayed per request
- **Structured error handling** with retryable/non-retryable error cards

## Project Structure

```
instant2/
  instant2.py        # FastAPI app -- single-file, all routes, prompts, and HTML
  requirements.txt   # Python dependencies (fastapi, uvicorn, openai)
  vercel.json        # Vercel deployment config (Python runtime, catch-all route)
  .env               # OPENAI_API_KEY (not committed)
  .gitignore         # Excludes .vercel/
```

## How It Works

1. User selects content type, tone, length, language, and model via the UI
2. `build_prompt()` assembles a system prompt from the selected options
3. `call_model()` sends the prompt to OpenAI with full error handling
4. The response (or error card) is rendered as inline HTML and returned

All UI is server-rendered HTML -- no JavaScript framework, no build step.

## Setup

### Prerequisites

- Python 3.12+
- An [OpenAI API key](https://platform.openai.com/api-keys)

### Local Development

```bash
# Clone
git clone https://github.com/kahramanmurat/instant.git
cd instant

# Create .env
echo "OPENAI_API_KEY=sk-..." > .env

# Install and run
pip install -r requirements.txt
uvicorn instant2:app --reload
```

Open [http://localhost:8000](http://localhost:8000).

### Deploy to Vercel

```bash
# Install Vercel CLI
npm i -g vercel

# Link and deploy
vercel link
vercel env add OPENAI_API_KEY
vercel --prod
```

## Query Parameters

All controls are URL query parameters, so you can bookmark or share specific configurations:

| Parameter  | Values | Default |
|------------|--------|---------|
| `type`     | `welcome`, `poem`, `haiku`, `joke`, `facts`, `motivational` | `welcome` |
| `tone`     | `enthusiastic`, `professional`, `sarcastic`, `pirate`, `shakespearean`, `gen_z` | `enthusiastic` |
| `length`   | `short`, `medium`, `long` | `medium` |
| `language` | `English`, `Spanish`, `French`, `German`, `Japanese`, `Portuguese`, `Turkish` | `English` |
| `model`    | `gpt-5.4-nano`, `gpt-5.4-mini`, `gpt-5.4`, `o1`, `o3-mini` | `gpt-5.4-nano` |
| `compare`  | `true`, `false` | `false` |
| `topic`    | Free text (max 120 chars) | none |
| `audience` | Free text (max 120 chars) | none |

Example: `/?type=haiku&tone=pirate&model=gpt-4o&language=Japanese`

## License

MIT
