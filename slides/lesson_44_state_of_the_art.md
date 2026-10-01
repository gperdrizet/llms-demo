---
marp: true
theme: default
paginate: true
style: |
  section {
    background-color: #1a1a2e;
    color: #e0e0e0;
  }
  section h1, section h2, section h3 {
    color: #e2b55a;
  }
  section a {
    color: #93c5fd;
  }
  section strong {
    color: #f0f0f0;
  }
  section table th {
    background-color: #2a2a4a;
    color: #e2b55a;
  }
  section table td {
    background-color: #1e1e36;
    color: #e0e0e0;
  }
  section code {
    background-color: #2a2a4a;
    color: #e0e0e0;
  }
  section pre {
    background-color: #12122a;
  }
  section::after {
    color: #888;
  }
---

# Lesson 44: State of the art in generative AI

**The landscape beyond text, and the LLM state of the art**

---

## Recap: how LLMs work

- **Decoder-only transformers**: self-attention and feed-forward layers
- **Tokenization:** text is split into tokens (subwords), not characters or words
- **Autoregressive generation:** LLMs predict the next token given all previous tokens
- **Training:** learned from massive corpora via self-supervised learning
- **Scale:** billions of parameters encode language patterns and knowledge

We covered transformers in depth earlier; today we focus on the **landscape** of what's available and where the field is heading.

---

## Today's outline

**Part 1 · The landscape beyond text**
- Modalities and the architectures behind them
- One dated snapshot of the landscape (the "what's real today" table)
- Per-modality highlights: image, video, audio
- What changed in 2026

**Part 2 · The LLM state of the art**
- Frontier + open-weight models
- Architecture themes and economics
- The benchmark shift: how we measure capability now

> Conventions: concepts stay evergreen and are **not** dated. All dated facts (model names, prices, benchmark scores, dates) live in the single dated landscape table, footnoted for re-verification.

---

# Part 1 · The landscape beyond text

Generative AI is no longer just text-in, text-out

---

## Beyond LLMs: the modality picture

A large language model is only one point in a broader landscape. Frontier systems now **emit and consume** a range of modalities:

- **Text**: the LLM core (reasoning, code, agents)
- **Image**: create and edit photorealistic images
- **Video**: cinematic clips, increasingly *with audio*; interactive 3D worlds
- **Audio**: music and speech, increasingly natural and real-time
- **Worlds / 3D**: explorable, interactive environments ("world models")

The durable skill is to map a task to the **right modality + architecture**, not to memorize a model name that will be stale next quarter.

---

## The modality map

| Modality | What it does | Representative direction |
|----------|--------------|--------------------------|
| **Text / LLM** | Reason, code, use tools, drive agents | Agentic, long-horizon tasks |
| **Image** | Generate and *edit* detailed images | Diffusion + fine-tunable models |
| **Video** | Cinematic clips **with audio**; interactive worlds | Video+audio generation; explorable 3D worlds |
| **Audio** | Music and speech, real-time voice | Emotive TTS; real-time conversational voice |
| **Worlds / 3D** | Interactive, explorable environments | "World models" you can step into |

Each modality has its own set of models, architectures, and cost structures.

---

## Architectures (the evergreen menu)

Underneath the modality names, a few architecture families recur:

- **Autoregressive**: predict the next token/unit; the LLM workhorse. Scales to very long contexts; now extended to image and video *tokens*.
- **Diffusion**: iteratively denoise to generate; the dominant engine for image (and increasingly video and audio).
- **Mixture of Experts (MoE)**: many total parameters, but only a subset is active per token → large capacity at lower per-token compute.
- **World models**: models of environments you can *interact with* (interactive 3D, steerable "world action" models).

**Takeaway:** the same four ideas (autoregressive, diffusion, MoE, world models) get recombined across every modality.

---

## The dated landscape: LLM frontier (1 of 3)

| Model (vendor) | Tier | Key facts |
|----------------|------|-----------|
| **GPT-6 Astra** (OpenAI) | Flagship | $10/$50 per MTok (in/out) · 1.05M context, 128K max output · knowledge cutoff Apr 30 2026 · reasoning effort low→max · tools: functions, web/file search, computer use |
| **GPT-6.1 Sol** (OpenAI) | Near-flagship | $2/$10 per MTok · cached input $0.10/MTok · released Sep 29 2026 · "≈Astra intelligence at ~1/5 the price" · ≈matches Astra on DeepSWE v1.1 |
| **GPT-6 Luna** (OpenAI) | Cost / high-volume | $0.10/$0.50 per MTok · knowledge cutoff May 18 2026 |

---

| Model (vendor) | Tier | Key facts |
|----------------|------|-----------|
| **GPT-5.6 Cyber / GPT-Rosalind** (OpenAI) | Specialized | Cybersecurity (Cyber + Daybreak Red/Blue) and life-sciences (Rosalind) lines: a specialized product pattern |
| **Claude Opus 5.5** (Anthropic) | Flagship | Released Sep 22 2026 · complex open-ended work · ≈66.4% Terminal-Bench 4.0 |
| **Claude Sonnet 5.5** (Anthropic) | Mid | $2/$10 per MTok · cache reads $0.20/MTok · released Sep 28 2026 · 30%+ faster, up to 30% cheaper per task vs Sonnet 5 · ≈70.6% vs ≈10.3% (Sonnet 5) on Terminal-Bench 4.0 · first Sonnet with cyber safeguards + fallbacks |
| **Claude Haiku 5.5** (Anthropic) | Cost / high-volume | Announced "coming within weeks" |
| **Gemini 3.7 Flash / 3.8 Live** (Google) | Fast / real-time | Flash = fast tier (drives Antigravity agentic coding); 3.8 Live = real-time voice; **Gemini Omni** = stated omni-modal direction |
| **Grok 4.7** (xAI) | Flagship | 500K context · $2/$6 per MTok · long-context tier (≥200K) billed $4/$12 |

> *Verified 2026-09-30; re-verify before reusing. Benchmark scores are vendor-reported (≈) and drift week to week.*

---

## The dated landscape: open weights (2 of 3)

| Org (current line) | Key facts |
|--------------------|-----------|
| **Meta (Llama 4)** | Natively multimodal (text+image), mixture-of-experts; Llama Guard / Prompt Guard safety families alongside |
| **DeepSeek (DeepSeek-V4 / V4.1-Flash)** | V4-Flash-Base 292B (Apr 27); V4.1-Flash focuses on KV-cache compression |
| **Moonshot (Kimi K3)** | Image-text-to-text; "Open Frontier Intelligence" paper (Jul 27); ≈2.8T total MoE params (approx, confirm framing) |
| **Alibaba (Qwen3 family)** | Qwen3Guard (safety); Qwen-Image (20B) + Qwen-Image-Edit; docs at qwen.ai |
---
| Org (current line) | Key facts |
|--------------------|-----------|
| **Google (Gemma)** | Open-model line from DeepMind |
| **Mistral (Medium 3.5 / Small 4 / OCR 4 / Voxtral TTS)** | €3B Series D at >€21B valuation (Sep 8 2026); explicit "sovereign, open-weight AI" strategy; Mozilla partnership → open multilingual AI in Firefox (Sep 16); products: Studio, Forge, Vibe, Vibe for code, AI Cloud |

> *Verified 2026-09-30; re-verify before reusing.*

---

## The dated landscape: image, video, audio (3 of 3)

| Modality | Model (vendor) | Key facts |
|----------|----------------|-----------|
| **Image** | **Nano Banana / Gemini Image** (Google) | "Create and edit detailed images" |
| **Image** | **FLUX 3 / flux-3-action** (Black Forest Labs) | Fine-tunable *world-action* model (~Sep 23 2026); FLUX.2 [klein] LoRA fine-tune in <60 min |
| **Image** | **GPT-Image-2.5** (OpenAI) | Sunburst (most capable) / Flare (fast, everyday) |
| **Image** | **Qwen-Image / Qwen-Image-Edit** (Alibaba) | 20B |
| **Image** | **grok-imagine** (xAI) | xAI image line |
| **Video** | **Veo** (Google) | "Generate cinematic video **with audio**" |
| **Video** | **Genie 3** (Google) | Interactive 3D world generation / exploration (world-model direction) |
| **Video** | **Sora** (OpenAI) | **DISCONTINUED**: web/app Apr 26 2026, API Sep 24 2026 (cautionary example only) |
| **Audio** | **Lyria** (Google) | Music generation |
| **Audio** | **Gemini Audio / 3.8 Live / 3.5 Transcribe** (Google) | "Talk, create, control audio"; real-time voice; STT |
| **Audio** | **Voxtral TTS** (Mistral) | Speech synthesis |
| **Audio** | **Eleven v4** (ElevenLabs) | "Most emotive model yet" (recent release) |
| **Audio** | **GPT-Live 1 / GPT-Realtime-2.1 / GPT-Transcribe** (OpenAI) | Voice, real-time, STT |

> *Verified 2026-09-30; re-verify before reusing. Several third-party video/music lines were not re-verified this pass and are intentionally omitted.*

---

## Image: mature, competitive, and editable

The image field has moved from "generate a picture" to **create *and* edit**, with real fine-tuning workflows:

- **Creation + editing** are now first-class, not one-shot generation.
- **Fine-tuning is a real workflow**: e.g., LoRA fine-tunes completing in well under an hour on some lines.
- **"World action" models** (fine-tunable, steerable image models) are an emerging direction.

Exemplars: Google **Nano Banana**, Black Forest Labs **FLUX 3**, OpenAI **GPT-Image-2.5**, Alibaba **Qwen-Image**. *(Specs in the dated landscape table.)*

---

## Video: audio is in, worlds are the frontier

Two shifts define video generation:

- **Video comes with audio**: "cinematic video with audio" is the bar (e.g., Google **Veo**).
- **Interactive worlds**: explorable, 3D, world-model direction (e.g., Google **Genie 3**), not just passive clips.

**Cautionary example:** **Sora** (OpenAI) was **discontinued**: a reminder of how fast this space turns. Don't build a thesis on a single model name. *(Dates in the dated landscape table.)*

---

## Audio: speech is converging with the LLM

Audio splits into **music** and **speech**, with speech increasingly a first-class LLM interface:

- **Music** generation (e.g., Google **Lyria**).
- **Speech / TTS** getting more natural and emotive (e.g., **Eleven v4**, Mistral **Voxtral TTS**).
- **Real-time voice**: low-latency conversational voice is now a product line of its own (e.g., Google **Gemini 3.8 Live**, OpenAI **GPT-Live**).

**Takeaway:** voice is becoming a primary *interface*, not a bolt-on.

---

## What changed in 2026 (the durable themes)

- **Agentic is the frontier**: coding, computer use, and multi-step business workflows replace Q&A as the benchmark currency.
- **Cost stratification**: near-flagship performance at 1/3–1/5 the price is the norm; cached-input pricing is common.
- **Reasoning-effort tiers**: same model, different capability/cost dials (low→max) as a practical dial.
- **Native multimodality**: text+image input is table stakes; omni-modal is the stated direction.
- **Video shake-up**: Sora exit; Veo (video+audio) and Genie 3 (interactive worlds) define the push.
- **Specialized categories + safety as features**: cyber, life sciences, OCR, TTS as distinct offerings; safety safeguards and fallback behaviors are built in.
- **Open-weight frontier + "sovereign AI"**: large open models rival closed ones; some vendors (Mistral, for example) explicitly target national or regional deployment.

---

# Part 2 · The LLM state of the art

How to think about the text frontier (concepts); exact figures in the dated table

---

## Frontier closed models: a product family, not a single model

The "frontier" is now a **tiered family** per vendor, plus specialized lines:

| Tier | Role | Exemplars |
|------|------|-----------|
| **Flagship** | Maximum capability, highest cost | GPT-6 Astra, Claude Opus 5.5, Grok 4.7, Gemini 3.x |
| **Near-flagship** | ≈flagship at a fraction of the price | GPT-6.1 Sol, Claude Sonnet 5.5, Gemini Flash |
| **Cost / high-volume** | Cheap, fast, high throughput | GPT-6 Luna, Claude Haiku 5.5 |
| **Specialized** | Domain-specific product lines | Cyber (GPT-5.6 Cyber, Daybreak), life sciences (GPT-Rosalind), OCR, TTS |

**Plus:** **reasoning-effort tiers** (low→max): the *same* model at different capability/cost settings. *(Prices and scores: see the dated landscape table.)*

---

## Open weights: the frontier is open too

Large open-weight models now rival closed models on many tasks:

- **Llama 4, DeepSeek V4, Kimi K3, Qwen3, Gemma, Mistral** (Medium 3.5 / Small 4) form the current open frontier.
- **MoE** is common: big total capacity, lower active compute per token.
- **Native multimodality** in open models too (e.g., Llama 4, Kimi K3: text+image input).
- **"Sovereign AI"** positioning is growing (notably Mistral: open-weight models marketed for national deployment).

**Trade-off:** open weights = control, privacy, fine-tuning, self-hosting, at the cost of your own infrastructure and ops. *(Param counts and specs: see the dated landscape table.)*

---

## Architecture themes (evergreen)

Four themes define the current LLM frontier:

- **Mixture of Experts (MoE)**: many parameters, sparse activation → capacity without proportional per-token cost.
- **Reasoning-effort tiers**: a *product* dimension (low→max) more than a distinct architecture: one model, multiple cost/capability settings.
- **~1M-token context windows**: now common at the frontier; long-context tiers are often priced separately.
- **Native multimodality**: text+image input is table stakes; omni-modal (any-in → any-out) is the stated direction.

**Takeaway:** capability is increasingly a *configuration* (model × effort × context), not a fixed point.

---

## Economics: capability is priced per task

- **Per-token pricing** (input vs output) is the unit; capability is now measured **at a cost**.
- **Near-flagship at 1/3–1/5 price** is the norm: choose the tier that clears your bar, not the most expensive one.
- **Cached-input pricing** is common: repeated/shared context can be dramatically cheaper than standard input.
- **Match the dial:** model tier × reasoning effort × context length, against your task's budget.

**Takeaway:** optimize for *cost-per-acceptable-task*, not raw benchmark rank.

---

## The benchmark shift: how we measure capability now

Static multiple-choice benchmarks (MMLU and friends) are **saturated**: no longer *the* story. Frontier evaluation is now **long-horizon, agentic, real-environment**:

| Area | Representative evals |
|------|----------------------|
| Agentic coding | Terminal-Bench 4, DeepSWE, FrontierCode, CursorBench |
| Professional / knowledge work | GDPval, AutomationBench, GDP.pdf |
| Reasoning | Humanity's Last Exam |
| Computer use | OSWorld 2 |
| Tool use / preference | Toolathlon, LMArena |

**Teaching takeaway:** capability is measured **per-task, per-effort-level, at a cost**: not as a single score. The old MMLU-style "one number to rule them all" is gone.

---

## Limitations and considerations

- **Hallucinations:** LLMs are probabilistic generators, not databases of facts.
- **Reasoning gaps:** still struggle with some multi-step logic, math, and planning.
- **Cost:** API usage is metered and scales with tokens, context, and reasoning effort.
- **Privacy:** closed models send data to third-party servers; open weights don't (but cost you ops).
- **Staleness:** model names and scores go stale fast; re-verify the dated table before relying on it.
- **Ethics:** training-data bias, copyright, and environmental impact remain live concerns.

---

## Summary

- **Part 1, the landscape beyond text:** modality map, evergreen architectures (autoregressive, diffusion, MoE, world models), one dated landscape table, and per-modality highlights (image, video, audio).
- **Part 2, the LLM state of the art:** frontier and open-weight models, architecture themes (MoE, reasoning-effort tiers, ~1M context, native multimodality), per-task economics with cached input, and the shift to agentic, long-horizon benchmarks.
- **Sora is discontinued**: a cautionary example of how fast the landscape turns; never present it as a current offering.
- Always test on your own tasks; benchmarks and hype don't tell the full story.

---

## Next lesson

**Lesson 45: Practical LLM deployment**
- Decoding strategies and hyperparameters
- Quantization for running models locally
- Local hosting, cloud hosting, and API access
- Performance and cost tradeoffs
