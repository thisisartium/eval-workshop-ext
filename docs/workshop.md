# Eval Workshop: Building Evals for a Helpdesk Agent

A hands-on workshop covering the eval loop: error analysis, baselines, failure analysis, and prompt iteration.

**Time:** ~60 minutes

---

## Prerequisites

Before the workshop starts, you need an **Anthropic API key** ([get one here](https://console.anthropic.com/settings/keys)) or an **OpenAI API key** ([get one here](https://platform.openai.com/api-keys)).

### Option A: GitHub Codespaces (recommended)

Click the Codespaces button in the repo README — a fully configured environment opens in your browser. Then:

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY (or OPENAI_API_KEY)

# Test the agent
uv run helpdesk-agent "My laptop won't turn on"
# Should show: Department: IT
```

### Option B: VS Code Dev Container

1. Clone the repo and open in VS Code
2. When prompted, click **"Reopen in Container"**
3. Wait for the container to build (first time takes a few minutes)

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY (or OPENAI_API_KEY)

# Test the agent
uv run helpdesk-agent "My laptop won't turn on"
```

### Option C: Local Setup

Requires Python 3.13+, [uv](https://docs.astral.sh/uv/) (`brew install uv`), and Docker.

```bash
# 1. Clone and install
git clone <repo-url>
cd eval-workshop-ext
uv sync

# 2. Set up environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY (https://console.anthropic.com/settings/keys)
# Or OPENAI_API_KEY if using OpenAI (https://platform.openai.com/api-keys)

# 3. Start services (pull latest images first)
docker compose pull && docker compose up -d

# 4. Verify services are running
docker compose ps
# Should show cat-cafe as running/healthy

# 5. Test the agent
uv run helpdesk-agent "My laptop won't turn on"
# Should show: Department: IT
```

---

## The Problem

You're building an AI assistant for a company's internal helpdesk. Employees send requests like:

- "My laptop won't connect to WiFi"
- "How do I submit an expense report?"
- "The conference room projector is broken"

Your agent needs to:

1. **Route requests** to the right department (IT, HR, Facilities, Finance, Legal, Security)
2. **Answer questions** when possible, using company policy documents
3. **Escalate to humans** when it can't help

### The Architecture

```
                         ┌─────────────────┐
                         │  User Request   │
                         └────────┬────────┘
                                  │
                         ┌────────▼────────┐
                         │    Concierge    │
                         │  (front-line)   │
                         └────────┬────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
     ┌────────▼────────┐ ┌───────▼───────┐ ┌────────▼────────┐
     │  HR Specialist  │ │ IT Specialist │ │    Escalate     │
     │  (has KB)       │ │ (future)      │ │    to Human     │
     └────────┬────────┘ └───────────────┘ └─────────────────┘
              │
     ┌────────▼────────┐
     │  Return answer  │
     │  to concierge   │
     └─────────────────┘
```

The concierge is like a customer service rep: it talks to the employee, consults internal specialists for domain-specific answers, and decides whether to relay the answer or escalate to a human team.

### The Challenge

How do you know if your agent is working? You need **evals** — a systematic way to:

- Find failures before users do
- Measure if changes actually help
- Catch regressions when you update prompts

Today we'll build evals for this helpdesk agent, from error analysis to LLM-as-judge.

---

# Part A: Routing

**The capability:** Can our agent route requests to the correct department?

This is the first capability we're building. Before the agent can answer questions or take actions, it needs to understand what kind of request it's dealing with and send it to the right place.

Let's find out if it works.

---

## Part 1: Error Analysis

**Goal:** Understand how the agent behaves before measuring it.

### Run the Agent

We start with the baseline config — routing-only (no specialists). Requests get classified and escalated but not answered.

Try these requests and predict the department before seeing the result:

```bash
uv run helpdesk-agent -c configs/baseline.yaml "My laptop won't turn on"
```

<details>
<summary>What department did you predict?</summary>

**IT** — This is a clear hardware issue.

</details>

```bash
uv run helpdesk-agent -c configs/baseline.yaml "How do I expense my home office chair?"
```

<details>
<summary>What department did you predict?</summary>

This is ambiguous:

- **Finance** — It's about expenses/money
- **HR** — Employee reimbursements are often HR
- **Facilities** — It's office equipment

Your organization decides. That's why labeling guidelines matter.

</details>

```bash
uv run helpdesk-agent -c configs/baseline.yaml "The conference room projector is broken"
```

<details>
<summary>What department did you predict?</summary>

Another ambiguous one:

- **IT** — It's technology
- **Facilities** — It's building equipment

</details>

```bash
uv run helpdesk-agent -c configs/baseline.yaml "My badge doesn't work at the printer"
```

<details>
<summary>What department did you predict?</summary>

Tricky:

- **Security** — Badge/access issues
- **IT** — Printer pairing problem

The specific context matters: "at the printer" suggests IT (device pairing), while "at the door" suggests Security (access control).

</details>

```bash
uv run helpdesk-agent -c configs/baseline.yaml "Thanks for your help last week!"
```

<details>
<summary>What department did you predict?</summary>

**None** — This isn't a helpdesk request at all.

</details>

### View Traces

Open CAT Cafe to see what happened inside the agent:

**http://localhost:8000/traces**

Click on any trace to see:

- The input message
- The `escalate_to_department` tool call with the chosen department
- The final response

### Key Insight

> Before we can evaluate the agent, we need to decide what "correct" means. Error analysis reveals the ambiguous cases that need explicit rules.

---

## Part 2: Building a Dataset

**Goal:** Understand where eval datasets come from.

### The Problem

In Part 1, we tested 5 requests. That's not enough to evaluate an agent. We need hundreds of examples with known correct answers.

Where do we get them?

### Two Sources of Data

**Real data** — Label actual requests from your organization. This is your ground truth. You collect raw requests, then annotate them with the correct department using tools like Label Studio or spreadsheets.

**Synthetic data** — Generate more examples to fill gaps. Use an LLM to create realistic requests across all departments and topics, controlling the distribution so rare categories get coverage too.

### Register the Workshop Dataset

For this workshop, we've pre-built a small 11-example dataset that includes key confusions (HR vs Finance, IT vs Security):

```bash
uv run python scripts/register_dataset.py data/live_workshop_routing.jsonl --name live_routing
```

View it: **http://localhost:8000/datasets**

### Key Insight

> Real data gives you ground truth. Synthetic data gives you coverage. You need both.

---

## Part 3: Routing Baseline

**Goal:** Establish baseline metrics so we can measure improvement.

### The Prompt

We start with a simple baseline prompt — don't overthink it yet. Get something working, then let the evals tell you what to fix.

Look at the baseline routing prompt ([prompts/routing/baseline.txt](../prompts/routing/baseline.txt)):

```bash
cat prompts/routing/baseline.txt
```

**Before running the eval:** What accuracy do you predict? (0-100%)

### Run the Experiment

```bash
uv run cat-experiments run experiments/routing.py \
  -c experiments/baseline.yaml \
  --dataset live_routing
```

### View Results

Open CAT Cafe: **http://localhost:8000/datasets**

Click on `live_routing`, then the **Experiments** tab. Click on your experiment (`routing-baseline`) to see:

- Overall accuracy: **\_**%
- How close was your prediction?

> **Note:** LLM outputs are non-deterministic. Your results may differ slightly from the examples shown in this document. That's expected — focus on the patterns, not the exact numbers.

### Filter to See Failures

In the filter box, enter:

```
evaluations.department_match.score == 0
```

This shows only the misclassified examples. For each failure, you can see:

- The input request
- Expected department (from dataset)
- Actual department (from agent)
- The evaluator's explanation

**Which two departments are getting confused? Why might that be?**

<details>
<summary>What we typically see</summary>

With the baseline prompt, HR requests about expense reimbursement get misclassified as Finance. The prompt doesn't clarify that employee reimbursements are HR's responsibility.

</details>

### Analyze with the CLI

For a quick summary with precision/recall and confusion matrix:

```bash
uv run python scripts/analyze_routing_experiment.py --target cat-cafe
```

This shows:

- **Precision/Recall by department** — Which departments are we over/under-predicting?
- **Confusion matrix** — Which departments get confused with each other?

Example output:

```
Experiment: routing-baseline
Total Examples: 11 | Routing Accuracy: 81.8%

            Precision/Recall by Department
  Department   Precision   Recall       F1   Support
  IT              100.0%   100.0%   100.0%         3
  HR              100.0%    33.3%    50.0%         3  ← Low recall!
  Finance          33.3%   100.0%    50.0%         1  ← Low precision!
  ...

                 Confusion Matrix
  True \ Pred   IT   HR   Finance   ...
  IT             3    0         0
  HR             0    1         2   ← HR misclassified as Finance
  Finance        0    0         1
```

**What does this tell us?** HR has low recall (33%) because 2 of 3 HR examples were predicted as Finance. These are the expense reimbursement cases.

### Key Insight

> Filtering to failures is the fastest way to understand what's broken. The confusion matrix tells you exactly which categories are confused.

---

## Part 4: Improving Routing

**Goal:** Use eval results to make targeted improvements.

The analysis told us exactly what's broken:

- HR examples are being misclassified as Finance
- Specifically: expense reimbursement requests

Now we fix it.

### How to Fix Prompts

When you find a failure pattern, resist the urge to add specific examples for each failure — that's overfitting. Instead, look for the **general rule** that's missing.

Common techniques:

- **Clarify definitions** — "HR handles employee reimbursements, Finance handles vendor payments"
- **Add disambiguation rules** — "If the request mentions X, route to Y"
- **Reorder or emphasize** — Put the most confused categories closer together with clearer distinctions

For our HR vs Finance confusion, the fix is a clearer definition, not "expense reports go to HR."

### Step 1: Create Your Fix

Copy the baseline prompt and create your own version:

```bash
cp prompts/routing/baseline.txt prompts/routing/my_fix.txt
```

Edit it with your fix:

```bash
# Use your preferred editor
nano prompts/routing/my_fix.txt
# or
code prompts/routing/my_fix.txt
```

<details>
<summary>Hint: What to change</summary>

The v1 prompt doesn't say who handles expense reimbursements. Finance sounds right for "expenses," but employee reimbursements are typically HR.

Try adding something like:

- "HR handles employee expense reimbursements"
- "Finance handles vendor payments, not employee reimbursements"

</details>

### Step 2: Run Your Experiment

Test your fix:

```bash
uv run cat-experiments run experiments/routing.py \
  -c experiments/baseline.yaml \
  --dataset live_routing \
  --param prompt_file=prompts/routing/my_fix.txt
```

### Step 3: Check Your Results

```bash
uv run python scripts/analyze_routing_experiment.py --target cat-cafe
```

> **Tip:** You can also compare experiments visually in CAT Cafe at **http://localhost:8000/datasets** (click the dataset, then the Experiments tab)

**Did it work?**

- Did HR recall improve?
- Did you break anything else?

| Metric           | v1 (baseline) | Your fix | Change |
| ---------------- | ------------- | -------- | ------ |
| Overall accuracy | 81.8%         | \_\_\_%  |        |
| HR recall        | 33.3%         | \_\_\_%  |        |

### Compare to Our Fix

See what we changed in the tuned version:

```bash
diff prompts/routing/baseline.txt prompts/routing/tuned.txt
```

Key changes in the tuned prompt:

- Reframed as a "concierge" that consults specialists and relays answers
- Added detailed department descriptions with responsibilities
- Added NOTE to Finance: "Employee expense reimbursements go to HR"
- Clarified Security is physical security ONLY; information security goes to IT
- Added explicit disambiguation rules section
- Added specialist consultation workflow (consult -> relay or escalate)

Run the tuned config to compare:

```bash
uv run cat-experiments run experiments/routing.py \
  -c experiments/tuned.yaml \
  --dataset live_routing
```

```bash
uv run python scripts/analyze_routing_experiment.py --target cat-cafe
```

### Key Insight

> One change, one comparison, one decision. The eval tells you if your fix worked — or if you broke something else.

---

# Part B: Answering Questions

**The capability:** Can our agent answer HR questions correctly using a knowledge base?

This is the second capability. Once requests are routed to HR, the specialist agent needs to look up relevant policies and provide accurate answers — or know when to escalate to a human.

This requires a different kind of eval than routing. Let's explore why.

---

## Part 5: HR Specialist + LLM-as-Judge

**Goal:** Evaluate response quality, not just routing accuracy.

### The Architecture Change

So far we've only routed requests. Now we enable the HR specialist.

Compare routing-only vs with specialist:

```bash
# Baseline: routing only - just classifies and escalates
uv run helpdesk-agent -c configs/baseline.yaml "How many vacation days do I get?"
```

```bash
# With HR specialist - concierge consults specialist, relays answer
uv run helpdesk-agent -c configs/with_specialists.yaml "How many vacation days do I get?"
```

Notice the difference:

- **Routing only**: "I've forwarded your request to HR..."
- **With specialist**: Actually answers the question from the knowledge base

```
User Request -> Concierge -> HR Specialist -> Return answer
                                  |            to concierge
                          Search Knowledge Base
```

The concierge consults the specialist, which can:

- Search the HR knowledge base
- Return answers from policy documents
- Report when it can't find relevant info

The concierge then decides to relay the answer or escalate to a human team.

### The Knowledge Base

View the HR policies:

```bash
ls kb/hr/
```

```bash
head -30 kb/hr/time_off.md
```

### Exercise: Be the Judge

For each example, decide: **PASS** or **FAIL**?

**Example 1:** Pass or fail?

```
User: "How many vacation days do I get?"
KB: "15 days for 0-2 years, 20 days for 3-5 years..."
Agent: "You get 30 days of vacation starting immediately."
```

<details>
<summary>Answer</summary>

**FAIL** — Hallucination. The agent made up "30 days" when the KB says 15-25 depending on tenure.

</details>

**Example 2:** What about this one?

```
User: "How many vacation days do I get?"
KB: "15 days for 0-2 years, 20 days for 3-5 years..."
Agent: "Based on our policy, you get 15-25 days depending on tenure."
```

<details>
<summary>Answer</summary>

**PASS** — Grounded in KB content and addresses the question.

</details>

**Example 3:** The KB doesn't have the answer. What should the agent do?

```
User: "What's the company's stance on moonlighting?"
KB: (no relevant policies found)
Agent: "I'll connect you with HR to discuss our moonlighting policy."
```

<details>
<summary>Answer</summary>

**PASS** — Correct escalation. The KB doesn't cover this topic, so escalating is the right call.

</details>

**Example 4:** Tricky one. Pass or fail?

```
User: "How do I enroll in the 401k?"
KB: "401k enrollment is available through the benefits portal..."
Agent: "I'll escalate this to HR for you."
```

<details>
<summary>Answer</summary>

**FAIL** — Over-cautious. The KB had the answer but the agent escalated anyway.

</details>

### The Evaluator

The `hr_response_quality` evaluator makes a single pass/fail judgment with a critique:

**PASS if:**

- Agent answered AND answer is grounded in KB AND addresses the question
- Agent escalated AND KB doesn't have the needed information

**FAIL if:**

- Hallucination (answer contradicts or goes beyond KB)
- Irrelevant (doesn't address the question)
- Over-cautious (escalated when KB had the answer)
- Over-confident (answered when KB didn't have sufficient info)

### Run the Experiment

First, register the small HR dataset:

```bash
uv run python scripts/register_dataset.py data/live_workshop_hr.jsonl \
  --name live_hr
```

Then run the experiment:

```bash
uv run cat-experiments run experiments/hr_specialist.py \
  -c experiments/with_specialists.yaml \
  --dataset live_hr
```

### View Results

Open CAT Cafe: **http://localhost:8000/datasets**

Click on `live_hr`, then the **Experiments** tab. Click on your experiment (`hr-with-specialists`). Unlike routing where we used precision/recall, the HR specialist uses an **LLM-as-judge** evaluator that gives pass/fail with a critique.

Click into any example to see:

- The user's question
- The agent's response
- The `hr_response_quality` evaluation with **pass/fail** and **critique**

Click into a failure and read the critique. It tells you exactly what went wrong — far more useful than a numeric score.

### Key Insight

> Pass/fail with critique beats metric soup. "Groundedness: 0.3" tells you nothing. "Response contradicts KB — KB says 15 days, agent said 30" tells you everything.

---

## Part 6: Improving the HR Specialist

**Goal:** Use pass/fail critiques to improve HR specialist responses.

The analysis from Part 5 shows what's broken. Now we fix it — same process as routing, but with different failure modes.

### HR Specialist Failure Modes

Unlike routing (wrong department), the HR specialist can fail in several ways:

| Failure Mode   | Description                         | Example Critique                         |
| -------------- | ----------------------------------- | ---------------------------------------- |
| Hallucination  | Answer includes info not in KB      | "Response says 30 days, KB says 15"      |
| Irrelevant     | Answer doesn't address the question | "User asked about 401k, got dental info" |
| Over-cautious  | Escalated when KB had the answer    | "KB clearly covers this topic"           |
| Over-confident | Answered when KB didn't have info   | "Should have escalated"                  |

### Step 1: Review Failures

In CAT Cafe, filter to failures:

```
evaluations.hr_response_quality.score == 0
```

Read the critiques. What pattern do you see?

<details>
<summary>Common patterns with baseline prompt</summary>

The baseline prompt is minimal:

- No guidance on what topics the KB covers
- No instructions about citing sources
- No examples of good responses

This leads to:

- Vague answers that don't cite specific policies
- Occasional hallucinations when the agent guesses
- Inconsistent response format

</details>

### Step 2: Create Your Fix

Copy the baseline specialist prompt and create your own version:

```bash
cp prompts/specialists/hr/with_specialists.txt prompts/specialists/hr/my_fix.txt
```

Edit it with your fix:

```bash
# Use your preferred editor
nano prompts/specialists/hr/my_fix.txt
# or
code prompts/specialists/hr/my_fix.txt
```

<details>
<summary>Hint: What to change</summary>

The baseline prompt lacks:

- Clear instructions to ONLY use KB content
- Guidance on what to do when the answer isn't in the KB
- Expected response format
- Examples of good responses

Try adding:

- "ONLY include information from the knowledge base"
- "Cite the source document when providing policy information"
- A few-shot example of a good response

</details>

### Step 3: Run Your Experiment

To test your fix, create a copy of the HR agent config that points to your prompt:

```bash
cp configs/agents/hr_with_specialists.yaml configs/agents/hr_my_fix.yaml
```

Edit it to use your prompt:

```bash
# Change prompt_file to point to your fix
# prompt_file: "prompts/specialists/hr/my_fix.txt"
```

Run the experiment with your agent config:

```bash
uv run cat-experiments run experiments/hr_specialist.py \
  -c experiments/with_specialists.yaml \
  --dataset live_hr \
  --param agent_config=configs/agents/hr_my_fix.yaml
```

### Step 4: Check Your Results

View results in CAT Cafe: **http://localhost:8000/datasets**

Click on `live_hr` -> **Experiments** tab -> your new experiment.

**Did it work?**

- Did pass rate improve?
- Are the critiques different?
- Did you introduce any new failure modes?

| Metric    | v1      | Your fix | Change |
| --------- | ------- | -------- | ------ |
| Pass rate | \_\_\_% | \_\_\_%  |        |

### Compare to Our Fix

See what we changed in the tuned version:

```bash
diff prompts/specialists/hr/with_specialists.txt prompts/specialists/hr/tuned.txt
```

Key changes in the tuned prompt:

- Reframed as an "internal knowledge consultant" (specialist doesn't talk to users directly)
- Added explicit process: always search KB first, return what you find
- Added list of topics the KB covers
- Added citation instructions
- Added three worked examples (full answer, partial answer, no answer)
- Clear instruction to state what it couldn't answer and why

Run the tuned config to compare:

```bash
uv run cat-experiments run experiments/hr_specialist.py \
  -c experiments/hr_tuned.yaml \
  --dataset live_hr
```

### Key Insight

> The critique tells you exactly what to fix. "KB says 15 days, agent said 30" -> add grounding instructions. "Should have escalated" -> clarify when to escalate. Let the failures guide the improvements.

---

## Wrap-up

### The Eval Loop

You've now seen the complete cycle:

```
Error Analysis -> Build Dataset -> Run Baseline
                                        |
                    Promote <- Compare <- Improve
```

### Principles to Remember

```
1. Error analysis first — observe before you measure
2. Every eval targets a failure mode
3. One change, one comparison, one decision
4. Pass/fail + critique beats metric soup
5. If you can't explain the eval, you can't trust it
```

### Resources

- [Hamel's LLM Evals FAQ](https://hamel.dev/blog/posts/evals-faq/)
- [AI Evals Course](https://maven.com/parlance-labs/evals)

---

## Quick Reference

### Commands

```bash
# Run agent (baseline - routing only, escalation only)
uv run helpdesk-agent -c configs/baseline.yaml "your request here"

# Run agent with HR specialist (untuned)
uv run helpdesk-agent -c configs/with_specialists.yaml "your HR question"

# Run agent with HR specialist (tuned)
uv run helpdesk-agent -c configs/tuned.yaml "your HR question"

# Register workshop datasets
uv run python scripts/register_dataset.py data/live_workshop_routing.jsonl --name live_routing
uv run python scripts/register_dataset.py data/live_workshop_hr.jsonl --name live_hr

# Run routing baseline (11 examples, ~30 seconds)
uv run cat-experiments run experiments/routing.py \
  -c experiments/baseline.yaml \
  --dataset live_routing

# Run routing with tuned prompts
uv run cat-experiments run experiments/routing.py \
  -c experiments/tuned.yaml \
  --dataset live_routing

# Analyze experiment results
uv run python scripts/analyze_routing_experiment.py --target cat-cafe

# Run HR specialist (6 examples, ~60 seconds)
uv run cat-experiments run experiments/hr_specialist.py \
  -c experiments/with_specialists.yaml \
  --dataset live_hr

# Compare HR specialist prompts
diff prompts/specialists/hr/with_specialists.txt prompts/specialists/hr/tuned.txt

# Run HR specialist with tuned prompts
uv run cat-experiments run experiments/hr_specialist.py \
  -c experiments/hr_tuned.yaml \
  --dataset live_hr

# Run HR specialist with custom agent config
uv run cat-experiments run experiments/hr_specialist.py \
  -c experiments/with_specialists.yaml \
  --dataset live_hr \
  --param agent_config=configs/agents/hr_my_fix.yaml
```

### URLs

| Service                | URL                            |
| ---------------------- | ------------------------------ |
| CAT Cafe               | http://localhost:8000          |
| Traces                 | http://localhost:8000/traces   |
| Datasets & Experiments | http://localhost:8000/datasets |

### Key Files

| File                                      | Description                            |
| ----------------------------------------- | -------------------------------------- |
| `prompts/routing/baseline.txt`            | Baseline routing prompt                |
| `prompts/routing/tuned.txt`               | Tuned routing prompt (concierge)       |
| `prompts/specialists/hr/with_specialists.txt` | Baseline HR specialist prompt      |
| `prompts/specialists/hr/tuned.txt`        | Tuned HR specialist prompt             |
| `configs/baseline.yaml`                   | Baseline system config (no specialists)|
| `configs/with_specialists.yaml`           | With specialists (untuned)             |
| `configs/tuned.yaml`                      | Tuned system config (production)       |
| `kb/hr/`                                  | HR knowledge base documents            |
