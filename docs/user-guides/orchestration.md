# StochasBase: The Simulation Conductor

!!! abstract

    **StochasBase** serves as the central conductor for your simulation lifecycle, synchronizing **Aleatory** randomness with **Epistemic** design choices into a single, deterministic stream. By leveraging "Salted Seed" logic, it ensures that complex Monte Carlo campaigns remain perfectly repeatable across different machines and collaborators. This guide explores the orchestration of model inputs, the mechanics of the "Baked" registry, and the use of manual overrides to isolate variables for high-fidelity debugging and "Golden Case" testing.

---

`StochasBase` is the central "brain" for a simulation trial. It orchestrates the lifecycle of a model by bridging the gap between mathematical uncertainty and deterministic execution.

### Centralized Trial Orchestration

A single instance of `StochasBase` manages [two distinct pillars](https://en.wikipedia.org/wiki/Uncertainty_quantification#Aleatoric_and_epistemic) of simulation inputs:

- **Aleatory Uncertainty (`sample_dist`):** Represents "luck" or noise. These are random draws from probability distributions (Normal, Uniform, etc.) that you cannot control but must account for.
- **Epistemic Uncertainty (`sample_design`):** Represents "choices." These are tunable parameters (Design Variables) used by optimizers like Optuna or pymoo to find peak performance.

### The Pillars of Repeatability

To ensure that a simulation can be perfectly recreated by a colleague, the orchestrator utilizes a "Salted Seed" logic. Every random draw is a deterministic function of the trial's metadata:

$$Seed_{local} = \mathcal{H}(Seed_{global}, Trial_{num}, Variable_{name})$$

- **Global Seed:** Controls the entire campaign.
- **Trial Number:** Ensures trial 10 is different from trial 11.
- **Variable Name:** Ensures "stiffness" doesn't get the same random value as "damping."

---

## The "Baked" Registry (named)

While `StochasBase` is designed to automate randomness and optimization, you often need to bypass the math to test a specific "Golden Case" or debug a known failure point. Overrides allow you to inject fixed values into the Baked Registry before the simulation begins.

The orchestrator follows a strict priority sequence when a variable is requested:

1. **Registry Check:** It looks in the `named` dictionary. If a value exists, it returns it immediately.
2. **Logic Execution:** If the registry is empty, it proceeds to draw from a distribution or ask an optimizer for a suggestion.

By using `with_overrides()`, you populate the registry early, effectively "locking" those variables for the duration of the trial.

---

## Quick Implementation

```python
--8<-- "docs/user-guides/orchestration.py:quick_implementation"
```

---

## Retrying Correlated Draws (transaction)

Every example so far has sampled one value at a time. Sometimes a draw only makes sense *together* with another one, and there's no way to check that from a single distribution alone.

!!! example

    Say you're generating a ring and randomly drawing both its inner and outer radius: each radius is a perfectly reasonable number on its own, but the combination is only valid if the outer radius ends up bigger than the inner one. If it isn't, you don't want to keep half a bad draw around; you want to throw both numbers away and pick a fresh pair.

`StochasBase.transaction()` does exactly that: it re-runs a block of sampling code as many times as it takes for the whole block to succeed, and it does so safely, so a rejected attempt never leaves partial values behind in your model.

### How it works

1. Call `model.transaction(retry_on=..., max_retries=...)` and loop over it with `for attempt in ...:`. Each loop iteration is one attempt.
2. Wrap your sampling code in `with attempt:`.
3. Sample through `attempt.sample_dist(...)` (not `model.sample_dist(...)`) for any value that's part of the retry. This matters: `attempt.sample_dist` only resets a distribution's random-number stream the *first* time it's used in the transaction, so a retry actually draws a new value instead of quietly reproducing the same rejected one.
4. If the combination is invalid, `raise` one of the exception types you listed in `retry_on` (commonly `ValueError`, the same convention `pydantic` validators already use). The transaction catches it, discards anything registered during that attempt, and returns to the top of the for loop to try again.
5. If the combination is valid, just let the `with` block finish normally. The loop stops on its own. Everything sampled during that attempt is saved to `model.dists` and `model.named`, even though `attempt` itself is gone once the loop ends.
6. If no attempt succeeds within `max_retries`, the transaction gives up and raises a `RuntimeError` instead of silently returning a bad answer.

!!! tip

    Only errors listed in `retry_on` are treated as "bad draw, try again." Anything else, like a typo or a real bug, is not retried. It raises immediately, exactly as it would outside a transaction. The one difference is that `stochas` cleans up after that attempt first, so a half-finished attempt never leaves stray values sitting on the model.

### Example

```python
--8<-- "docs/user-guides/orchestration.py:transactions"
```

Running it prints something like:

```
Found a valid ring on attempt 1
inner=[0.85812869], outer=[1.14115587]
```

The attempt count depends on the random draws (it might take one or several). The second print reads the values from `model.named`, not the local variables, since that's where the finished, valid pair actually lives.

### Nesting Transactions

Not every check needs the same amount of redoing. A transaction can nest inside another one, splitting a validation into two steps: one that only depends on a single value and can retry on its own, and one that depends on several values together and needs to redraw all of them.

!!! example

    You draw a random paycheck and a random bill. The bill just has to be a plausible bill on its own, say between $50 and $400; that's its own step, and only the bill gets redrawn if it fails. But your leftover balance (paycheck minus bill) also has to stay above some minimum reserve, and that depends on both numbers together. If it fails, there's no way to tell whether the paycheck was too small or the bill too big, so both get redrawn.

```python
--8<-- "docs/user-guides/orchestration.py:nested_transactions"
```

Running it prints something like:

```
paycheck=[1004.25400539], bill=[190.16994879]
```

`bill_attempt`'s transaction handles the bill-only step. `paycheck_attempt`'s transaction handles the balance check: rejecting it redraws the paycheck and restarts the bill's transaction from scratch.

!!! warning

    Give every nesting level its own attempt variable name. Reusing `attempt` for both loops would make the inner one silently shadow the outer one, so any code after the inner loop would see the wrong attempt.
