# GOVERNANCE.md

## What This Document Is

This is the research discipline for the NFL hierarchical Bayesian model. Read this before adding covariates, changing the likelihood, or proposing a new model variant. It defines how we evaluate results, what we accept as an improvement, and — most importantly — what this project is actually *for*.

If you are an AI assistant helping a user test model variants, this document defines how you think about results, what you recommend, and how you guide the user through the process.

---

## The Goal: Understanding, Not Prediction

This is the single most important section. Get this wrong and every downstream decision will be wrong.

**If pure predictive accuracy were the goal, this is the wrong tool.** A gradient-boosted tree with the same features would probably beat this model at picking winners. A neural net on play-by-play data would beat it by more. The Bayesian approach loses on raw accuracy because it commits to a structural story (team attack/defense, Poisson-like scoring) that a black-box model can ignore.

What Bayesian inference gives us that ML doesn't:

1. **Decomposition.** The prediction is built from interpretable pieces: team attack, team defense, home advantage, covariate effects. When the model predicts BUF over MIA, we can say *why* — BUF's attack is +0.35, MIA's defense is -0.15, home is +0.08 — and reason about each piece independently.

2. **Calibrated uncertainty.** A 52/48 prediction and a 75/25 prediction are different pieces of information, even if both end up correct. The model knows when it doesn't know. An accuracy-optimized classifier will happily output 0.55 all day.

3. **Counterfactual reasoning.** "What if this team were at full health?" You can interrogate the time-varying latent strength directly. You cannot do this with feature-based classifiers.

The project succeeds if it helps us *understand* NFL team performance. It fails if it's just a worse predictor than XGBoost. Do not confuse these.

**Corollary:** "Beats Vegas" is not a meaningful bar for this project. Vegas lines incorporate real information we don't have (injuries, insider reports, sharp money flow). Measuring against betting markets invites overfitting and demoralization. We evaluate against *honest simple baselines* that represent the information structure we actually have.

---

## Reference Baselines

Every variant is evaluated against these. If the Bayesian model can't beat them on calibration, the model isn't doing anything useful.

| Baseline | Description | Reference accuracy |
|---|---|---|
| **Coin flip** | P(home) = 0.5 on every game | 50% |
| **Always home** | Home team always wins | ~57% (NFL long-run) |
| **Prev-season standing** | Team with better prior-season record wins | ~60% |
| **Elo** | Standard Elo with home-field bonus | ~62-65% (per FiveThirtyEight's historical NFL Elo) |

The Bayesian model needs to clear **Elo** to justify its complexity. Elo is simple, well-understood, and the honest competitor. The ~62-65% figure is from published work (FiveThirtyEight's NFL Elo model) — we accept it as a literature benchmark rather than reimplement Elo here.

Note that these are *winner-accuracy* reference numbers. Calibration is the ranking metric for this project (see next section), and we don't have equivalent literature benchmarks for Brier or ECE. The baselines exist primarily to sanity-check that our model isn't doing *worse* than a trivial alternative.

---

## The Headline Metrics

### Calibration is the metric

This is the Bayesian value-add, so this is where we rank variants. A model that claims 90% confidence should be right 90% of the time. A 90% credible interval on the score should contain the actual score 90% of the time.

The metrics we report:
- **Brier score** (lower is better; coin flip on 50/50 outcomes = 0.25)
- **Coverage** of 90% credible intervals (should be near 90%)
- **Expected Calibration Error (ECE)** across confidence bins
- **Reliability plot** — visual check that predicted probabilities match observed frequencies

### What counts as a meaningful improvement

The bar is "beat the current base model by more than the noise floor, across multiple seasons." We do not assert absolute "good/bad" thresholds without grounding them in measurement.

The noise floor is set by sample size:
- Brier SE at n ≈ 150 games is ~0.015. **Brier improvements < 0.01 are not distinguishable from noise.**
- Accuracy SE at n ≈ 150 is ~3.5 percentage points. **Accuracy improvements < 2pp are not distinguishable from noise.**

These are statistical thresholds, not opinions. They do not change based on what the base model achieves — they're the minimum detectable difference at our sample size.

A variant meaningfully beats the base model if:
- Brier improves by > 0.01 **and** the improvement holds across at least 2 seasons, OR
- Coverage moves from outside [85%, 95%] into that range, OR
- ECE improves by > 0.02

**AND** no other metric regresses by more than its own noise floor. The "OR" above gives multiple ways to demonstrate improvement, but improvement on one metric does *not* license regression on another. Specifically: a variant whose ECE improves while Brier worsens by > 0.01 is rejected, because Brier is a proper scoring rule that captures both informativeness and calibration. ECE alone can be gamed by a model that hedges to 0.5 (perfect calibration, zero information). Brier penalizes that correctly.

Raw accuracy is reported but does not rank variants. A variant that improves accuracy by 2 points while worsening calibration is *worse*, not better — it's become more overconfident.

### Decomposition quality is the other metric

This is less quantifiable but equally important. After fitting, check:

- **Do team strength rankings match expert consensus?** The top-5 attack teams should overlap substantially with ESPN/PFF top-5 offenses. If they don't, the model has learned something strange.
- **Do covariate coefficients have the expected sign and magnitude?**
  - `rest_advantage` should be positive (more rest → better)
  - `home_short_week` should be negative
  - `temp_std`, `wind_std` should be negative (bad weather → lower scoring)
  - `div_game` sign is unclear a priori — if it's large (|β| > 0.1), investigate why
- **Does the time-varying model capture known events?** If a team's attack parameter drops 0.3 in the week after their QB broke his leg, the model is working. If it doesn't move, the innovation scale is too tight.

If a variant improves calibration but produces nonsense team rankings or wrong-signed covariates, it's overfit in a way that matters for our goal.

---

## The Pre-Registration Workflow

Follow this process for every new variant. AI assistants: walk the user through each step; do not skip.

### Step 1: State the Hypothesis

Before touching code, write down:
- What are you changing? (likelihood, parameterization, new covariate, time dynamics)
- Why do you think it will help? (causal/statistical reasoning, not "might improve accuracy")
- What do you expect to see? (better calibration in late-season games? smaller covariate coefficient for indoor games?)

**AI guidance:** Push back if the reasoning is "I want to try it." Trying things is fine, but note the prior on improvement is low. Most sensible variants didn't beat the base model in prior testing.

### Step 2: Fix Parameters Before Running

Every parameter fixed before seeing any result:
- Training window size (e.g., 8 weeks)
- Number of MCMC samples (e.g., 500)
- Simulations per prediction (e.g., 500)
- Covariate list (exactly which columns)
- Test seasons and weeks

**AI guidance:** Suggest defaults that match prior runs. If the user changes a non-focal parameter ("let me just increase samples to 2000 for this one"), flag it — that changes the comparison.

### Step 3: Run Once, Across Multiple Seasons

Minimum: 2 full seasons of backtests. Ideally 3+. Rolling train window, predict the next week, measure calibration and accuracy.

Do not run, look at 2024 results, adjust the model, and rerun. That's curve-fitting to 2024.

If the user wants to test two variants (e.g., with and without a covariate), both get pre-registered and both run once. The comparison is honest only if neither was tuned based on the other's results.

### Step 4: Evaluate Honestly

Report, in this order:

1. **Calibration** across all seasons (Brier, coverage, ECE) — the ranking metric
2. **Decomposition sanity** — do team rankings and covariate signs look right?
3. **Accuracy** — reported, but not the ranking metric
4. **Per-season breakdown** — does the variant win everywhere or only in one season?
5. **Where it's worse** — every variant has a weakness. Find it.

**AI guidance:** Do not lead with the best number. Lead with the full picture. If the variant improves Brier by 0.015 in 2024 but is flat in 2023, that's a 2024-specific effect worth naming.

### Step 5: Make a Recommendation

One of:

- **Adopt**: meaningfully better calibration, decomposition still sensible, robust across seasons, complexity justified
- **Interesting but not worth it**: improvement within noise, or complexity cost too high
- **Reject**: worse calibration, or nonsense decomposition, or fragile across seasons
- **Investigate further**: promising signal but needs diagnosis (specify what)

Most variants will be "interesting but not worth it" or "reject." That's normal. The base model is surprisingly hard to beat without overfitting.

---

## Lessons from Prior Work

These findings are established. A new variant should not repeat these mistakes.

### Likelihood Choice Matters More Than Covariates

Switching from Poisson to Negative Binomial was a bigger improvement than any covariate we could add. NFL scores are overdispersed (scoring in 3s and 7s) and Poisson's equal-mean-variance assumption is wrong by roughly 30-40%. The NB's dispersion parameter α is learned from data and captures this properly.

The three-way likelihood experiment (Poisson vs NB-weak vs NB-tight) confirmed this empirically:

| Variant | 4-yr Brier | ECE | Sim std / obs std |
|---------|-----------|------|-------------------|
| poisson | 0.233 | 0.122 | 0.72 (28% too narrow) |
| nb_weak | 0.224 | 0.069 | 1.09 (9% too wide)    |
| nb_tight | 0.221 | 0.056 | 1.07 (7% too wide)   |

Poisson is dramatically over-confident; its 95% credible interval contains only 78% of observed spreads. The Baio/Blangiardo soccer choice does not transfer to NFL scoring.

**Takeaway:** If something is structurally wrong about the generative story, no amount of covariate engineering will fix it. Fix the likelihood first. And — separately — don't assume choices that worked in one sport (soccer goals) transfer to another (NFL scores) without checking the variance/mean ratio first.

### Inheriting Priors from a Different Domain Has Hidden Costs

When the project was migrated from PyMC3 → PyMC5, the likelihood was changed from Poisson to Negative Binomial to handle NFL overdispersion. The α prior was set to `Exponential(1)` as a generic weakly-informative default, with no analysis of plausible α values for NFL scoring.

The calibration inspection later showed this caused systematic under-confidence (predictive distribution ~10% too wide). The follow-up experiment tested a more informed prior (`LogNormal(2.7, 0.4)`, median ≈ 15), which produced consistent improvements across all 4 seasons but didn't clear the noise floor. Per the governance criteria, the default stayed at the original Exp(1).

The tighter prior would have been a better starting point — but the lesson isn't "we picked the wrong prior." It's that **the time to think about priors is before fitting, not after**. The tighter LogNormal version would have been "interesting but not worth it" if we'd run it as an A/B against `Exp(1)` chosen with the same care; the only reason it looked tempting in inspection was that we found ourselves in a bad place we didn't intend to be in.

**Takeaway:** When you change a likelihood (Poisson → NB), the new dispersion parameter is a real modeling choice and deserves the same care as the rest of the model. A "weakly-informative default" is not a free pass — it carries assumptions about plausible parameter ranges that may or may not match your domain.

### Parameterization Affects Sampling, Not Just Aesthetics

Non-centered parameterization (`atts = sd_att * atts_raw` where `atts_raw ~ N(0,1)`) was not a cosmetic change. With 32 teams and ~17 games each, the centered form produces funnel geometries that NUTS explores inefficiently, yielding divergences and poor ESS. The non-centered form removes this.

**Takeaway:** When group-level variance is small relative to data, non-centered is almost always better. This is a well-documented pattern — it wasn't ours to discover.

### Time-Varying Team Strengths Collapsed to Identifiability Failure

A prior branch (`rolling-regression`) had explored time-varying team strengths via a custom AR(1) GaussianRandomWalk but never ran the comparison against a static baseline. The current project re-implemented it cleanly (`time_varying=True`) and ran the comparison. The variant failed:

- 3-year mean accuracy: **57.7%** (vs 66.2% for base)
- 3-year mean Brier: **0.247** (vs 0.223 for base)
- ~100× slower than static (40-60 min per season vs 30-40 seconds)

Diagnostics from the saved predictions showed *why* it failed:

- 95th percentile of home_win_prob was only 0.592 (vs 0.721 for base) — the model never commits to a prediction
- Mean |prob − 0.5| was 0.046 (half of base's 0.085)
- Predicted-spread std was 1.09 (vs 3.49 for base) — under-committing by 3×
- Picks home team 93.5% of games → essentially the "always home" baseline (~56.6% accuracy)

This is an **identifiability collapse**, not a sampler problem. The model has 8 weeks × 32 teams × 2 (attack + defense) = 512 team-week parameters, estimated from ~120 games per training window (~4 observations per parameter). The posterior over each `atts[week, team]` is near the prior, so team contributions cancel in the log-linear combination and only the home-field advantage survives.

Faster samplers (nutpie, NumPyro) would not fix this. Adding more weeks wouldn't help much either — 16 weeks × 32 = 1024 params vs ~240 games is still catastrophically under-identified.

**Takeaway:** Before adding parameters, count them against your data. For hierarchical time models, borrow strength from a static team intercept and let the GRW capture *deviations only* (`atts[t, team] ~ N(atts_static[team], sigma_t)`) — do not give each team-week an independent parameter.

Potential fixes to revisit if we ever want time-varying team strengths:
- Hierarchical time prior (static intercept + deviations, as above)
- Much tighter innovation scale (HalfNormal(0, 0.01) not 0.1)
- Multi-week smoothing in prediction (weighted average of recent weeks, not just last)
- Non-centered GRW parameterization

### The Hierarchical-Anchor AR(1) Fix Also Failed

The "potential fixes" listed above came directly from the lit review on Glickman & Stern (1998), the canonical NFL Bayesian framework. We implemented them: a static team-strength anchor + a `pm.GaussianRandomWalk` deviation around it, with a tight innovation scale (`HalfNormal(0.05)`) and shared variance across teams. The architecture compiles, samples, and avoids the identifiability collapse of the previous version (the static-anchor-dominates-deviation test in the test suite confirms this). Tests pass.

It still made predictions worse on 4 seasons of data:

| Season | base (Brier) | ar1 (Brier) | Δ      |
|--------|-------------|-------------|--------|
| 2022   | 0.229       | 0.243       | +0.014 |
| 2023   | 0.235       | 0.245       | +0.010 |
| 2024   | 0.208       | 0.252       | +0.044 |
| 2025   | 0.234       | 0.252       | +0.018 |
| Mean   | 0.227       | 0.248       | **+0.021** |

Mean accuracy: 61.9% → 54.7% (−7.2pp). Runtime exploded: the 4-season AR(1) backtest took 391 minutes (one season took 3.7 hours alone). All four seasons regress past the noise floor in the same direction.

Counterintuitively, the AR(1) model has **better aggregate ECE** (0.015 vs 0.085 for base). But this improvement comes from the model hedging to 0.5 — the 95th percentile of `home_win_prob` is much lower than base, so individual predictions are less informative. Brier (a proper scoring rule) catches this and shows the model is genuinely worse. ECE alone is gamed by uninformative hedging.

**Takeaway 1:** "Just add a hierarchical anchor" is not sufficient. The static anchor + RW interaction creates a different problem than independent RWs but doesn't solve the underlying issue: at 8-week training windows with ~5 games per team, there isn't enough information to identify both season-long ability AND week-to-week change, and the model's predictions degrade.

**Takeaway 2:** ECE is a necessary but not sufficient calibration metric. A model that hedges to 0.5 will achieve perfect ECE while making no useful predictions. Use Brier (proper scoring rule) as the primary metric and ECE as a diagnostic. The governance criterion was updated to require: "improvement on one metric does not license regression on another past the noise floor."

**Takeaway 3:** Lit-review-derived fixes are not free. The lit review pointed at hierarchical time priors as the canonical solution. We implemented it correctly per the published recipe (Glickman-Stern 1998 + the Pitt-Walker hierarchical anchor pattern). It still didn't work on our data scale. Either NFL within-season time variation is not the dominant source of predictability (and the static model is already capturing what's there), or our 8-week training window is too small for the structure to identify, or both. Glickman-Stern's reported gains may have come from full-season fits (not 8-week rolling windows) — a context our backtest cannot match.

The `time_varying=True` flag now points at the hierarchical-anchor implementation but is still **rejected as a default**. Code remains in the tree (tests pass, reproducible), documented as a rejected variant.

### Multi-Season Carryover Did Not Help

GS-style improvement #2: pass prior-year posterior team-strength means as informed priors for the current year (`team_priors` kwarg, `carryover_sd = 0.1` log-rate units). Implemented, tested (4 unit tests), backtested across the chain 2022 → 2023 → 2024 → 2025 (the `2022` season is identical for both arms since it's the chain origin and has no prior to inform it):

| Season | base (Brier) | carryover (Brier) | Δ      |
|--------|-------------|-------------------|--------|
| 2023   | 0.235       | 0.237             | +0.002 |
| 2024   | 0.205       | 0.212             | +0.007 |
| 2025   | 0.234       | 0.235             | +0.001 |
| Mean   | 0.225       | 0.228             | **+0.003** |

Pooled metrics (2023-2025): accuracy 64.0% → 61.8% (−2.2pp); Brier 0.225 → 0.228 (+0.003); ECE 0.054 → 0.072 (+0.018). All three regress in the same direction across 3 seasons. None individually clears the noise floor, but the consistent direction is informative.

The 2024 result is the most striking: 71.1% → 69.8% accuracy, Brier worse by 0.007. 2024 had several teams change substantively from 2023 (WAS with Jayden Daniels, multiple coaching shakeups). A tight carryover prior pulled estimates back toward stale 2023 posteriors and over-shrank the actual changes.

In the high-confidence bins, carryover commits to *more* predictions but is *less* accurate when it does:

| 2024 confidence tier | base (acc, n)     | carryover (acc, n) |
|---------------------|-------------------|--------------------|
| ≥70%                | 84.2% (n=19)      | 71.4% (n=28)       |

Same operational pattern as the AR(1) failure: prior-informed approaches commit more aggressively but with worse calibration in the bins where they commit. **Decision: REJECTED.** The `team_priors` kwarg stays in the API as a documented option but is not the default.

**Takeaway 1:** Carryover's intuitive value (don't throw away prior-year info) competes against its real cost (the prior is wrong about teams that genuinely changed). At an 8-week training window the in-season data already overwhelms a tight prior most weeks; the prior only matters in early-season weeks, which are inherently the hardest. The cost shows up most when teams change a lot.

**Takeaway 2:** A pre-registered tight `carryover_sd = 0.1` is what was tested. Looser values (e.g., 0.3) might fare better, but tuning the hyperparameter to chase a positive result would be the kind of post-hoc optimization the discipline doc warns against. The right next experiment, if revisited, would pre-register a small range (e.g., 0.05, 0.15, 0.30) and report all results — not pick the best post hoc.

### Per-Team Home-Field Advantage Didn't Help

GS-style improvement #3: replace the single shared `home` scalar with a hierarchical per-team home-field advantage (`mu_home + sigma_home * raw[t]`, non-centered). Tested across 2022-2025:

| Season | base (Brier) | per_team_home (Brier) | Δ      |
|--------|-------------|-----------------------|--------|
| 2022   | 0.223       | 0.225                 | +0.002 |
| 2023   | 0.236       | 0.234                 | −0.002 |
| 2024   | 0.205       | 0.205                 |  0.000 |
| 2025   | 0.231       | 0.233                 | +0.002 |
| Mean   | 0.224       | 0.224                 | **~0.000** |

Pooled metrics: accuracy 64.4% → 62.5% (−1.9pp); Brier identical (0.224 → 0.224); ECE 0.060 → 0.086 (+0.026, in the wrong direction but within noise floor of 0.02 give or take). All within noise. The cleanest "no effect" of the four GS experiments.

The model successfully estimates a per-team HFA structure on a full-season fit: global `mu_home = +0.076` (≈ 7.9% scoring uplift), `sigma_home = 0.054` (cross-team variation in HFA). The per-team estimates range from +0.04 to +0.10. Each team's individual SD is ~0.07 — comparable to the cross-team spread, meaning the hierarchical model can't resolve which teams have above-average home advantage with confidence.

This is a small-N identifiability problem: each team plays ~17 home games per season, and the hierarchical pooling between the `mu_home`/`sigma_home` hyperparameter and the team-level `home_raw[t]` works as expected — it just can't extract more signal than the data contains.

**Takeaway:** Per-team HFA is a real phenomenon (sigma_home > 0 in the posterior), but identification at 8-week training windows is poor, and the small per-team improvements wash out against the increased predictive variance. This isn't a "the structure was wrong" failure — it's a "the data isn't dense enough" failure. The kind of variant that would benefit most from more seasons of training data; would be worth re-testing if the project ever pools across multiple training years.

**Decision: REJECTED.** `per_team_home=True` stays in the API as a documented option. Default remains the single shared scalar.

### Covariates Didn't Help on 4 Seasons

The +covariates variant (`rest_advantage`, `temp_std`, `wind_std`) was run on 2022-2025. Brier differences vs base:

- 2022: −0.004, 2023: 0.000, 2024: +0.002, 2025: +0.002 → mean −0.000

All within the 0.01 noise floor. Accuracy differences were similarly small and non-directional (some seasons up, some down). Fit time increased ~30% per run.

**Takeaway:** Simple covariates from game metadata don't help the model. The Poisson/NB scoring-rate parameterization already captures most of what teams do; adding an `exp(β × rest_days)` multiplier on top doesn't pick up signal that team-attack/defense haven't already absorbed. If future covariates are considered, they should reflect something the team parameters cannot — e.g., starting-QB identity, injury to a specific player, weather that's extreme enough to change the sport (wind > 20mph, temp < 20°F).

**Also:** The original "covariates are not free" logic still holds — a covariate with a near-zero, noisy coefficient is probably making out-of-sample predictions slightly worse, not better.

### Accuracy Peaks Early in the Season Are Suspect

When the model is trained on weeks 1-4 and predicts week 5, it has access to very little data and the posterior is nearly the prior. Any accuracy result here is noisy — a few lucky upsets can swing it wildly. Reliable evaluation starts around week 9-10 when each team has enough games to inform its parameters.

**Takeaway:** Don't include weeks 5-8 in primary evaluations. They're too noisy and will mislead comparisons between variants.

---

## What to Be Skeptical About

When evaluating any new result, watch for:

**Single-season results.** A variant that improves Brier by 0.02 in 2024 and is flat in 2023 is probably fitting to 2024. NFL seasons vary; the 2022 season had dramatically more upsets than 2019. A real improvement shows up in both good and chaotic seasons.

**Accuracy-without-calibration.** If a variant improves accuracy by 3 points but Brier gets worse, it has become more overconfident — predicting 80% when it should predict 70%. On a week where it's wrong, it'll be badly wrong. This is the worst possible outcome for a model meant to *quantify uncertainty*.

**Covariate coefficients that don't match priors.** If `rest_advantage` has a negative coefficient (more rest → worse performance), something is wrong. Either the data is wrong, the feature engineering is wrong, or there's a confounder we're not capturing. Don't shrug and ship it.

**MCMC divergences.** A run with 50 divergences and `max_treedepth` hits is not a valid fit, no matter what the accuracy looks like. Divergences mean the sampler couldn't explore the posterior geometry, and the posterior samples may not represent the true distribution. Fix the geometry (non-centered reparam, tighter priors) before interpreting results.

**Small-sample confidence claims.** Brier improvement of 0.008 on 150 games is not statistically distinguishable from zero. The rough SE on Brier at n=150 is ~0.015. Don't celebrate improvements smaller than that.

**Recency bias in team rankings.** A team that won its last 3 games will show up high in a short-training-window model. That doesn't mean the model has identified a "hot team" signal — it means it's near-overfitting recent noise. Check that ranking changes correspond to real events (roster changes, coaching changes) and not just recent W/L.

---

## Encouraging Exploration

Nothing in this document should prevent testing a wild idea. Bivariate Poisson for correlated scores? Try it. QB-level parameters instead of team-level? Try it. Change the likelihood to Skellam for score differentials? Go for it.

The discipline is in the *evaluation*, not the ideation. Test anything. But test it honestly:
- Pre-register parameters
- Run across multiple seasons
- Rank by calibration, not accuracy
- Check decomposition for sense
- Report the full picture
- Be willing to conclude "the base model is still better"

The most valuable outcome from this project is not the winning variant — it's the growing understanding of *which modeling choices actually matter for NFL games and which are noise*. Every honest experiment that concludes "the simple version is still better" is information. That's what the Bayesian framing is for.
