# Responsible AI

All numbers below are measured on the held-out test set (1,065 wines: 272 red, 793 white) and are reproduced
by the pipeline: fairness metrics are logged on every MLflow run, the explainability report is stored as a
run artifact under `explainability/`.

## 1. Fairness

### What "group" means here

The dataset contains no personal data, so there is no protected attribute such as gender or age. The group
that matters is the **wine color**. The two colors come from different producers and product lines; a model
that is systematically worse on one color treats those producers unfairly and misleads the people relying on
it. Wine color is also not an input of the model, so any difference comes from the chemistry alone.

Code: `src/fairness.py`. For each group it computes accuracy, selection rate (share predicted "good"), true
positive rate and false positive rate, then the largest gap between groups.

### Bias detected in the first model

Version 1 is trained on red wine only and then meets white wine in production.

| Version 1 (red only) | n | Accuracy | Predicted good | Actually good | TPR | FPR |
|---|---|---|---|---|---|---|
| Red | 272 | 0.776 | 0.511 | 0.529 | 0.771 | 0.219 |
| White | 793 | 0.683 | 0.562 | 0.660 | 0.686 | 0.322 |

On white wine the model is barely better than always answering "good" (0.660). It misses 31% of the good
white wines and wrongly approves 32% of the others. This is representation bias: one group was absent from
the training data.

### Mitigation and its effect

The mitigation is the retraining loop itself: Evidently detects that production inputs no longer look like
the training data and the DAG retrains on both colors.

| Version 2 (red + white) | n | Accuracy | Predicted good | Actually good | TPR | FPR |
|---|---|---|---|---|---|---|
| Red | 272 | 0.783 | 0.504 | 0.529 | 0.771 | 0.203 |
| White | 793 | 0.753 | 0.687 | 0.660 | 0.834 | 0.404 |

| Gap between colors | Version 1 | Version 2 |
|---|---|---|
| Accuracy | 0.092 | 0.030 |
| Equal opportunity (TPR) | 0.084 | 0.063 |
| Demographic parity (selection rate) | 0.051 | 0.184 |
| False positive rate | 0.103 | 0.201 |

Retraining closed most of the accuracy gap and did not hurt red wine. It did not make the model fair in
every sense:

- **The false positive rate gap doubled.** 40% of the not-good white wines are now predicted good, against
  20% for red. For a pre-screening tool this is the costly error: a weak batch passes.
- **The demographic parity gap grew**, mostly because the colors differ in reality: 66% of white wines are
  good against 53% of red, and the model now reflects that. Equal selection rates would be the wrong goal here.

### Further mitigations (not implemented)

| Option | Effect | Cost |
|---|---|---|
| Separate decision threshold per color | Can equalise the false positive rates directly | Needs the color at prediction time |
| Add wine color as a feature | Lets the model learn color-specific rules | Same, and changes the API contract |
| Class or sample weights | Penalise false positives on white wine | Lowers recall on good white wines |
| Promotion rule that also checks the gaps | Blocks a model that improves accuracy by sacrificing one group | More retraining runs rejected |

The last one is the most relevant weakness of the current system: promotion looks only at overall accuracy,
so a future model could be promoted while getting worse on the smaller group (red, 26% of the test set).
The gaps are logged for every run precisely so that this is visible in MLflow.

## 2. Explainability

Code: `src/explainability.py`. Three methods are used because each answers a different question.

| Method | Question it answers | Scope |
|---|---|---|
| Impurity-based importance | Which features did the forest split on most? | Global, computed from the training process |
| Permutation importance | How much accuracy is lost on unseen data when a feature is shuffled? | Global, model-agnostic |
| SHAP (TreeExplainer) | How much does each feature push an individual prediction up or down? | Local, summarised in a beeswarm plot |

Results for version 2:

| Feature | Permutation importance (accuracy drop) | Impurity importance |
|---|---|---|
| alcohol | 0.081 | 0.207 |
| volatile_acidity | 0.053 | 0.114 |
| free_sulfur_dioxide | 0.020 | 0.082 |
| sulphates | 0.015 | 0.076 |
| total_sulfur_dioxide | 0.014 | 0.076 |
| chlorides | 0.011 | 0.080 |
| density | 0.003 | 0.107 |
| pH | -0.004 | 0.061 |

- **Alcohol and volatile acidity dominate** under both methods. This matches wine knowledge: volatile acidity
  is the vinegar taint, and higher alcohol goes with riper grapes.
- **The two methods disagree on density.** The forest splits on it often (0.107) but shuffling it costs almost
  no accuracy (0.003), because density is nearly determined by alcohol and sugar. Impurity importance
  overstates correlated features; this is why it is not used alone.
- Shuffling pH or fixed acidity does not reduce accuracy at all: the model does not depend on them.

The SHAP summary plot (`explainability/shap_summary.png` in each MLflow run) additionally shows the direction
of each effect for individual wines.

## 3. Data privacy

- **No personal data.** The dataset holds chemical measurements of wines. It was published without brand,
  producer or price, which also protects the commercial interests of the producers.
- **What the system stores.** Each inference (11 measurements, prediction, model version, timestamp) is kept
  in memory by the Evidently service, limited to the last 10,000 rows and lost on restart. Nothing links an
  inference to a person or a client; `prediction_id` is a random UUID.
- **If the system were used with real customers**, the measurements of an unreleased batch are commercially
  sensitive. The API has no authentication and the services use default passwords, which is acceptable only
  for a local demonstration. A real deployment needs authentication, TLS and a retention policy for
  captured inferences.

## 4. Ethical implications

| Risk | Why it matters | Mitigation in place or proposed |
|---|---|---|
| The model replaces the tasting panel | "Good" is a human sensory judgement; 24% of predictions are wrong | The system is positioned as pre-screening. The panel decides; the API returns a class, not a verdict on the batch. |
| Automation bias | People tend to trust a number over their own judgement | Explainability report and per-color error rates are published so users know where the model is weak. |
| Unequal errors between producers | 40% false positives on white against 20% on red | Measured and logged on every run; mitigations listed above. |
| Subjective, culturally narrow labels | Grades come from a few tasters of one Portuguese wine region | The model must not be presented as a measure of quality in general, nor reused for other regions without new data. |
| Silent degradation | A drifting model keeps answering with full confidence | Drift detection, alerting and automatic retraining are the core of this project. |
| Automatic retraining goes wrong | A bad model could be deployed without anyone looking | A candidate is deployed only if it is not worse than Production on the same test set; old versions stay archived in the registry for rollback. |
| Effect on jobs | Pre-screening reduces the work of tasters | Intended use is to prioritise the panel's limited time, not to remove the role. |

### Limits of this analysis

- One train/test split; the differences between colors have not been tested for statistical significance.
- Fairness is analysed for the only grouping available. Quality levels within a color (for example the rare
  very good or very bad wines) were not analysed.
- The business setting is an assumption of the project; the ethical discussion would need the real users.
