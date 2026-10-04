# TF-IDF + Logistic Regression baseline results

Run: 2026-09-28. Completed the character-level baseline only.

## Dataset and leakage checks

14,743 samples from 10,000 source functions: 10,000 CLEAN and 4,743 BUGGY.
Buggy mutations: 1,956 comparison, 1,838 arithmetic, and 949 boolean.

| Split | Source IDs | Samples | CLEAN | BUGGY |
|---|---:|---:|---:|---:|
| train | 7,000 | 10,315 | 7,000 | 3,315 |
| validation | 1,500 | 2,206 | 1,500 | 706 |
| test | 1,500 | 2,222 | 1,500 | 722 |

Unique source IDs are split 70/15/15 with seed 42. All three pairwise
source-ID intersections are **zero**. Saved CSVs were checked against the
original dataset: exact row coverage, valid binary labels, and preserved leading zeros.

## Validation-only configuration selection

| Character n-grams | Buggy precision | Buggy recall | Buggy F1 |
|---|---:|---:|---:|
| (2, 4) | 0.554147 | 0.681303 | 0.611182 |
| (3, 5) | 0.541806 | 0.688385 | 0.606363 |
| (3, 6) | 0.529736 | 0.681303 | 0.596035 |

**Selected `(2, 4)`**, by highest validation Buggy F1.
All configurations used `max_features=30000`, `lowercase=False`,
`sublinear_tf=True`, and `analyzer="char"`. Logistic Regression used
`max_iter=2000`, `class_weight="balanced"`, and `random_state=42`.
TF-IDF and classifiers were fitted only on training data. The selected
trained model was retained without refitting. Test was evaluated once.

## Validation results

Accuracy: **0.722575**.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| CLEAN | 0.831839 | 0.742000 | 0.784355 | 1500 |
| BUGGY | 0.554147 | 0.681303 | 0.611182 | 706 |

Confusion matrix: rows = true CLEAN/BUGGY, columns = predicted CLEAN/BUGGY.

```text
[[1113,  387],
 [ 225,  481]]
```

## Test results

Accuracy: **0.714221**.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| CLEAN | 0.837627 | 0.715333 | 0.771665 | 1500 |
| BUGGY | 0.546227 | 0.711911 | 0.618160 | 722 |

Confusion matrix: rows = true CLEAN/BUGGY, columns = predicted CLEAN/BUGGY.

```text
[[1073,  427],
 [ 208,  514]]
```

## Test recall by mutation type

| Mutation type | Samples | Detected correctly | Recall |
|---|---:|---:|---:|
| arithmetic_operator | 269 | 188 | 0.698885 |
| boolean_operator | 142 | 77 | 0.542254 |
| comparison_operator | 311 | 249 | 0.800643 |

Comparison mutations have the highest observed detection recall (80.06%);
boolean mutations have the lowest (54.23%). Overall, 514/722 buggy samples
were detected, 208 were missed, and 427/1,500 clean samples were flagged.
Test accuracy is 71.42%, compared with 67.51% for always predicting CLEAN.
The model detects some synthetic mutation patterns, but false positives remain frequent.
These results do not establish semantic correctness or real-world bug detection quality.

## Error analysis

The saved examples use the first five test rows in each outcome category,
without confidence-based cherry-picking. All **20 full examples** include
`source_id`, true label, predicted label, bug probability, mutation type, and code:
[complete examples](../results/tfidf_error_examples.txt).

| Outcome | Source ID | True → predicted | Bug probability | Mutation type |
|---|---|---|---:|---|
| false positives | `000011` | CLEAN → BUGGY | 0.514055 | clean |
| false positives | `000018` | CLEAN → BUGGY | 0.504103 | clean |
| false positives | `000024` | CLEAN → BUGGY | 0.778942 | clean |
| false positives | `000046` | CLEAN → BUGGY | 0.562580 | clean |
| false positives | `000047` | CLEAN → BUGGY | 0.589000 | clean |
| false negatives | `000032` | BUGGY → CLEAN | 0.494893 | arithmetic_operator |
| false negatives | `000040` | BUGGY → CLEAN | 0.435340 | comparison_operator |
| false negatives | `000070` | BUGGY → CLEAN | 0.334525 | boolean_operator |
| false negatives | `000080` | BUGGY → CLEAN | 0.424319 | arithmetic_operator |
| false negatives | `000109` | BUGGY → CLEAN | 0.486559 | boolean_operator |

### False Positives: `000024`

CLEAN → BUGGY; bug probability 0.778942; mutation type `clean`.

```python
def contrib_email(contrib_tag):
    
    email = []
    for email_tag in extract_nodes(contrib_tag, "email"):
        if email_tag.parent.name != "aff":
            email.append(email_tag.text)
    return email if len(email) > 0 else None
```

### False Negatives: `000070`

BUGGY → CLEAN; bug probability 0.334525; mutation type `boolean_operator`.

```python
def create_role(self, name=None, permissions=""):
        
        name = name and "autocreated-role"
        from qubell.api.private.role import Role
        return Role.new(self._router, organization=self, name=name, permissions=permissions)
```

### False Negatives: `000080`

BUGGY → CLEAN; bug probability 0.424319; mutation type `arithmetic_operator`.

```python
def HGPA(hdf5_file_name, cluster_runs, verbose = False, N_clusters_max = None):
    
    
    print('\n*****')
    print("INFO: Cluster_Ensembles: HGPA: consensus clustering using HGPA.")

    if N_clusters_max == None:
        N_clusters_max = int(np.nanmax(cluster_runs)) - 1

    return hmetis(hdf5_file_name, N_clusters_max)
```

## Artifacts and verification

- [Vectorizer](../models/tfidf_vectorizer.joblib): 30,000 character features.
- [Logistic Regression](../models/logistic_regression.joblib): matching 30,000 input features; classes `[0, 1]`.
- [Metrics JSON](../results/tfidf_logistic_regression_metrics.json): all experiment/evaluation values, examples, versions, and dataset hash.
- [Training log](../results/tfidf_training.log): full printed output.
- [Pytest log](../results/pytest.log): **45 passed, 31 subtests passed**.
- `pip check`: no broken requirements found.
- Saved-model inference on `def calculate_total(price, tax): return price - tax` returned
  `{"label": "buggy", "bug_probability": 0.514437584491533}`.

Models and CSVs remain local ignored artifacts, following the existing `.gitignore`.

Versions: python 3.12.3, pandas 2.3.3, numpy 2.5.3, scikit_learn 1.9.1, joblib 1.6.0.

Dataset SHA-256: `0734fa6e3a65830d2981878afec8fbbcfcc8cddec552dab633727efdf7925120`.
