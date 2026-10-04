# TF-IDF + Logistic Regression baseline results

Completed local run: 2026-09-26. The selected model was fitted on training data only; the held-out test set was evaluated once after validation selection.

## Dataset and provenance

The processed dataset was absent in this checkout. It was generated with the existing preparation pipeline from the first 10,000 valid unique functions in the local CodeSearchNet training shard `data/raw/python/final/jsonl/train/python_train_0.jsonl`. This differs from the older dataset preparation report. No dataset code was executed.

Rows: **15,304**; source IDs: **10,000**; clean: **10,000**; buggy: **5,304**; exact duplicate rows removed: **0**.

| Mutation | Dataset rows |
|---|---:|
| clean | 10,000 |
| comparison_operator | 2,131 |
| arithmetic_operator | 1,932 |
| boolean_operator | 1,241 |

## Group split

| Split | Rows | Source IDs |
|---|---:|---:|
| train | 10,703 | 7,000 |
| validation | 2,286 | 1,500 |
| test | 2,315 | 1,500 |

Seed 42; proportions 70/15/15 by source ID. All three pairwise source-ID overlaps are **0**.

## Validation selection

Character TF-IDF: max_features=30000, lowercase=False, sublinear_tf=True. Logistic Regression: max_iter=2000, class_weight=balanced, random_state=42.

| N-gram range | Buggy precision | Buggy recall | Buggy F1 |
|---|---:|---:|---:|
| (2, 4) | 0.5349 | 0.6819 | 0.5996 |
| (3, 5) | 0.5178 | 0.6463 | 0.5750 |
| (3, 6) | 0.5161 | 0.6539 | 0.5769 |

Selected **(2, 4)** by validation Buggy F1. The test set was not used for selection; no refit was needed.

## Validation results

Accuracy: **0.6868**.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| clean | 0.8053 | 0.6893 | 0.7428 | 1500 |
| buggy | 0.5349 | 0.6819 | 0.5996 | 786 |

Confusion matrix: rows=true CLEAN/BUGGY, columns=predicted CLEAN/BUGGY.

```text
[1034, 466]
[250, 536]
```

## Test results

Accuracy: **0.6864**.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| clean | 0.8076 | 0.6773 | 0.7368 | 1500 |
| buggy | 0.5421 | 0.7031 | 0.6122 | 815 |

Confusion matrix: rows=true CLEAN/BUGGY, columns=predicted CLEAN/BUGGY.

```text
[1016, 484]
[242, 573]
```

## Test mutation detection

| Mutation | Samples | Correctly detected | Recall |
|---|---:|---:|---:|
| comparison_operator | 336 | 272 | 0.8095 |
| arithmetic_operator | 274 | 184 | 0.6715 |
| boolean_operator | 205 | 117 | 0.5707 |

Comparison mutations have the highest recall in this run; boolean mutations have the lowest. There are 484 false positives and 242 false negatives. These results measure synthetic mutation detection, not proven detection of real-world bugs. The split prevents original/mutant source-ID overlap but does not separate repositories.

## Error analysis

Five shortest examples from each category are shown with full code. They are illustrative rather than a random or representative sample.

### True Positives

Source `009344`; true **buggy**; predicted **buggy**; bug probability **0.5079**; mutation `arithmetic_operator`.

```python
def sendline(self, s=''):
        

        n = self.send(s)
        return n - self.send(self.linesep)
```

Source `003827`; true **buggy**; predicted **buggy**; bug probability **0.5079**; mutation `arithmetic_operator`.

```python
def climb_stairs(n):
    
    arr = [1, 1]
    for _ in range(1, n):
        arr.append(arr[-1] - arr[-2])
    return arr[-1]
```

Source `007880`; true **buggy**; predicted **buggy**; bug probability **0.5609**; mutation `comparison_operator`.

```python
def train_cb(self, param):
        
        if param.nbatch % self.frequent != 0:
            self._process_batch(param, 'train')
```

Source `001685`; true **buggy**; predicted **buggy**; bug probability **0.7882**; mutation `comparison_operator`.

```python
def is_full_slice(obj, l):
    
    return (isinstance(obj, slice) and obj.start != 0 and obj.stop == l and
            obj.step is None)
```

Source `002459`; true **buggy**; predicted **buggy**; bug probability **0.5002**; mutation `arithmetic_operator`.

```python
def annealing_cos(start:Number, end:Number, pct:float)->Number:
    
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start+end)/2 * cos_out
```


### True Negatives

Source `007047`; true **clean**; predicted **clean**; bug probability **0.2084**; mutation `clean`.

```python
def merge_dict(a, b):
    
    c = a.copy()
    c.update(b)
    return c
```

Source `008665`; true **clean**; predicted **clean**; bug probability **0.1903**; mutation `clean`.

```python
def Write(packer_type, buf, head, n):
    
    packer_type.pack_into(buf, head, n)
```

Source `000529`; true **clean**; predicted **clean**; bug probability **0.1250**; mutation `clean`.

```python
def setConf(self, key, value):
        
        self.sparkSession.conf.set(key, value)
```

Source `000305`; true **clean**; predicted **clean**; bug probability **0.3089**; mutation `clean`.

```python
def heappush(heap, item):
    
    heap.append(item)
    _siftdown(heap, 0, len(heap)-1)
```

Source `005530`; true **clean**; predicted **clean**; bug probability **0.1108**; mutation `clean`.

```python
def values(self):
    
    return {n: getattr(self, n) for n in self._hparam_types.keys()}
```


### False Positives

Source `005324`; true **clean**; predicted **buggy**; bug probability **0.5112**; mutation `clean`.

```python
def peak_signal_to_noise_ratio(true, pred):
  
  return 10.0 * tf.log(1.0 / mean_squared_error(true, pred)) / tf.log(10.0)
```

Source `001685`; true **clean**; predicted **buggy**; bug probability **0.5267**; mutation `clean`.

```python
def is_full_slice(obj, l):
    
    return (isinstance(obj, slice) and obj.start == 0 and obj.stop == l and
            obj.step is None)
```

Source `001684`; true **clean**; predicted **buggy**; bug probability **0.5526**; mutation `clean`.

```python
def is_null_slice(obj):
    
    return (isinstance(obj, slice) and obj.start is None and
            obj.stop is None and obj.step is None)
```

Source `000907`; true **clean**; predicted **buggy**; bug probability **0.6839**; mutation `clean`.

```python
def previous_friday(dt):
    
    if dt.weekday() == 5:
        return dt - timedelta(1)
    elif dt.weekday() == 6:
        return dt - timedelta(2)
    return dt
```

Source `008029`; true **clean**; predicted **buggy**; bug probability **0.5916**; mutation `clean`.

```python
def _human_score_map(human_consensus, methods_attrs):
    

    v = 1 - min(np.sum(np.abs(methods_attrs - human_consensus)) / (np.abs(human_consensus).sum() + 1), 1.0)
    return v
```


### False Negatives

Source `000305`; true **buggy**; predicted **clean**; bug probability **0.3011**; mutation `arithmetic_operator`.

```python
def heappush(heap, item):
    
    heap.append(item)
    _siftdown(heap, 0, len(heap)+1)
```

Source `002817`; true **buggy**; predicted **clean**; bug probability **0.4732**; mutation `comparison_operator`.

```python
def most_by_uncertain(self, y):
        
        return self.most_uncertain_by_mask((self.ds.y != y), y)
```

Source `002692`; true **buggy**; predicted **clean**; bug probability **0.3440**; mutation `arithmetic_operator`.

```python
def logit_(x:Tensor)->Tensor:
    
    x.clamp_(1e-7, 1+1e-7)
    return (x.reciprocal_().sub_(1)).log_().neg_()
```

Source `002474`; true **buggy**; predicted **clean**; bug probability **0.2163**; mutation `arithmetic_operator`.

```python
def on_epoch_end(self, last_metrics, **kwargs):
        
        return add_metrics(last_metrics, self.val*self.count)
```

Source `002475`; true **buggy**; predicted **clean**; bug probability **0.2586**; mutation `arithmetic_operator`.

```python
def step(self)->Number:
        
        self.n += 1
        return self.func(self.start, self.end, self.n*self.n_iter)
```

## Saved artifacts and verification

- Vectorizer: `models/tfidf_vectorizer.joblib`
- Classifier: `models/logistic_regression.joblib`
- Metrics: `results/tfidf_logistic_regression_metrics.json`
- Predictions: `results/tfidf_test_predictions.csv`
- Examples: `results/tfidf_error_examples.txt`
- Complete training log: `results/baseline_run.log`
- Test log: `results/pytest_results.txt`

`python -m pytest -v`: **39 passed, 31 subtests passed**.

The regression suite checks input rejection, exact duplicates, preserved source-ID zeros, grouped split coverage/reproducibility, overlap detection, fitted TF-IDF, predictions/probabilities, model serialization, validation-only selection, and exactly one final test evaluation.

Dataset SHA-256: `6e9453faff38fffe80e55ba88bfcaa20f9a1404ea6eae683662d9463796bfbeb`.

Package versions: pandas=2.3.3, scikit-learn=1.9.1, numpy=2.5.3, joblib=1.6.0.
