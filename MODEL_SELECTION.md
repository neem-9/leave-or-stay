# Model selection

Fill this in after you have trained your candidates in MLflow. Short sentences are fine. Use numbers from your runs.

## The question
The model predicts whether a customer will leave the company,
if the model predicts that a customer will stay but the customer actually leaves, we lose them.
If the model predicts that a customer will leave but the customer actually stays, we waste a retention offer.

## Candidates
| Candidate | Why I tried it | AUC | F1 | F1 per slice (lowest) | p95 latency (ms) | Size (KB) | MLflow run ID |

| logreg | simple and fast baseline | 0.798 | 0.500 | 0.25 | 4.94 | 3.5 | c25e387f |
| random_forest | many trees vote together and can catch non-linear patterns | 0.791 | 0.444 | 0 | 12.95 | 478.1 | 513a6311 |
| gradient_boosting | trees are built one after another and each one fixes the previous errors | 0.759 | 0.396 | 0.25 | 5.17 | 205.7 | 855dea6a |

## The champion
I chose logreg.
It has the best F1 (0.500) and the best AUC (0.798).
It is also the fastest (4.94 ms) and the smallest (3.5 KB).
What I accept losing: its weakest contract group has F1 0.25, and gradient_boosting was better on two-year contracts (0.364). Also, F1 0.5 is not very high.
I did not choose random_forest because its F1 was 0 on one-year and two-year contracts, and it is the slowest and the biggest.

## My promotion rules
- Latency budget (p95): 20 ms, because our best model takes about 5 ms and the CI runner is slower than my laptop, so I leave a safety margin.
- Minimum gain to replace the champion (F1 points): 0.01, because my test data is small (about 60 customers who left), so a smaller difference can be just luck.

## What I would check next
The one-year and two-year groups are small. In my test data, only 7 customers left in each group, so their F1 numbers are not stable. I would test with more data before trusting them.
