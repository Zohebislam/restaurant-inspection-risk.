# Business Impact Simulation

- **Period:** Jan 2024 to Oct 2026 (34 months, 15,363 routine inspections, 3,565 failures)
- **Model:** `gradient_boosting.joblib` (trained only on inspections before 2024-01-01)
- **Setup:** each month keeps its real inspection dates; the model only changes which restaurant gets visited first.

## Headline results
- Failing restaurants would have been found **3.2 days sooner on average** (median 3 days).
- **57.9%** of failing restaurants would have been found sooner, 37.1% later.
- In total, that's **11,381 fewer days** that failing kitchens stayed open before an inspector arrived.
- By the middle of each month, inspectors would have found **65.5%** of that month's failures, compared to **50.1%** in the actual order.

## Same workload as the simple rule
Flagging the same 3,184 inspections as the baseline ("failed last routine inspection"), the model finds **1,155** failing restaurants vs. **934** for the baseline (**+221**).

## Chart
![simulation](figures/11_simulation.png)

## Caveats
- Assumes the same restaurants would have been inspected each month, and that a restaurant's result wouldn't change if it were visited a few days earlier or later.
- Only routine inspections are reordered; complaint inspections happen when complaints come in.
