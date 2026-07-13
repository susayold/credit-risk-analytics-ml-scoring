# Credit Risk Portfolio Case Study

## Executive Summary

This project builds an evidence-based credit review workflow from 307,511 labeled Home Credit applications. The analytical grain is one row per customer. Application data and historical credit behavior are consolidated into a 271-feature master table, then translated into Power BI monitoring, diagnostic analysis, and an ML review-prioritization model.

The portfolio default baseline is 8.07%. Because the positive class is uncommon, the project evaluates ranking and concentration using ROC-AUC, PR-AUC, KS, Lift, and default capture rather than relying on accuracy.

## Portfolio Conclusions

### Payment behavior is the most actionable risk family

Credit card utilization above 100% produced a 25.50% default rate, 3.16 times the portfolio baseline. Late-payment rates above 30% and underpayment rates above 30% produced default rates of 13.35% and 12.34%. These measures reflect observed repayment stress and therefore support early-warning monitoring and manual review.

### Historical credit reveals exposure missing from the current application

Applicants with at least two overdue bureau loans reached 36.80% default. A previous-application refusal rate above 50% reached 15.93%. Current income and requested credit should therefore not be evaluated without external obligations and historical outcomes.

### Affordability is contextual and nonlinear

Credit-to-income and annuity-to-income ratios summarize repayment burden, but their binned relationships are not perfectly linear. A large credit amount is not automatically risky when income, existing debt, and payment behavior support it. The ratios are screening features, not independent approval rules.

### Segmentation supports capacity allocation

The dashboard's rule-based default rate rises from 4.66% in the low-risk group to 15.06% in the very-high-risk group. This monotonic separation supports prioritizing operational review. It does not by itself justify automatic rejection, because threshold choice also depends on credit policy, customer value, and review capacity.

## Model Decision

LightGBM was selected as the practical champion. On the 61,504-row validation set it achieved:

| Metric | Result | Interpretation |
|---|---:|---|
| ROC-AUC | 0.7907 | Strong ranking separation across thresholds |
| PR-AUC | 0.3127 | Meaningful precision-recall performance on the minority default class |
| KS | 0.437 | Clear separation between cumulative good and bad borrower scores |
| Lift@10 | 3.66x | The top score decile concentrates default risk 3.66 times above baseline |
| Top 30% recall | 69.1% | Reviewing 30% of applications captures about 69.1% of validation defaults |
| Top 30% lift | 2.30x | The selected review group is 2.30 times more efficient than random selection |

The weighted ensemble did not materially improve AUC, so LightGBM was retained to reduce operational complexity. Logistic Regression remains useful for diagnostic interpretation, not as the production champion.

## Governance Conclusion

SHAP identifies external credit scores and affordability ratios as leading signals. However, external scores are partially black-box, and occupation or organization can act as proxy variables for protected characteristics. Removing sensitive features causes little AUC loss, but fairness gaps can remain through correlated variables.

The recommended operating model is human-in-the-loop:

1. Use model scores to order applications for review.
2. Apply documented policy rules and affordability checks before the final decision.
3. Monitor performance, calibration, and group gaps over time.
4. Provide reason codes and escalation routes for adverse outcomes.
5. Retrain or recalibrate when population or policy drift is detected.

## Scope and Limitations

- This is a portfolio case study using public, anonymized data; it is not a deployed lending policy.
- Validation metrics are measured on labeled training records, not the unlabeled Kaggle test set.
- Reported associations are analytical evidence, not causal claims.
- Thresholds must be adjusted to real loss costs, approval strategy, capacity, and regulation before production use.
