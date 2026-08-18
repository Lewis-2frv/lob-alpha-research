# Interpretable-to-nonlinear model ladder

Every candidate is evaluated on the same paired FI-2010 CF_1-CF_8 development folds. The ladder is
intended to show what additional predictive value is obtained as model flexibility increases.

## 1. Naive baselines

`AlwaysStationaryClassifier` always predicts publisher class 2. `DummyClassifier(strategy="prior")`
estimates class frequencies using the training member only. These establish non-informative reference
points for macro-F1 and log loss.

## 2. Manual liquidity-pressure rule

The first 40 FI-2010 dimensions contain the publisher's 10-level basic book representation in repeated
ask-price, ask-volume, bid-price, bid-volume order. For each snapshot, the rule forms a depth-weighted
standardised volume-pressure proxy

\[
s_{depth}=\sum_{i=1}^{10} w_i (V^{bid}_i-V^{ask}_i),\qquad
w_i\propto i^{-1/2},
\]

then combines it with the sign-reversed accumulated ask-minus-bid volume-difference feature from the
publisher's `u5` block. Fixed logits `[s, b, -s]` represent up, stationary and down, respectively.
No label is used to fit this model. Because the repository uses the publisher's Z-score representation,
this is deliberately described as a **standardised liquidity-pressure proxy**, not a raw queue-imbalance
ratio.

## 3. NumPy shared-diagonal LDA

The classifier estimates class means and a pooled per-feature variance from the training member only.
A shrinkage parameter moves the pooled diagonal variance toward its across-feature mean to stabilise
near-constant dimensions. With equal class priors, the discriminant score is

\[
\delta_k(x)=x^T\Sigma_d^{-1}\mu_k-
\frac{1}{2}\mu_k^T\Sigma_d^{-1}\mu_k.
\]

The implementation is written directly in NumPy and avoids materialising a dense 144x144 covariance
matrix.

## 4. NumPy multiclass ridge

Fold-local mean and standard deviation are fitted on training rows only. The model then solves a
class-balanced weighted least-squares problem against one-hot class targets:

\[
\hat W=(X^T S X+\lambda R)^{-1}X^T S Y,
\]

where `S` contains training-only class weights and the bias term is not regularised. The sufficient
statistics are accumulated in chunks to keep memory bounded on `Train_CF_9`.

## 5. NumPy multinomial softmax

The softmax model uses the same fold-local standardisation and training-only balanced class weights.
For logits `z_k=x^T w_k`, probabilities are

\[
p(y=k\mid x)=\frac{\exp z_k}{\sum_j\exp z_j}.
\]

Parameters are trained directly in NumPy using deterministic mini-batch Adam and L2 regularisation.
This provides a transparent linear probabilistic model before the sklearn and tree ensembles.

## 6. sklearn SGD log-loss

Two L2 settings provide a production-library linear benchmark. Scaling and early-stopping state are
fitted inside each training fold only.

## 7. Histogram gradient boosting

The previously audited sklearn `HistGradientBoostingClassifier` remains in every environment. Keeping
it in the registered universe preserves the historical nonlinear benchmark while the v0.6 study is
regenerated.

## 8. LightGBM

When installed, nine fixed combinations of learning rate and leaf count are evaluated with a fixed
180-tree budget and fixed regularisation/subsampling settings. LightGBM is the final flexible tabular
model, not the starting point of the experiment.

## Selection

The model with the highest mean macro-F1 over CF_1-CF_8 wins. Worst-fold macro-F1 is the first
tie-breaker, followed by lower registered complexity and then stable specification ID. CF_9 is never
used to tune the model family, hyperparameters or confidence threshold.

## Feature-layout reference

The FI-2010 feature-family layout used by the manual rule follows Table 4 of Ntakaris et al.,
*Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data with Machine Learning Methods*,
DOI `10.1002/for.2543`. Dataset attribution and licence details are in `NOTICE-FI2010.md`.
