# Online News Popularity - ML Challenge

Predicts the 5-class popularity (A-E) of Mashable news articles. Metric: Macro F1.

## Approach
1. **Preprocessing:** missing values (`NA`) are left as NaN; all three models handle them natively.
2. **Feature engineering:** link/media/token ratios, keyword ranges, log transforms,
   sentiment gap, LDA max/argmax, missing-value count (see `fe()` in `train.py`).
3. **Models:** LightGBM, XGBoost and CatBoost, each trained with stratified 5-fold CV
   with early stopping. Out-of-fold (OOF) and test probabilities are averaged across folds
   [and seeds].
4. **Blend:** simple average of the three models' probabilities.
5. **Macro-F1 optimisation:** per-class probability multipliers are tuned by coordinate
   search on the blended OOF predictions, then applied to the test predictions before argmax.
   Held-out check of this tuning: about +0.06 macro F1 over plain argmax.
6. **Validation:** blend raw OOF macro F1 about 0.265; tuned about 0.33 (honest estimate about 0.32-0.33).

No external data and no pretrained models were used.

## Files
- `news-popularity-model.ipynb`: preprocessing, feature engineering, training, tuning, prediction.
- `requirements.txt`: dependencies.