# Customer Churn-Risk Segmentation (RFM + K-Means)

Segments customers by buying behaviour (Recency, Frequency, Monetary value) with K-Means to flag churn-risk groups for targeted retention.

Originated as an internship project at CodeValue Technologies (Python, Power BI).

## Method
1. Clean data: drop missing `CustomerID`, remove non-positive quantity/price.
2. Compute RFM per customer (snapshot = last invoice date + 1 day).
3. Log-transform + standardize RFM.
4. Elbow method (k=2–10) and silhouette score; fit K-Means with k=4.
5. Name segments from cluster averages: Champions, Loyal, Recent, At Risk.

## Run
```bash
pip install -r requirements.txt
python churn_rfm_segmentation.py
```
The script downloads the dataset automatically into `data/Online_Retail.xlsx` on first run. If the download fails, get it from https://archive.ics.uci.edu/dataset/352/online+retail and save it at that path.

Outputs (in `outputs/`): `rfm_segments.csv`, `elbow_curve.png`, `segment_profiles.png`, `segment_share.png`.

## Results
![Elbow curve](outputs/elbow_curve.png)
![Segment profiles](outputs/segment_profiles.png)
![Segment share](outputs/segment_share.png)

## Dataset
UCI Online Retail (transactions of a UK online retailer, 2010–2011).

## Notes
- This is behavioural segmentation; no supervised churn label is trained.
- The Power BI dashboard (RFM Insights) was built separately; add screenshots to `docs/`.

## Author
<your name>
