"""Customer churn-risk segmentation using RFM analysis + K-Means (Online Retail dataset)."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

DATA_PATH = Path("data/Online_Retail.xlsx")
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
K = 4


UCI_ZIP = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"


def download(path=DATA_PATH):
    import io, urllib.request, zipfile
    path.parent.mkdir(exist_ok=True)
    print("Downloading Online Retail dataset...")
    try:
        raw = urllib.request.urlopen(UCI_ZIP, timeout=120).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".xlsx"))
            path.write_bytes(z.read(name))
    except Exception as e:
        raise SystemExit(f"Auto-download failed ({e}). Get the file from https://archive.ics.uci.edu/dataset/352/online+retail and save it as {path}")


def load(path=DATA_PATH):
    if not path.exists():
        download(path)
    return pd.read_excel(path)


def clean(df):
    df = df.dropna(subset=["CustomerID"])
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["Total"] = df["Quantity"] * df["UnitPrice"]
    return df


def build_rfm(df):
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        MonetaryValue=("Total", "sum"),
    )
    return rfm


def cluster(rfm):
    X = StandardScaler().fit_transform(np.log1p(rfm[["Recency", "Frequency", "MonetaryValue"]]))
    inertia = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).inertia_ for k in range(2, 11)]
    plt.figure(); plt.plot(range(2, 11), inertia, marker="o")
    plt.title("Elbow Curve"); plt.xlabel("k"); plt.ylabel("Inertia"); plt.grid(True)
    plt.savefig(OUT / "elbow_curve.png", dpi=150, bbox_inches="tight"); plt.close()

    km = KMeans(n_clusters=K, n_init=10, random_state=42)
    rfm["Cluster"] = km.fit_predict(X)
    print(f"Silhouette (k={K}): {silhouette_score(X, rfm['Cluster']):.3f}")
    return rfm


def label_segments(rfm):
    m = rfm.groupby("Cluster")[["Recency", "Frequency", "MonetaryValue"]].mean()
    names = {}
    names[m["MonetaryValue"].idxmax()] = "Champions"
    rest = m.drop(index=list(names))
    names[rest["Recency"].idxmax()] = "At Risk"
    rest = m.drop(index=list(names))
    names[rest["Frequency"].idxmax()] = "Loyal"
    names[next(c for c in m.index if c not in names)] = "Recent"
    rfm["Segment"] = rfm["Cluster"].map(names)
    return rfm


def plots(rfm):
    s = rfm.groupby("Segment")[["Recency", "Frequency", "MonetaryValue"]].mean()
    s.plot(kind="bar", subplots=True, legend=False, figsize=(7, 8), title="Average RFM by Segment")
    plt.savefig(OUT / "segment_profiles.png", dpi=150, bbox_inches="tight"); plt.close()
    rfm["Segment"].value_counts().plot(kind="pie", autopct="%1.1f%%", ylabel="", title="Customers per Segment")
    plt.savefig(OUT / "segment_share.png", dpi=150, bbox_inches="tight"); plt.close()


def main():
    rfm = label_segments(cluster(build_rfm(clean(load()))))
    plots(rfm)
    rfm.to_csv(OUT / "rfm_segments.csv")
    print(rfm.groupby("Segment").agg(Customers=("Cluster", "size"), Recency=("Recency", "mean"),
          Frequency=("Frequency", "mean"), Monetary=("MonetaryValue", "mean")).round(1))


if __name__ == "__main__":
    main()
