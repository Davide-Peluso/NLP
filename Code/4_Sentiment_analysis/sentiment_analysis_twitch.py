from __future__ import annotations
import os, re, json, random, emoji
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import numpy as np, pandas as pd, torch
from tqdm.auto import tqdm
from transformers import pipeline

SEED = 42
CHUNK_DEFAULT = 200_000          
BATCH_GPU, BATCH_CPU = 32, 8
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps")  if torch.backends.mps.is_available()
    else torch.device("cpu")
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

EMOTES = [
    "Kappa","PogChamp","LUL","4Head","Kreygasm","BibleThump","FailFish",
    "ResidentSleeper","WutFace","HeyGuys","KappaPride","DansGame","VoHiYo",
    "NotLikeThis","CoolStoryBob","BabyRage","KappaRoss","MrDestructoid",
    "PepeLaugh","FeelsBadMan","FeelsGoodMan","OMEGALUL","monkaS","Pog",
    "EZ","Sadge","KEKW"
]
EMOTE_RE  = re.compile(r"\b(?:" + "|".join(map(re.escape,EMOTES)) + r")\b", re.I)
URL_RE, MENTION_RE = re.compile(r"https?://\S+"), re.compile(r"@\w+")
EMOTE_TAG = {e.lower(): f"<EMO_{e.upper()}>" for e in EMOTES}

def clean_for_sent(txt:str)->str:
    txt = URL_RE.sub(" ", str(txt))
    txt = MENTION_RE.sub(" ", txt)
    txt = EMOTE_RE.sub(lambda m: EMOTE_TAG[m.group(0).lower()], txt)
    txt = emoji.demojize(txt, language="en")
    txt = re.sub(r"\s{2,}", " ", txt)
    return txt.strip()


def run_sentiment_pipeline(
    infile: str | Path,
    out_dir: str | Path,
    *,
    chunk_rows:int = CHUNK_DEFAULT,
    batch_size:int | None = None,
):

    infile, out_dir = Path(infile), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("device:", DEVICE)

    batch_size = batch_size or (BATCH_GPU if DEVICE.type=="cuda" else BATCH_CPU)

    sent_pipe = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        device=0 if DEVICE.type=="cuda" else -1,
    )

    tox_repo = "textdetox/xlmr-large-toxicity-classifier-v2"
    tox_pipe = pipeline(
        "text-classification",
        model=tox_repo,
        tokenizer=tox_repo,
        device=0 if DEVICE.type=="cuda" else -1,
    )

    stats: Dict[str,int] = dict(rows=0,toxic=0,pos=0,neg=0,neu=0)

    total_rows = sum(1 for _ in open(infile, encoding="utf-8", errors="ignore")) - 1
    chunk_rows = min(chunk_rows, total_rows) if total_rows < chunk_rows else chunk_rows

    for idx, df in enumerate(pd.read_csv(infile, chunksize=chunk_rows)):
        print(f"— chunk {idx:03d} (len={len(df):,})")
        df["msg_clean"] = df["message"].map(clean_for_sent)
        texts:List[str] = df["msg_clean"].tolist()

        sent_lbl, sent_scr, tox_lbl, tox_scr = [], [], [], []

        for i in tqdm(range(0, len(texts), batch_size), leave=False):
            batch = texts[i:i+batch_size]

            outs = sent_pipe(batch, truncation=True, padding=True, max_length=512)
            sent_lbl += [o["label"].lower()[:3] for o in outs]   
            sent_scr += [round(o["score"], 4) for o in outs]

            outs = tox_pipe(batch, truncation=True, padding=True, max_length=512)
            tox_lbl  += [o["label"].lower() for o in outs]      
            tox_scr  += [round(o["score"], 4) for o in outs]


        df["sentiment_label"], df["sentiment_score"] = sent_lbl, sent_scr
        df["toxicity_label"],  df["toxicity_score"]  = tox_lbl, tox_scr
        df["toxic"] = df["toxicity_label"].eq("toxic")

        stats["rows"]  += len(df)
        stats["toxic"] += int(df["toxic"].sum())
        stats["pos"]   += int((df["sentiment_label"] == "pos").sum())
        stats["neg"]   += int((df["sentiment_label"] == "neg").sum())
        stats["neu"]   += int((df["sentiment_label"] == "neu").sum())

        df.to_parquet(out_dir / f"sent_chunk_{idx:03d}.parquet", index=False)

    summary = {
        "rows": stats["rows"],
        "pct_toxic": round(stats["toxic"] / stats["rows"], 4),
        "sentiment": dict(pos=stats["pos"], neu=stats["neu"], neg=stats["neg"]),
    }
    meta = {
        "datetime": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "device": str(DEVICE),
        "chunk_rows": chunk_rows,
        "batch_size": batch_size,
        "models": {
            "sentiment": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
            "toxicity":  tox_repo,
        },
        "summary": summary,
    }
    with open(out_dir / "sentiment_run_metadata.json", "w") as fp:
        json.dump(meta, fp, indent=2)

    print("   Toxicity:", f"{summary['pct_toxic']*100:.2f}%")

run_sentiment_pipeline(
    infile="filt.csv",
    out_dir="sent_test_out_t",     
)