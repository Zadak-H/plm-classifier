# Data folder

Place your input CSV files here. The tool accepts **any CSV** — the column names are configurable in the GUI.

---

## Training / classification CSV

Must have one sequence column and one label column. Labels can be strings or integers. Any additional columns are ignored.

```csv
name,sequence,label
WT,MKTIIALSYIFCLVFA,active
V12A,MKTIIALSAIFCLVFA,inactive
K5R,MRTIIALSYIFCLVFA,active
L17F,MKTIIALSYIFCLVFF,active
D3N,MNTIIALSYIFCLVFA,inactive
```

Binary or multiclass — both work without any extra configuration.

---

## Embed-only CSV (no label needed)

Just sequences. Useful for generating embeddings before labels are available, or for the **Embed** tab.

```csv
id,sequence
seq1,MKTIIALSYIFCLVFA
seq2,MKTIIALSAIFCLVFA
seq3,MRTIIALSYIFCLVFA
```

---

## Predict CSV

Same format as embed-only — sequences you want class predictions for after a model is trained.

```csv
id,sequence
candidate1,MKTIIALSYIFFLVFA
candidate2,MKTIIALSYIYCLVFA
```

---

## Tips

- **Column names**: set them in the GUI dropdowns — no renaming required.
- **Labels**: use any consistent label — `0/1`, `True/False`, `active/inactive`, `A/B/C`, etc.
- **Positive class** (for binary tasks): select the label that counts as "positive" in the Predict tab to get calibrated probabilities.
- **Sequences**: standard 20-letter amino-acid alphabet. Non-standard residues (B, U, Z, O, X) are handled by most backends.
