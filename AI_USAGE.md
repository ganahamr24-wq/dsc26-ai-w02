# AI Usage Log

- **Tools used**: ChatGPT / Gemini
- **Functions helped with**: `rawaj/audit.py` (leakage checks) and `rawaj/metrics.py` (`best_threshold`).
- **Where the AI was incorrect**: When generating `best_threshold`, the AI used `np.argmin` directly without handling ties correctly (it selected the lowest threshold on ties instead of the highest threshold as specified in the rules). I noticed this by comparing it against the spec requirements.
- **Function written without AI**: `confusion_at` in `metrics.py`, because it is a straightforward count of TP, FP, TN, and FN.