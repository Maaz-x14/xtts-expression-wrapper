
## Cross-actor reference selection

Maintain a small pool of candidate clips per emotion (different actors and/or different statements/reps),

At inference time select/score/rank — whichever clip's prosodic profile most plausibly matches your target sentence, rather than always using Actor 01.

**Why it could work, mechanically:** 

The Perceiver conditioning problem you're fighting is duration/rhythm mismatch between reference and target.

**Candidate selection mechanisms, cheapest to most involved:**

1. **Duration-based heuristic (no model, no training).** Estimate target sentence duration (word count × avg syllable duration for the language, or a cheap TTS pass in neutral first to get actual duration), then pick whichever candidate reference clip has closest recorded duration. Cheap, deterministic, explainable — but crude, since duration alone doesn't capture rhythm/stress pattern.
2. **Prosodic feature matching (lightweight signal processing, no training).** Extract pitch contour shape, syllable count, or speech rate (via librosa/parselmouth) from both candidate clips and a synthesized-neutral pass of your target text, then pick the reference whose contour shape correlates best. More principled than duration alone, still no model training — this is roughly what the ReStyle-TTS paper's own evaluation methodology uses (Parselmouth for pitch, in Appendix B) just repurposed as a *selection* signal instead of a *training* signal.
3. **Emotion2vec-style embedding similarity (pretrained model, no training).** Use an off-the-shelf pretrained speech emotion embedding model (e.g. emotion2vec, mentioned in the paper) to score each candidate clip's emotion embedding, and just pick the single most confidently-classified clip per emotion once, offline — not per-inference, just a one-time better clip selection than "Actor 01 by default." This doesn't solve duration mismatch at all, it solves "is this actually the most convincing angry clip available in the dataset," a different (also real) problem from the one you diagnosed.
4. **Best-of-N sampling with a scoring pass (no training, more compute).** Generate the same sentence with N candidate reference clips, score each output automatically (WavLM speaker similarity to confirm timbre held, emotion2vec confidence for emotion strength), pick the best-scoring result at inference time. Most robust, but costs N× inference time per request — expensive on your CPU setup, more viable once you're on Kaggle GPU.

**Why it might not work / real risks:**

- **Cross-actor introduces a timbre confound you already flagged in your own plan.** Different actors have genuinely different voices; if the wrapper's whole point is speaker-agnostic emotion tagging (or eventually your Urdu speaker's cloned voice), swapping actors per emotion means your speaker identity is no longer stable across emotions — a real regression, not free.
- **RAVDESS is small (24 actors, fixed sentences).** You don't have unlimited candidates to search over; the marginal gain from "better selection among 4-8 mediocre-fit clips" may be small compared to the fundamental problem the paper names — that zero-shot reference-conditioned TTS *structurally* imports reference-audio style regardless of which reference you pick.
- **No training-free selection mechanism fixes rhythm mismatch for arbitrary-length target text.** Even your best-matching reference clip was recorded for "Kids are talking by the door" or "Dogs are sitting by the door" — 6 words. A future 15-word Urdu sentence has no good RAVDESS match at all. Selection helps at the margin; it doesn't remove the ceiling.
- **This is unvalidated conjecture on my part**, not benchmarked: I don't have evidence that duration/prosody-matching selection meaningfully improves convincingness for XTTS specifically — the paper doesn't test this, it tests LoRA-based control instead. Treat mechanism (2) as a real hypothesis to test, not a proven fix.

**My actual recommendation for research conclusion:** given your no-training constraint, mechanism (1) or (2) are the only realistic candidates — cheap, no new dependencies beyond librosa (already in your stack) or parselmouth (light add). Mechanism (3)/(4) add real value but are heavier and better suited after you've validated whether selection helps at all. This research doesn't change your sequencing — you're doing A first regardless — but it tells you *if* B is worth building later, start with duration/prosody heuristic matching, not cross-actor emotion-embedding scoring, since the latter solves a different problem than the one your Step 2 data actually pointed at.

Ready for you to move to implementing (A) whenever you want — say the word and I'll ask the same structural questions I asked before touching `reference_map.py` last time (clip source scope, selection param naming, etc.) rather than assume.
