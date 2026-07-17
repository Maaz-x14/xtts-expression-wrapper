
## The core constraint you need to internalize

- fearful → maps to fear/anxiety/panic/nervousness
- disgust → maps to contempt/revulsion/disdain
- surprised → maps to shock/amazement/disbelief
- calm → maps to serenity/relaxation/calmness

---

## The actual problem with expanding emotions

Does XTTS-v2 actually transfer emotion reliably from reference audio, or is it primarily transferring speaker voice?

Is the transfer consistent enough that a `<fearful>` reference clip actually sounds fearful, or does it just sound like a different speaker?

Phase 2 measurement question.

---

## Proposed plan

**Step 1 — Add Tier 1 emotions first (RAVDESS, no new data)**
Map all 8 RAVDESS emotions. Add sub-emotion aliases pointing to the closest base emotion. For example: `<anxious>` → fearful.wav, `<rage>` → angry.wav, `<grief>` → sad.wav. This is just a lookup table change, 20 minutes of work.

**Step 2 — Quick subjective evaluation**
Run one sentence through all 8 base emotions. Listen. Does each one sound distinctly different? If fearful, disgust, surprised all sound the same as neutral, then the Perceiver isn't picking up fine-grained emotion from those clips and expanding further is pointless.

**Step 3 — Decide on Tier 2 based on Step 2 results**
If Tier 1 transfer is strong → source EmoV-DB clips for the missing emotions and expand. If transfer is weak → we skip to Level 3 (latent mixing) which gives us actual control.

**Step 4 — Implement remaining prosodic tags**
`<fast>`, `<slow>`, `<break>`, `<silence>` are already in the parser. The wrapper already handles them. So these are already done — just need verification.

---

## My honest assessment

The sub-emotion aliases (joy → happy, grief → sad) are cheap and worth doing now. The Tier 2 emotions requiring new data should wait until Step 2 tells us if emotion transfer is actually working well enough to justify sourcing more clips.

Does this order make sense to you, or do you want to push Tier 2 immediately?
