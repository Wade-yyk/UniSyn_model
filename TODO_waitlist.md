# Waitlist

## Future TTS Prosody Experiment

- Change TTS `pos` generation in `prepare_filelists.py` from flat `1.0` to syllable-internal positions.
- Suggested rule: optional initial + toned final form one syllable, e.g. `w uo3 -> 0.5 1.0`.
- Regenerate `filelists/unisyn_train.txt` and `filelists/unisyn_val.txt`.
- Train a new run or fine-tune explicitly for the new `pos` distribution.
- When using that future checkpoint, run `free_tts_infer.py` with `--syllable-pos`.

Reason: current flat TTS `pos` may contribute to choppy, one-character-at-a-time speech in free-text inference.
