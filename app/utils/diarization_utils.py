def build_speaker_map(labels):
    uniq = []
    for lb in labels:
        if lb not in uniq:
            uniq.append(lb)
    return {orig: f"S{i}" for i, orig in enumerate(uniq)}

def to_rttm(segments):
    lines = []
    for s in segments:
        dur = max(0, s["end"] - s["start"])
        lines.append(
            f"SPEAKER unknown 1 {s['start']:.3f} {dur:.3f} <NA> <NA> {s['speaker']} <NA> <NA>"
        )
    return "\n".join(lines)
