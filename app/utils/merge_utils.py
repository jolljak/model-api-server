def segments_to_text(segs):
    lines = []
    for s in segs:
        lines.append(f"[{s['start']:.1f}~{s['end']:.1f}] {s['speaker']}: {s['text']}")
    return "\n".join(lines)


def assign_speakers(asr_segments, diar_segments):
    def overlap(a1, a2, b1, b2):
        return max(0, min(a2, b2) - max(a1, b1))

    out = []
    for a in asr_segments:
        best = "UNK"
        best_ov = 0
        for d in diar_segments:
            ov = overlap(a["start"], a["end"], d["start"], d["end"])
            if ov > best_ov:
                best_ov = ov
                best = d["speaker"]

        out.append({
            "start": a["start"],
            "end": a["end"],
            "speaker": best,
            "text": a["text"]
        })

    return out
