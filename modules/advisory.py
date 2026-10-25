def generate_advisories(fusion):
    if fusion is None or fusion.empty: return []
    advisories = []
    mean_score = float(fusion["engagement_index"].mean())
    if mean_score < 45:
        advisories.append("Aggregate signal indicator is low. Consider a quick poll, think-pair-share, or a short stretch break; check context before interpreting the signal.")
    elif mean_score < 65:
        advisories.append("Aggregate signal indicator is moderate. Consider asking a comprehension-check question or inviting a brief discussion.")
    else:
        advisories.append("Aggregate signal indicator is elevated. Maintain the current activity, and continue checking understanding directly.")
    if "estimated_class" in fusion and fusion["estimated_class"].astype(str).str.contains("High ambient").any():
        advisories.append("One or more windows show high ambient acoustic energy. Determine whether this is productive group discussion or disruptive noise before changing the activity.")
    advisories.append("Use direct, inclusive feedback from learners alongside these signals. Do not make individual-level judgments from this dashboard.")
    return advisories
