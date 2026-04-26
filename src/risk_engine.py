def classify_risk(result: dict):
    flags = []
    risk = "LOW"

    # -------------------------
    # HIGH RISK CONDITIONS
    # -------------------------
    if result["in_water"]:
        risk = "HIGH"
        flags.append("Inside water body")

    if result["in_restricted"]:
        risk = "HIGH"
        flags.append("Inside restricted/protected area")

    # -------------------------
    # MEDIUM RISK CONDITIONS
    # -------------------------
    if risk != "HIGH":
        if result["near_water"]:
            risk = "MEDIUM"
            flags.append("Near water body")

        if result["near_forest"]:
            risk = "MEDIUM"
            flags.append("Near forest zone")

        if result["near_restricted"]:
            risk = "MEDIUM"
            flags.append("Near restricted area")

    # -------------------------
    # LOW RISK
    # -------------------------
    if not flags:
        flags.append("No major environmental or legal risks detected")

    # -------------------------
    # EXPLANATION (IMPORTANT)
    # -------------------------
    explanation = " | ".join(flags)

    return {
        "risk_level": risk,
        "flags": flags,
        "explanation": explanation
    }