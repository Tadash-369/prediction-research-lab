CHAMINI_SP_GOD_MODE_KEY = "chamini_sp_god_mode"
CHAMINI_SP_GOD_MODE_LABEL = "ChaminiSP God Mode"
CHAMINI6_LEGACY_GOD_MODE_KEY = "chamini6_god_mode"
CHAMINI6_LEGACY_GOD_MODE_LABEL = "Chamini6 God Mode"


MODEL_ALIAS = {
    CHAMINI6_LEGACY_GOD_MODE_KEY: CHAMINI_SP_GOD_MODE_KEY,
    CHAMINI6_LEGACY_GOD_MODE_LABEL: CHAMINI_SP_GOD_MODE_KEY,
    "chamini6": CHAMINI_SP_GOD_MODE_KEY,
    "god_mode": CHAMINI_SP_GOD_MODE_KEY,
    "chaminisp": CHAMINI_SP_GOD_MODE_KEY,
    "chamini_sp": CHAMINI_SP_GOD_MODE_KEY,
}

MODEL_DISPLAY_NAME = {
    CHAMINI_SP_GOD_MODE_KEY: CHAMINI_SP_GOD_MODE_LABEL,
    CHAMINI6_LEGACY_GOD_MODE_KEY: CHAMINI_SP_GOD_MODE_LABEL,
    CHAMINI6_LEGACY_GOD_MODE_LABEL: CHAMINI_SP_GOD_MODE_LABEL,
}


def normalize_model_key(model_key):
    text = str(model_key or "").strip()
    return MODEL_ALIAS.get(text, text)


def get_model_display_name(model_key):
    text = str(model_key or "").strip()
    normalized = normalize_model_key(text)
    return MODEL_DISPLAY_NAME.get(text, MODEL_DISPLAY_NAME.get(normalized, text))


def is_chamini_sp_model(model_key):
    return normalize_model_key(model_key) == CHAMINI_SP_GOD_MODE_KEY
