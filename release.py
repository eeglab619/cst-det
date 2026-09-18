RELEASE_NOTICE = 'Core implementation is withheld during peer review and will be released in this repository after the accompanying paper is accepted.'
CORE_AVAILABLE = False

def require_core():
    if not CORE_AVAILABLE:
        raise NotImplementedError(RELEASE_NOTICE)
