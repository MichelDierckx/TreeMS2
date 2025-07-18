# UTILS


def format_execution_time(seconds: float) -> str:
    millis = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)

    parts = []

    days, seconds = divmod(seconds, 86400)
    if days:
        parts.append(f"{days}d")

    hours, seconds = divmod(seconds, 3600)
    if hours:
        parts.append(f"{hours}h")

    minutes, seconds = divmod(seconds, 60)
    if minutes:
        parts.append(f"{minutes}m")

    if seconds:
        parts.append(f"{seconds}s")

    if millis:
        parts.append(f"{millis}ms")

    return " ".join(parts) if parts else "0ms"
