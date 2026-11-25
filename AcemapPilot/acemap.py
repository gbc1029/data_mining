import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent
ENDPOINT = "https://acemap.info/api/v1/work/search"
HEADERS = {"User-Agent": "AI3602-PilotProject/1.0", "Accept": "application/json"}


def normalize(text):
    return " ".join(text.lower().replace("-", " ").replace("_", " ").split())


def snapshot_path(query, size, *, order_by=None):
    snapshot_key = f"{query}\n{size}"
    if order_by:
        snapshot_key += f"\n{order_by}"
    key = hashlib.sha256(snapshot_key.encode()).hexdigest()[:20]
    return ROOT / "data" / "responses" / f"{key}.json"


class SnapshotMissing(ValueError):
    """Replay requested a query that was never recorded."""


def fetch(query, size=20, replay=False, save=True, *, order_by=None):
    path = snapshot_path(query, size, order_by=order_by)
    if replay:
        if not path.exists():
            raise SnapshotMissing(f"no snapshot for query={query!r} size={size} order_by={order_by!r}")
        return json.loads(path.read_text(encoding="utf-8"))["response"]["results"]
    params = {"keyword": query, "order": "desc", "page": 1, "size": size}
    if order_by:
        params["order_by"] = order_by
    response = requests.get(ENDPOINT, params=params, headers=HEADERS, timeout=30)
    response.raise_for_status()
    payload = response.json()
    record = {
        "url": response.url,
        "query": query,
        "size": size,
        "status": response.status_code,
        "elapsed_seconds": round(response.elapsed.total_seconds(), 4),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "response": payload,
    }
    if save:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload["results"]
