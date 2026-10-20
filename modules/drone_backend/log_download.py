"""MAVLink DataFlash log download, Mission-Planner style.

ArduPilot writes its DataFlash log (``.BIN``) to its own storage - an SD card on
a real FCU, a local file in SITL. It starts a new log automatically every arm
and finalises it on disarm. There is no MAVLink message that "starts recording":
the FCU already does that. What this module does is pull that log off the
vehicle over MAVLink, which is what Mission Planner's *Download DataFlash Log
Via MAVLink* does, so the log survives even when the SD card is not coming back
with the airframe.

Protocol (MAVLink log-download), following MAVProxy's reference module:

1. ``LOG_REQUEST_LIST`` (start id 0, end id ``0xFFFF``) asks the autopilot to
   enumerate its logs; it answers with one ``LOG_ENTRY`` per log (id, size,
   time). The total count arrives in the ``num_logs`` field of any entry.
2. For each log, **one** ``LOG_REQUEST_DATA`` (id, ofs=0, count huge) is sent.
   ArduPilot then **streams** the whole log back as a burst of ``LOG_DATA``
   messages in 90-byte blocks; it does not expect a request per chunk. (Issuing
   a fresh small request for every chunk instead makes the FCU fragment and
   interleave replies, which stalls the download a few hundred bytes in.)
3. Each block is written straight into the file **at its own ``ofs``**, so a
   re-sent or out-of-order reply lands in the right place and arrival order is
   never trusted. Blocks that never arrived are re-requested in a batch, so a
   lossy link still finishes; a short (<90 byte) or zero-count reply that
   extends the log marks the true end.

The module holds no MAVLink state of its own: the caller supplies a ``master``
(a pymavlink connection) and this code only sends requests and consumes the
``LOG_ENTRY`` / ``LOG_DATA`` replies the master already routes to it. That keeps
the download loop off the shared ''recv_match'' call that the live telemetry
listener uses - the caller runs it on its own thread.

Nothing here raises on a partial download: a link that drops mid-transfer leaves
a shorter file and a status string that says so, rather than an exception in a
daemon thread.
"""
from __future__ import annotations

import os
import time

try:
    from pymavlink import mavutil
except ImportError:  # pragma: no cover - dependency may be absent during static checks
    mavutil = None

from modules.logging_util import log as _log


# ArduPilot serves a log in fixed 90-byte ``LOG_DATA`` blocks; a reply smaller
# than this is the documented end-of-log signal.
LOG_DATA_MAX_BYTES = 90

# Ask for the *whole* log with one ``LOG_REQUEST_DATA``. ArduPilot then streams
# it back rather than expecting a request per chunk; MAVProxy uses ``0xFFFFFFFF``
# for exactly this reason. A per-chunk request model makes the FCU fragment and
# interleave replies, which stalls the download a few hundred bytes in.
LOG_STREAM_COUNT = 0xFFFFFFFF

# When the stream stalls, re-request the missing blocks in at most this many
# spans per pass (MAVProxy uses 20).
MISSING_RETRY_BATCH = 20

# How long to wait for the next LOG_DATA / LOG_ENTRY before giving up. After a
# stall we first try to re-request missing blocks; only a silence this long with
# nothing left to request ends the download.
CHUNK_TIMEOUT_S = 5.0
ENTRY_TIMEOUT_S = 5.0

# Tags are limited to this set across the app; log download uses [LOG].
_TAG = "LOG"


def _safe_name(text, fallback="log"):
    """One path-safe filename component; never returns an empty string."""
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(text))
    cleaned = cleaned.strip("._")
    return cleaned or fallback


def log_filename(entry, vehicle_type="vehicle", stamp=None):
    """Filename for one downloaded log.

    ``<vehicle_type>-<id>-<YYYYMMDD-HHMMSS>.bin`` mirrors Mission Planner's
    naming closely enough to be recognisable, while carrying the FCU log id so
    two downloads of the same vehicle never collide.
    """
    when = time.localtime(stamp if stamp is not None else time.time())
    stamp_text = time.strftime("%Y%m%d-%H%M%S", when)
    return "{}-{}-{}.bin".format(
        _safe_name(vehicle_type, "vehicle"),
        int(entry.get("id", 0)),
        stamp_text,
    )


def request_log_list(master, timeout=ENTRY_TIMEOUT_S):
    """Ask the autopilot to enumerate its logs; return ``{id: entry}``.

    Sends ``LOG_REQUEST_LIST`` and consumes ``LOG_ENTRY`` replies until the
    expected ``num_logs`` ids have arrived, the autopilot reports it has none,
    or ``timeout`` elapses after the last entry. Sends the request a second time
    on a short deadline, because the first request on a fresh link is easily
    lost and Mission Planner retries for the same reason.
    """
    entries = {}
    total = None
    deadline = time.time() + timeout
    sent_again = False
    _send_request_list(master)
    while time.time() < deadline:
        if not sent_again and time.time() > deadline - timeout + max(timeout / 2.0, 1.0):
            _send_request_list(master)
            sent_again = True
        msg = _recv_log_message(master, timeout=0.5)
        if msg is None:
            continue
        if msg.get_type() == "LOG_ENTRY":
            entry = _entry_from_msg(msg)
            entries[entry["id"]] = entry
            total = entry["num_logs"]
            deadline = time.time() + timeout
            if total and len(entries) >= total:
                break
            if total == 0:
                break
    return entries


def download_log(master, entry, directory, vehicle_type="vehicle", progress=None):
    """Download one log to ``directory``; return the written path or ``None``.

    ``entry`` is a dict from :func:`request_log_list`. This follows MAVProxy's
    reference implementation (``mavproxy_log.py``) rather than the
    one-request-per-chunk model:

    1. Send a **single** ``LOG_REQUEST_DATA`` with a very large ``count``
       (``LOG_STREAM_COUNT``). ArduPilot then **streams** the whole log back as
       a burst of ``LOG_DATA`` messages without a request per chunk. Issuing a
       new small request for every chunk instead makes the FCU interleave and
       fragment replies (observed live: 4-byte fragments and out-of-order
       answers that stall the download a few hundred bytes in).
    2. Follow **each reply's own ``ofs``**: ``seek(ofs)`` then write
       ``data[:count]``. The offset in the message is the single source of
       truth, so re-sent or out-of-order replies land in the right place.
    3. Track which 90-byte blocks arrived (``ofs // 90``). If the stream goes
       quiet before the log is complete, re-request **only the missing blocks**
       in a batch, exactly as MAVProxy does, so a lossy link still finishes.
    4. Stop at a short reply (``count < 90``) or a zero-count reply - the
       documented end-of-log signals. ArduPilot only sends a zero-count
       ``LOG_DATA`` when asked at or past ``size``; the final partial frame is
       therefore captured without trusting an approximate ``size``.

    ``size`` is used only as a progress denominator and a rough completion
    check, never as a hard stop. ``progress`` is an optional
    ``callable(done_bytes, total_bytes)``.
    """
    size = int(entry.get("size", 0))
    log_id = int(entry.get("id", 0))
    if size <= 0:
        _log(_TAG, f"Log {log_id} has no data (size 0); skipping")
        return None

    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, log_filename(entry, vehicle_type=vehicle_type))

    received = 0
    complete = False
    blocks = set()  # which 90-byte blocks have arrived
    with open(path, "wb") as handle:
        # Request the whole log in one shot; the FCU streams it back.
        _send_request_data(master, log_id, 0, LOG_STREAM_COUNT)
        last_data_time = time.time()
        while True:
            chunk = _next_chunk(master, log_id)
            if chunk is None:
                # Stream went quiet. Fill any holes the link dropped, then
                # decide whether we are done.
                if _request_missing(master, log_id, received, blocks):
                    last_data_time = time.time()
                    continue
                if time.time() - last_data_time >= CHUNK_TIMEOUT_S:
                    break
                continue
            chunk_ofs, chunk_count, chunk_data = chunk
            last_data_time = time.time()
            if chunk_count == 0:
                # End-of-log marker: ArduPilot sends this once asked at/past the
                # true end. Only meaningful if we have actually caught up, so a
                # spurious zero-count from a stale queue cannot cut us short.
                if chunk_ofs <= received and _finish(master, log_id, received, blocks):
                    complete = True
                    break
                last_data_time = time.time()
                continue
            handle.seek(chunk_ofs)
            handle.write(chunk_data)
            blocks.add(chunk_ofs // LOG_DATA_MAX_BYTES)
            end = chunk_ofs + len(chunk_data)
            advanced = end > received
            received = max(received, end)
            if progress is not None:
                progress(min(received, size), size)
            if chunk_count < LOG_DATA_MAX_BYTES and advanced:
                # A short reply *that extends the log* is the other documented
                # end-of-log signal - but only once every block below ``received``
                # has actually arrived. A gap means the link dropped data, so
                # re-request it and keep going instead of finishing short.
                if _finish(master, log_id, received, blocks):
                    complete = True
                    break
                last_data_time = time.time()
        handle.flush()

    if received == 0:
        try:
            os.remove(path)
        except OSError:
            pass
        return None
    if not complete:
        _log(_TAG, f"Log {log_id}: link went quiet at {received}/{size} bytes; "
                   f"file kept (short)")
    return path


def _finish(master, log_id, received, blocks):
    """True if the log is complete; otherwise re-request the gaps and return False.

    Called at a possible end-of-log marker (a short block or a zero-count
    reply). Completion is only real when every 90-byte block below ``received``
    has arrived; if not, the link lost data and the missing blocks are
    re-requested rather than accepting a short file.
    """
    if received <= 0:
        return False
    return not _request_missing(master, log_id, received, blocks)


def _request_missing(master, log_id, received, blocks):
    """Ask for the not-yet-received 90-byte blocks; return True if any were.

    ArduPilot delivers the log in 90-byte blocks. When the stream stalls before
    ``received``, the gap is the blocks below it we never saw. Re-request those
    (at most ``MISSING_RETRY_BATCH`` spans per pass) so a lossy link can still
    complete, mirroring MAVProxy's ``handle_log_data_missing``.
    """
    if received <= 0:
        return False
    missing = [block for block in range(received // LOG_DATA_MAX_BYTES)
               if block not in blocks]
    if not missing:
        return False
    requested = 0
    start = missing[0]
    prev = missing[0]
    for block in missing[1:]:
        if block == prev + 1:
            prev = block
            continue
        _send_request_data(master, log_id, start * LOG_DATA_MAX_BYTES,
                           (prev - start + 1) * LOG_DATA_MAX_BYTES)
        requested += 1
        start = prev = block
        if requested >= MISSING_RETRY_BATCH:
            break
    if requested < MISSING_RETRY_BATCH:
        _send_request_data(master, log_id, start * LOG_DATA_MAX_BYTES,
                           (prev - start + 1) * LOG_DATA_MAX_BYTES)
    _log(_TAG, f"Log {log_id}: re-requesting {len(missing)} missing blocks")
    return True


def session_log_entries(entries, since):
    """Entries that belong to an arming that began at ``since`` (epoch seconds).

    ArduPilot stamps each ``LOG_ENTRY`` with ``time_utc`` (seconds since the
    epoch). A log opened at or after ``since`` is this session's log. When the
    stamp is unusable - SITL and some FCUs report 0 until the clock is set - the
    filter cannot decide, so every entry is returned rather than silently
    downloading nothing.

    A small negative slack absorbs clock skew between the ground station and the
    FCU: the vehicle's clock can lag the host by a second or two, which would
    otherwise push its own log just before ``since``.
    """
    ordered = [entries[key] for key in sorted(entries)]
    if since is None or not ordered:
        return ordered
    stamped = [entry for entry in ordered if int(entry.get("time_utc", 0)) > 0]
    if not stamped:
        return ordered
    threshold = since - 2
    selected = [entry for entry in ordered if int(entry.get("time_utc", 0)) >= threshold]
    return selected or ordered[-1:]


def download_all(master, directory, vehicle_type="vehicle", progress=None, since=None):
    """Enumerate and download logs the autopilot holds.

    Returns ``(paths, entries)`` - the files written and the entries seen - so a
    caller can report "0 of 3" as easily as it reports what it fetched. With
    ``since`` set, only the log(s) from that arming are fetched; otherwise every
    log on the vehicle is.
    """
    entries = request_log_list(master)
    ordered = session_log_entries(entries, since)
    paths = []
    for index, entry in enumerate(ordered, start=1):
        size_kb = int(entry.get("size", 0)) / 1024.0
        _log(_TAG, f"Downloading log {entry['id']} "
                   f"({index}/{len(ordered)}, {size_kb:.1f} KiB)")
        path = download_log(master, entry, directory,
                            vehicle_type=vehicle_type, progress=progress)
        if path is not None:
            paths.append(path)
            _log(_TAG, f"Log {entry['id']} saved to {path}")
        else:
            _log(_TAG, f"Log {entry['id']} was not saved")
    if not ordered:
        _log(_TAG, "The vehicle reports no logs to download")
    return paths, entries


# --- MAVLink plumbing -------------------------------------------------------


def _send_request_list(master):
    # start=0, end=0xFFFF asks for every entry (Mission Planner's request form).
    master.mav.log_request_list_send(
        master.target_system, master.target_component, 0, 0xFFFF
    )


def _send_request_data(master, log_id, offset, count):
    master.mav.log_request_data_send(
        master.target_system, master.target_component, log_id, offset, count
    )


def _entry_from_msg(msg):
    return {
        "id": int(getattr(msg, "id", 0)),
        "num_logs": int(getattr(msg, "num_logs", 0)),
        "size": int(getattr(msg, "size", 0)),
        "time_utc": int(getattr(msg, "time_utc", 0)),
    }


def _recv_log_message(master, timeout):
    """One LOG_ENTRY / LOG_DATA reply, or None.

    ``recv_match`` with a type filter, so a stray heartbeat on the link does not
    get consumed here (the live telemetry listener owns those).
    """
    try:
        return master.recv_match(type=["LOG_ENTRY", "LOG_DATA"], timeout=timeout)
    except Exception:
        return None


def _next_chunk(master, log_id, timeout=0.5):
    """The next streamed ``(ofs, count, data)`` for ``log_id``, or None.

    ArduPilot streams the whole log back as a burst of ``LOG_DATA`` messages
    once ``LOG_REQUEST_DATA`` is sent with a large ``count``; there is no reply
    per request to match. Each reply is taken at face value and written where
    its own ``ofs`` says it belongs, so re-sent or out-of-order replies are
    harmless. ``count == 0`` is the end-of-log marker and is returned as-is.
    """
    msg = _recv_log_message_id(master, log_id, timeout=timeout)
    if msg is None:
        return None
    ofs = int(getattr(msg, "ofs", 0))
    count = int(getattr(msg, "count", 0))
    data = bytes(getattr(msg, "data", b""))[:count] if count else b""
    return ofs, count, data


def _recv_log_message_id(master, log_id, timeout):
    """One ``LOG_DATA`` for ``log_id``, or None; drops other logs' replies."""
    try:
        while True:
            msg = master.recv_match(type=["LOG_DATA"], timeout=timeout)
            if msg is None:
                return None
            if int(getattr(msg, "id", -1)) == log_id:
                return msg
    except Exception:
        return None
