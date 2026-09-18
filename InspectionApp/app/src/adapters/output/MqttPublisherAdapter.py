import json
import threading

import paho.mqtt.client as mqtt

from app.src.interfaces.IRepository import IRepository
from app.src.core.models.Part import Part


class MqttPublisherAdapter(IRepository):
    """
    Publishes inspection results to an MQTT broker.

    Implements IRepository so InspectionService/SequenceExecutor never need to
    know MQTT exists — this adapter is meant to be combined with
    LocalStorageAdapter through CompositeRepository, never used alone (image
    saving is a no-op here; the JSONL/disk adapter remains the source of truth
    for traceability review and recalibration).

    Connection is deferred by _CONNECT_DELAY_S seconds after construction (see
    __init__) so it never competes with camera hardware initialization for
    CPU/GIL time during AppFactory._build_hardware(). Publishing itself is
    fire-and-forget and non-blocking: once connected, the client runs its
    network loop in a background thread, so a broker outage never stalls or
    fails an inspection cycle. Failed publishes are logged and return False;
    they do not raise.

    Attributes:
        _client (mqtt.Client): Paho MQTT client instance.
        _topic (str): Fixed topic all results are published to.
        _qos (int): MQTT QoS level used for publishes (0, 1, or 2).
        _closed (bool): True once close() has been called. Guards against the
            deferred connect timer firing after shutdown (see _start_connection).
        _connect_timer (threading.Timer): Pending deferred-connect timer,
            cancelled by close() so a fast reload_sequence() (shutdown() right
            after construction, before the delay elapses) can never leave a
            zombie timer that reconnects — with the same client_id as the
            adapter's replacement — after this instance is supposed to be dead.
    """

    _CONNECT_DELAY_S: float = 5.0  # Delay before the first connection attempt.

    def __init__(
        self,
        broker_host: str,
        topic: str,
        broker_port: int = 1883,
        client_id: str = "inspectionapp",
        qos: int = 1,
        username: str | None = None,
        password: str | None = None,
        keepalive: int = 60,
    ):
        """
        Args:
            broker_host (str): MQTT broker hostname or IP.
            topic (str): Topic to publish every inspection result to.
            broker_port (int): Broker port. Defaults to 1883 (plain MQTT).
            client_id (str): MQTT client identifier. Should be unique per Pi/line
                to avoid the broker kicking out a duplicate client_id.
            qos (int): QoS level for publishes. 1 (default) gives at-least-once
                delivery without requiring persistent sessions.
            username (str | None): Broker auth username, if required.
            password (str | None): Broker auth password, if required.
            keepalive (int): MQTT keepalive interval in seconds.
        """
        self._topic = topic
        self._qos = qos
        self._closed = False

        self._client = mqtt.Client(client_id=client_id, clean_session=True)
        if username:
            self._client.username_pw_set(username, password)

        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect

        # Conservative backoff: if the broker ever rejects this client_id again
        # (e.g. a stale session on the broker side), this caps how fast paho
        # hammers it with reconnect attempts — a tight loop here previously
        # spawned enough threads/sockets per second to OOM-kill the gunicorn
        # worker. This does not fix a client_id collision, only limits the
        # blast radius if one happens again.
        self._client.reconnect_delay_set(min_delay=5, max_delay=60)

        self._broker_host = broker_host
        self._broker_port = broker_port
        self._keepalive = keepalive

        # Deferred start: connecting immediately here would spawn paho's network
        # thread in the same critical window where Picamera2/libcamera is still
        # finishing sensor initialization, and thread/GIL contention on the Pi's
        # limited cores can turn a marginal capture timeout into a hard failure.
        # A short delay lets camera init settle first.
        self._connect_timer = threading.Timer(self._CONNECT_DELAY_S, self._start_connection)
        self._connect_timer.daemon = True
        self._connect_timer.start()

    def _start_connection(self) -> None:
        """
        Connect to the broker and start paho's background network thread.

        Called after a startup delay (see __init__) so this never competes
        with camera hardware initialization for CPU/GIL time. No-ops if
        close() was already called before this fired — otherwise a fast
        reload_sequence() (construct → shutdown() within the delay window)
        would let this timer resurrect a connection using this adapter's
        client_id after a replacement adapter has already claimed it on the
        broker, causing the two to fight over the same client_id forever.

        Returns:
            None
        """
        if self._closed:
            return
        try:
            self._client.connect_async(self._broker_host, self._broker_port, self._keepalive)
            self._client.loop_start()
        except Exception as e:
            print(f"[ERROR] MqttPublisherAdapter: deferred connect failed: {e}")

    # =========================================================================
    # IRepository interface
    # =========================================================================

    def save_inspection_result(self, part: Part) -> bool:
        """
        Publish the inspection result for a completed part to MQTT.

        Args:
            part (Part): Completed Part with inspection_results, triggers,
                date_inspected, and time_inspected populated.

        Returns:
            bool: True if the publish was handed off successfully, False on
                any error (broker unreachable, serialization failure, etc.).
        """
        try:
            payload = self._build_payload(part)
            result = self._client.publish(
                self._topic, json.dumps(payload, ensure_ascii=False), qos=self._qos
            )
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                print(f"[WARN] MqttPublisherAdapter: publish rc={result.rc} for part '{part.part_id}'.")
                return False
            return True
        except Exception as e:
            print(f"[ERROR] MqttPublisherAdapter: failed to publish part '{part.part_id}': {e}")
            return False

    def get_inspection_result(self, part_id: str) -> Part | None:
        """
        Not supported — MQTT is a publish destination, not a queryable store.

        Args:
            part_id (str): Unused.

        Returns:
            None
        """
        print("[WARN] MqttPublisherAdapter.get_inspection_result: not supported by this adapter.")
        return None

    def save_frames(self, part_id: str, captured_frames: dict) -> bool:
        """
        No-op. Images are never published over MQTT (payload size/broker load);
        LocalStorageAdapter remains responsible for image persistence.

        Args:
            part_id (str): Unused.
            captured_frames (dict): Unused.

        Returns:
            bool: Always True (nothing to do, not a failure).
        """
        return True

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def close(self) -> None:
        """
        Cancel any pending deferred connection and disconnect from the broker.

        Marks the adapter closed BEFORE touching the client/timer so that even
        if _start_connection() is already mid-flight when this runs, it will
        see _closed=True on its next check. Safe to call even if never
        connected, and safe to call more than once.

        Should be called once when the application shuts down (see
        AppFactory.shutdown()) or when a sequence reload replaces this adapter.

        Returns:
            None
        """
        self._closed = True
        if self._connect_timer is not None:
            self._connect_timer.cancel()
        try:
            self._client.loop_stop()
            self._client.disconnect()
        except Exception as e:
            print(f"[WARN] MqttPublisherAdapter: error during close(): {e}")

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _on_connect(self, client, userdata, flags, rc) -> None:
        if rc == 0:
            print(f"[OK] MqttPublisherAdapter: connected to broker (topic='{self._topic}').")
        else:
            print(f"[WARN] MqttPublisherAdapter: connect failed, rc={rc}.")

    def _on_disconnect(self, client, userdata, rc) -> None:
        if rc != 0 and not self._closed:
            print(f"[WARN] MqttPublisherAdapter: unexpected disconnect (rc={rc}); paho will retry.")

    @staticmethod
    def _build_payload(part: Part) -> dict:
        """
        Serialize a Part into the same normalized structure as the JSONL
        traceability record (part / view_results / trigger_events), so
        downstream MQTT consumers see identical fields to the local log.

        Args:
            part (Part): Completed Part instance.

        Returns:
            dict: JSON-serializable payload.
        """
        date_str = part.date_inspected.strftime("%Y%m%d_%H%M%S")
        return {
            "part": {
                "part_id": part.part_id,
                "model_id": part.model_id,
                "date_inspected": date_str,
                "duration_s": round(part.time_inspected, 4) if part.time_inspected is not None else None,
                "overall_status": "ERROR_ABORTED" if getattr(part, "system_error_paused", False) else (
                    ("OK" if part.overall_status else "NOK") + ("_SCRAP" if part.forced_scrap else "") + ("_DR" if part.dry_run else "")
                ),
                "piece_detected": part.piece_detected,
            },
            "view_results": [
                {
                    "view_name": result.view,
                    "classification": "OK" if result.is_ok else "NOK",
                    "score": round(result.score, 6),
                    "threshold_min": round(result.threshold_used[0], 6),
                    "threshold_max": round(result.threshold_used[1], 6),
                }
                for result in part.inspection_results
            ],
            "trigger_events": [
                {
                    "step_number": trig.step_number,
                    "direction": trig.direction,
                    "pin": trig.pin,
                    "action": trig.action,
                    "result": trig.result,
                }
                for trig in part.triggers
            ],
        }