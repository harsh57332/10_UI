"""Inspire RH56DFTP dexterous hands (FTP / tactile version) on the G1: Modbus TCP, one connection per hand.

Unitree's G1 wiring (inspire_hand_ws, Headless_driver_double.py): left 192.168.123.210, right 192.168.123.211,
port 6000, unit 1. Registers (int16, 6 values in finger order little, ring, middle, index, thumb bend, thumb
rotation): ANGLE_SET 1486 (0 = closed .. 1000 = open), FORCE_SET 1498, SPEED_SET 1522, ANGLE_ACT 1546,
clear errors 1004. Plain sockets (no pymodbus on the robot). The hand's firmware echoes a wrong transaction id
(off by one), so replies are matched by function code, not by id.

Only open and close, from the UI (section 5). The skills and the LLM treat the hands as dummy hands.
"""
import logging
import socket
import struct
import threading
import time

log = logging.getLogger("inspire")
ANGLE_SET, FORCE_SET, SPEED_SET, ANGLE_ACT, CLEAR_ERROR = 1486, 1498, 1522, 1546, 1004


class HandError(RuntimeError):
    pass


class Hand:
    def __init__(self, side, ip, port, unit, timeout_s):
        self.side, self.ip, self.port, self.unit, self.timeout = side, ip, port, unit, timeout_s
        self.sock, self.tid, self.lock = None, 0, threading.Lock()
        self.angles, self.error, self.t = None, None, 0.0      # last read (polling thread)

    def _connect(self):
        s = socket.create_connection((self.ip, self.port), timeout=self.timeout)
        s.settimeout(self.timeout)
        self.sock = s

    def _close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None

    def _recv(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise HandError("connection closed by the hand")
            buf += chunk
        return buf

    def _request(self, fc, payload):
        """One Modbus TCP request/response (reconnects once on a broken connection)."""
        with self.lock:
            for attempt in (0, 1):
                try:
                    if self.sock is None:
                        self._connect()
                    self.tid = (self.tid + 1) & 0xFFFF
                    pdu = bytes([fc]) + payload
                    self.sock.sendall(struct.pack(">HHHB", self.tid, 0, len(pdu) + 1, self.unit) + pdu)
                    _, _, length, _ = struct.unpack(">HHHB", self._recv(7))
                    body = self._recv(length - 1)
                    if body[0] == fc | 0x80:
                        raise HandError(f"{self.side} hand: Modbus exception {body[1]}")
                    if body[0] != fc:
                        raise HandError(f"{self.side} hand: unexpected reply function {body[0]}")
                    return body[1:]
                except (OSError, struct.error) as e:
                    self._close()
                    if attempt:
                        raise HandError(f"{self.side} hand ({self.ip}:{self.port}) not reachable: {e}")
                    time.sleep(0.2)             # the hand needs a short pause between reconnects

    def write(self, addr, values):
        values = [int(v) for v in values]
        self._request(0x10, struct.pack(">HHB", addr, len(values), 2 * len(values)) + struct.pack(f">{len(values)}h", *values))

    def read(self, addr, n):
        data = self._request(0x03, struct.pack(">HH", addr, n))
        return list(struct.unpack(f">{n}h", data[1:1 + 2 * n]))


def probe(cfg):
    """Both Inspire hands answer on their Modbus port (detection at start-up)."""
    for side in ("left", "right"):
        try:
            socket.create_connection((cfg["hands"][side]["ip"], cfg["port"]), timeout=cfg["timeout_s"]).close()
        except OSError:
            return False
    return True


class Hands:
    """Both hands: a polling thread keeps their finger angles; open/close run as short sequences."""

    def __init__(self, cfg):
        self.cfg = cfg
        h = cfg["hands"]
        self.hands = {s: Hand(s, h[s]["ip"], cfg["port"], cfg["unit"], cfg["timeout_s"]) for s in ("left", "right")}
        self.busy = {s: None for s in self.hands}
        threading.Thread(target=self._poll, daemon=True, name="inspire_poll").start()

    def _poll(self):
        while True:
            for hand in self.hands.values():
                try:
                    hand.angles, hand.error, hand.t = hand.read(ANGLE_ACT, 6), None, time.time()
                except HandError as e:
                    hand.error = str(e)
            absent = any(h.error for h in self.hands.values())
            time.sleep(self.cfg["detect_s"] if absent else 1.0 / self.cfg["poll_hz"])   # absent: just look now and then

    def state(self):
        out = {}
        for s, hand in self.hands.items():
            a = hand.angles if hand.error is None else None
            pose = None
            if a is not None:
                lo, hi = min(a), max(a)
                tol = self.cfg["at_tolerance"]
                pose = "open" if lo >= 1000 - tol else ("closed" if hi <= self.cfg["closed_max"] else "between")
            out[s] = {"ip": hand.ip, "connected": hand.error is None and hand.t > 0, "angles": a, "pose": pose,
                      "moving": self.busy[s], "error": hand.error}
        return out

    def command(self, side, pose):
        """open: every finger fully open. close: the four fingers first, then the thumb over them."""
        if side not in self.hands:
            raise ValueError("side must be left or right")
        if pose not in ("open", "close"):
            raise ValueError("state must be open or close")
        hand = self.hands[side]
        if hand.error is not None or hand.t == 0:
            raise HandError(hand.error or f"{side} hand not connected yet")
        c = self.cfg
        hand.write(CLEAR_ERROR, [1])
        hand.write(SPEED_SET, [c["speed"]] * 6)
        hand.write(FORCE_SET, [c["force"]] * 6)

        def run():
            self.busy[side] = pose
            try:
                if pose == "open":
                    hand.write(ANGLE_SET, [1000] * 6)
                else:
                    hand.write(ANGLE_SET, [0, 0, 0, 0, 1000, 1000])     # fingers, the thumb still out of the way
                    time.sleep(c["thumb_delay_s"])
                    hand.write(ANGLE_SET, [0] * 6)                     # then the thumb over them
                log.info("%s hand: %s", side, pose)
            except HandError as e:
                log.warning("%s hand %s failed: %s", side, pose, e)
            finally:
                self.busy[side] = None
        threading.Thread(target=run, daemon=True, name=f"inspire_{side}").start()
        return {"side": side, "state": pose}
