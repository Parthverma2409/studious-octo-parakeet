"""Encrypted-at-rest storage for OAuth tokens (NFR3).

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
A refresh token is a long-lived bearer credential for someone's calendar and
mail. Writing it to ``~/.booking-sync/token.json`` in the clear -- which is what
most tutorials do -- means any process, backup, sync client or screen-share that
touches that file owns the account. So the file is encrypted.

``cryptography`` is not available in this project, so the construction below is
built from ``hashlib`` and ``hmac``:

    key material : PBKDF2-HMAC-SHA256(passphrase, 16-byte random salt,
                   600 000 iterations, 64 bytes) split into a 32-byte
                   encryption key and a separate 32-byte MAC key.
    cipher       : SHA-256 keystream in counter mode -- block i of the keystream
                   is HMAC-SHA256(enc_key, nonce || big-endian uint32 i) -- XORed
                   with the plaintext.
    integrity    : encrypt-then-MAC. tag = HMAC-SHA256(mac_key, version || salt ||
                   nonce || ciphertext), verified with hmac.compare_digest before
                   a single byte is decrypted.
    freshness    : a new random salt AND a new random nonce on every save.

WHAT IT GUARANTEES
  * Confidentiality against someone who obtains the file but not the passphrase,
    to the strength of the passphrase and 600k PBKDF2 iterations.
  * Detection of any modification to the file, including bit-flips in the
    ciphertext, a swapped salt or nonce, and truncation. Tampering fails closed:
    ``TokenStoreError``, never a partial or corrupted credential.
  * Keystream uniqueness across saves, provided ``os.urandom`` is sound.

WHAT IT DOES NOT GUARANTEE
  * It is NOT a vetted, audited, or peer-reviewed cryptographic implementation.
    A production deployment should use a vetted library (``cryptography``'s
    Fernet or AES-GCM) or, better, an OS keystore -- DPAPI on Windows, Keychain
    on macOS, libsecret on Linux -- which also protects the key at rest instead
    of deriving it from something a human typed.
  * No protection once the process is compromised: the passphrase, the derived
    keys and the decrypted token are all plaintext in memory.
  * No constant-time guarantee for the XOR loop, and no memory zeroisation --
    Python strings and bytes are immutable, so the plaintext cannot be wiped.
  * No key rotation, no versioned re-encryption, no multi-user access control.
  * A weak passphrase is still a weak passphrase; PBKDF2 raises the cost of
    guessing, it does not remove it.
  * Decryption cannot distinguish "wrong passphrase" from "file tampered with",
    because both are just a failed MAC. The error says so rather than guessing.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path

#: Bumped if the construction ever changes, so an old file fails loudly instead
#: of being decrypted with the wrong rules.
FORMAT_VERSION = "bsync-token-v1"

PBKDF2_ITERATIONS = 600_000
SALT_BYTES = 16
NONCE_BYTES = 16
KEY_BYTES = 32


class TokenStoreError(RuntimeError):
    """The token file could not be read, decrypted, or authenticated."""


def _derive(passphrase: str, salt: bytes) -> tuple[bytes, bytes]:
    """One PBKDF2 call, split into two independent keys.

    Separate encryption and MAC keys, rather than one key used for both, is the
    cheap way to avoid the class of attacks that come from reusing key material
    across two primitives.
    """
    material = hashlib.pbkdf2_hmac(
        "sha256", passphrase.encode("utf-8"), salt, PBKDF2_ITERATIONS, dklen=KEY_BYTES * 2
    )
    return material[:KEY_BYTES], material[KEY_BYTES:]


def _keystream(enc_key: bytes, nonce: bytes, length: int) -> bytes:
    blocks = bytearray()
    counter = 0
    while len(blocks) < length:
        blocks += hmac.new(enc_key, nonce + struct.pack(">I", counter), hashlib.sha256).digest()
        counter += 1
    return bytes(blocks[:length])


def _xor(data: bytes, stream: bytes) -> bytes:
    return bytes(a ^ b for a, b in zip(data, stream))


def _tag(mac_key: bytes, salt: bytes, nonce: bytes, ciphertext: bytes) -> bytes:
    """MAC over the whole envelope, not just the ciphertext.

    Covering the salt and nonce is what stops an attacker swapping in a salt from
    an older file and getting a different plaintext to decrypt cleanly.
    """
    mac = hmac.new(mac_key, FORMAT_VERSION.encode("ascii"), hashlib.sha256)
    mac.update(salt)
    mac.update(nonce)
    mac.update(ciphertext)
    return mac.digest()


def encrypt(plaintext: bytes, passphrase: str) -> dict:
    """Return the JSON-serialisable envelope for ``plaintext``."""
    salt = os.urandom(SALT_BYTES)
    nonce = os.urandom(NONCE_BYTES)
    enc_key, mac_key = _derive(passphrase, salt)
    ciphertext = _xor(plaintext, _keystream(enc_key, nonce, len(plaintext)))
    return {
        "version": FORMAT_VERSION,
        "kdf": {"name": "pbkdf2-hmac-sha256", "iterations": PBKDF2_ITERATIONS},
        "salt": base64.b64encode(salt).decode("ascii"),
        "nonce": base64.b64encode(nonce).decode("ascii"),
        "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
        "tag": base64.b64encode(_tag(mac_key, salt, nonce, ciphertext)).decode("ascii"),
    }


def decrypt(envelope: dict, passphrase: str) -> bytes:
    """Verify then decrypt. Raises TokenStoreError on any failure."""
    if envelope.get("version") != FORMAT_VERSION:
        raise TokenStoreError(
            "unsupported token file version %r (expected %r)"
            % (envelope.get("version"), FORMAT_VERSION)
        )
    iterations = (envelope.get("kdf") or {}).get("iterations")
    if iterations != PBKDF2_ITERATIONS:
        raise TokenStoreError("token file declares %r KDF iterations; refusing to guess" % iterations)
    try:
        salt = base64.b64decode(envelope["salt"], validate=True)
        nonce = base64.b64decode(envelope["nonce"], validate=True)
        ciphertext = base64.b64decode(envelope["ciphertext"], validate=True)
        tag = base64.b64decode(envelope["tag"], validate=True)
    except (KeyError, ValueError, TypeError) as exc:
        raise TokenStoreError("token file is malformed: %s" % exc) from exc

    enc_key, mac_key = _derive(passphrase, salt)
    if not hmac.compare_digest(_tag(mac_key, salt, nonce, ciphertext), tag):
        # Deliberately one message for both causes: a failed MAC cannot tell them
        # apart, and pretending otherwise would be a lie in a security path.
        raise TokenStoreError(
            "token file failed authentication: wrong passphrase, or the file has been modified"
        )
    return _xor(ciphertext, _keystream(enc_key, nonce, len(ciphertext)))


@dataclass
class EncryptedTokenStore:
    """A single credential document, encrypted at rest under a passphrase.

    Deliberately stores an opaque ``dict`` rather than an ``OAuthCredentials``:
    keeping this module ignorant of OAuth is what stops ``oauth.py`` and
    ``tokens.py`` importing each other in a circle, and lets the store hold a
    future credential shape without changing.
    """

    path: Path
    passphrase: str = field(repr=False)

    def exists(self) -> bool:
        return Path(self.path).exists()

    def save(self, payload: dict) -> None:
        envelope = encrypt(json.dumps(payload, sort_keys=True).encode("utf-8"), self.passphrase)
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-replace: a crash mid-write must not leave a half-file where
        # a working credential used to be.
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
        os.replace(temp, path)
        _restrict(path)

    def load(self) -> dict:
        path = Path(self.path)
        if not path.exists():
            raise TokenStoreError("no token file at %s; run the authorisation flow first" % path)
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise TokenStoreError("token file at %s is not valid JSON: %s" % (path, exc)) from exc
        payload = json.loads(decrypt(envelope, self.passphrase).decode("utf-8"))
        if not isinstance(payload, dict):
            raise TokenStoreError("token file did not contain a credential object")
        return payload


def _restrict(path: Path) -> None:
    """Best-effort owner-only permissions.

    POSIX mode bits are close to meaningless on Windows, which is where this
    project runs, so this is defence in depth and not a control we rely on. The
    encryption is the control.
    """
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
