"""Fail-closed Episode certification tests."""

from __future__ import annotations

import unittest
from dataclasses import replace

from src.assembly.episode_assembler import EpisodeAssembler
from src.contracts.episode_certification import CertificationStatus
from src.contracts.trace_event import EventStatus, EventType
from src.validation.episode_semantics import certify_episode
from tests.execution_fixtures import make_complete_event_stream, make_episode_context


def _assemble(events, episode_id: str, task_id: str = "task-cert-1"):
    context = make_episode_context(task_id=task_id)
    return EpisodeAssembler().assemble(
        events, contexts={episode_id: context}
    ).episodes[0]


def _override(events, *, verifier: dict | None = None, terminal: dict | None = None):
    out = [
        replace(
            event,
            status=(verifier.get("status") if verifier and event.event_type is EventType.VERIFICATION_FINISHED else event.status),
            attributes=(
                {**event.attributes, **verifier.get("attributes", {})}
                if verifier and event.event_type is EventType.VERIFICATION_FINISHED
                else (
                    {**event.attributes, **terminal}
                    if terminal and event.event_type is EventType.EPISODE_FINISHED
                    else event.attributes
                )
            ),
        )
        for event in events
    ]
    return tuple(out)


class EpisodeCertificationTest(unittest.TestCase):
    def test_normal_certified_success_episode(self) -> None:
        ep = _assemble(make_complete_event_stream(episode_id="ep-cert-pos"), "ep-cert-pos")
        cert = certify_episode(ep)
        self.assertIs(cert.certification_status, CertificationStatus.CERTIFIED)
        self.assertTrue(cert.verifier_attested)
        self.assertEqual(cert.rejection_reasons, ())
        # certification is derived and never mutates the episode
        self.assertEqual(ep.checksum, certify_episode(ep).episode_checksum)
        self.assertEqual(cert.episode_checksum, ep.checksum)
        self.assertEqual(cert.episode_id, ep.episode_id)

    def test_verifier_failed_terminal_success_is_rejected(self) -> None:
        events = _override(
            make_complete_event_stream(episode_id="ep-fail-success"),
            verifier={"status": EventStatus.FAILED, "attributes": {"passed": False}},
            terminal={},  # terminal still SUCCESS / verifier PASSED
        )
        ep = _assemble(events, "ep-fail-success")
        cert = certify_episode(ep)
        self.assertIs(cert.certification_status, CertificationStatus.REJECTED)
        self.assertTrue(
            any("FAILED" in reason and "SUCCESS" in reason for reason in cert.rejection_reasons)
        )

    def test_verifier_passed_terminal_failure_is_rejected(self) -> None:
        events = _override(
            make_complete_event_stream(episode_id="ep-pass-fail"),
            terminal={
                "task_status": "FAILURE",
                "execution_validity": "VALID",
                "verifier_status": "FAILED",
                "score": 0.0,
            },
            # verifier event still PASSED
        )
        ep = _assemble(events, "ep-pass-fail")
        cert = certify_episode(ep)
        self.assertIs(cert.certification_status, CertificationStatus.REJECTED)
        self.assertTrue(
            any("PASSED" in reason and "not SUCCESS" in reason for reason in cert.rejection_reasons)
        )

    def test_terminal_success_without_verifier_event_is_insufficient(self) -> None:
        events = [
            e
            for e in make_complete_event_stream(episode_id="ep-no-verifier")
            if e.event_type
            not in (EventType.VERIFICATION_STARTED, EventType.VERIFICATION_FINISHED)
        ]
        events = _override(events, terminal={"evidence_event_ids": []})
        ep = _assemble(events, "ep-no-verifier")
        cert = certify_episode(ep)
        # A success with no verifier PASSED evidence must NOT be certified.
        self.assertIs(
            cert.certification_status, CertificationStatus.INSUFFICIENT_EVIDENCE
        )
        self.assertFalse(cert.verifier_attested)

    def test_partial_episode_not_certified(self) -> None:
        # drop the second half (tool result) to make progress partial-neutral is
        # hard; model response removed => missing MODEL_RESPONSE pair => PARTIAL
        events = list(make_complete_event_stream(episode_id="ep-partial"))
        events = [
            e for e in events if not (e.event_type is EventType.MODEL_RESPONSE)
        ]
        ep = _assemble(events, "ep-partial")
        self.assertNotEqual(ep.integrity.state.value, "COMPLETE")
        cert = certify_episode(ep)
        self.assertIsNot(cert.certification_status, CertificationStatus.CERTIFIED)

    def test_evidence_accepts_real_verifier_failure(self) -> None:
        # A valid FAILURE with real verifier FAILED evidence is a trustworthy
        # task failure (usable as a preference "rejected" half).
        events = _override(
            make_complete_event_stream(episode_id="ep-valid-fail"),
            verifier={"status": EventStatus.FAILED, "attributes": {"passed": False}},
            terminal={
                "task_status": "FAILURE",
                "execution_validity": "VALID",
                "verifier_status": "FAILED",
                "score": 0.0,
            },
        )
        ep = _assemble(events, "ep-valid-fail")
        cert = certify_episode(ep)
        self.assertIs(cert.certification_status, CertificationStatus.CERTIFIED)
        self.assertTrue(cert.verifier_attested)


if __name__ == "__main__":
    unittest.main()
