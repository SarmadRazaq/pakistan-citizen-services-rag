# Requirements Traceability Map

| Requirement | User Story | Test | Status |
|---|---|---|---|
| GO-11 emergency Notice (ISSUE-002, #3) | Emergency question gets emergency Notices first on every outcome | tests/unit/test_guard_emergency.py, tests/unit/test_ask_pipeline.py (emergency tests) | done |
| Closed Topic list for public Documents (ISSUE-003, #4) | Public upload/patch normalise a known topic and reject an unknown one with 422; private stays free text; frontend dropdown | tests/unit/test_topic_rules.py, tests/integration/test_topic_enforcement.py, tests/api/test_topic_enforcement_api.py, frontend/src/lib/topics.test.ts | done |
