# Current configuration boundary

`project.yaml` records the current design assumptions. Hardware values are
user-declared constraints, not evidence that this checkout probed a GPU.

`local-execution.example.json` is the secret-free input contract for
`prepare-local` and `execute-local`. Copy it per run, replace all placeholders,
and keep credentials in the named host environment variable rather than JSON.

There is intentionally no current `upstream-lock.yaml`. A new lock may be
created only after the selected producer/trainer, model server, model, driver,
container image and runtime smoke evidence have been captured on the rented GPU host. The old
two-Blackwell lock is preserved under `archive/configs/two-blackwell/`.
