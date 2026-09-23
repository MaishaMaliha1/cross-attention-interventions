# Contributing

Contributions are welcome through pull requests. Please keep changes focused, add or update tests for behavior changes, and run:

```bash
ruff check src tests
pytest -q
```

Before opening a pull request, confirm that public APIs remain documented and that new model-specific behavior is isolated behind configuration rather than hard-coded into the analysis logic.
