# Contributing

Contributions (pull requests) are very welcome! Here's how to get started.

---

First fork the library on GitHub.

Then clone and install the library in development mode:

```bash
git clone https://github.com/DelftMercurians/cotix
cd cotix
git checkout -b your-branch-name
pip install -e .
```

Then install the pre-commit hook:

```bash
pip install pre-commit
pre-commit install
```

These hooks use Black to format the code, and Ruff to lint it.

Next verify the tests all pass:

```bash
pip install pytest
pytest
```

Then push your changes back to your fork of the repository:

```bash
git push
```

Finally, open a pull request!
