PR: #429

- Record changes in separate dated pull request fragments and archive them in versioned changelog files only during explicit release preparation. Validate fragment format and pull request metadata in CI.
- Optionally prepare a changelog-only release commit and pull request with `release --commit`, using a dedicated branch without creating tags.
