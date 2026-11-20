PR: #482

- Register `ParameterAcceptor` constructors and inheritance separately, using Coral's current API and preserving its derived-type metadata.
- Suppress GCC 12's optimized-build `-Warray-bounds` false positive for Coral plugin targets only.
- Use one Coral registry identity for the parsed-function and generic scalar known-term APIs, and validate inheritance against the current `bases` registry field.

- Apply the GCC 12 workaround to Release Coral plugins in both Release and DebugRelease builds.
