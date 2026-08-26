# Parameter-file reference

Parameter files are parsed by the application-specific classes. The most
reliable reference for an option is the generated API documentation for that
class, linked from [Applications](applications). Tutorial inputs are kept
small and show canonical combinations of options.

All `.in` inputs under `gtests/`, `tests/`, `data/`, and `tutorials/` are
configured into the build tree by CMake. Use the generated file at runtime so
that configured data and output paths are resolved consistently.
