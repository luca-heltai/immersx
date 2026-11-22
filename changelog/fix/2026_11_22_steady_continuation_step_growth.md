PR: #490

- After a failed continuation stage is recovered by halving the eta increment, keep the smaller increment for one more accepted stage before increasing it again. This avoids immediately repeating an overshoot.
