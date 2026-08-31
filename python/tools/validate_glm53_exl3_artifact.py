#!/usr/bin/env python3
"""Validate a completed GLM-5.3 EXL3 K4 artifact before publication.

The shared validator retains its historical GLM-5.2 filename because release
automation imports it, but its contract is selected from the bound plan and it
validates both supported generations.
"""

from validate_glm52_exl3_artifact import main


if __name__ == "__main__":
    main()
