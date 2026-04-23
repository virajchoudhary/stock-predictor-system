TITLE:
Remove unused News and Pro Knowledge frontend tabs

DESCRIPTION:

### Problem

Two sections in the frontend were taking up space but were no longer needed or active:

1. The "Recent News" tab in the Market Trend Predictor (`1_Dashboard.py`)
2. The "Pro Knowledge" educational tab in the SABR Options Analysis page (`4_Options_Analysis.py`)

### Proposed Change

Remove these tabs completely to clean up the user interface.

### Implementation Tasks

- [x] Removed the "News" tab definition and its rendering logic (news expanders) from `frontend/pages/1_Dashboard.py`
- [x] Removed the "Pro Knowledge" tab definition and its heavy markdown content (educational content + Heston code block) from `frontend/pages/4_Options_Analysis.py`
- [x] Ensured all other tabs (Overview, Technicals, Financials, SABR, Volatility Surface) remain functional and properly spaced.

### Notes

- App state logic and functional components remain untouched.
- _Add `Closes #XYZ` (replace XYZ with your actual issue number) here to auto-close the issue on merge!_
