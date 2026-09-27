GitHub-hosted maintenance requests to the website returned HTTP 403 even after the scoped token was refreshed. The hourly job now calls the Sentinel maintenance endpoint directly, with both independent platform and maintenance credentials; missing credentials fail visibly.

Validation: deployed authenticated endpoint returned {"ok":true,"suggestions":0} from GitHub Actions run 33997147456. Hourly cron remains 17 minutes past the hour. No site, database, research, or payment behavior changed.
