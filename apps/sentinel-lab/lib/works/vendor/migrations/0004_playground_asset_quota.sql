-- Additive owner storage cap; existing historical assets are never deleted.
CREATE TRIGGER IF NOT EXISTS eidos_pg_asset_owner_cap
BEFORE INSERT ON eidos_pg_assets
WHEN NOT EXISTS(SELECT 1 FROM eidos_pg_assets WHERE owner_id=NEW.owner_id AND hash=NEW.hash)
AND (SELECT COALESCE(SUM(length(data)),0) FROM eidos_pg_assets WHERE owner_id=NEW.owner_id) + length(NEW.data) > 40000000
BEGIN
 SELECT RAISE(ABORT, 'Playground owner image quota exceeded; local content is preserved');
END;
