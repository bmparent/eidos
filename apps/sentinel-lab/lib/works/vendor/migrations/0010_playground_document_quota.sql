-- Additive admission only. Existing projects, images and revisions are retained.
CREATE TRIGGER IF NOT EXISTS eidos_pg_revision_storage_cap
BEFORE INSERT ON eidos_pg_revisions
WHEN (SELECT COUNT(*) FROM eidos_pg_revisions WHERE project_id=NEW.project_id) >= 100
OR (SELECT COALESCE(SUM(length(CAST(r.document AS BLOB))),0)
    FROM eidos_pg_revisions r JOIN eidos_pg_projects p ON p.id=r.project_id
    WHERE p.owner_id=(SELECT owner_id FROM eidos_pg_projects WHERE id=NEW.project_id))
    + length(CAST(NEW.document AS BLOB)) > 20000000
BEGIN
 SELECT RAISE(ABORT, 'Playground owner document quota exceeded; local content is preserved');
END;
