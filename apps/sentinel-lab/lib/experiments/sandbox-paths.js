export const SANDBOX_ROOT = "/vercel/sandbox";

/**
 * Vercel Sandbox checks a git source out beneath the Sandbox working directory
 * using the repository basename (for example, eidos.git becomes eidos).
 * Keep that provider-specific rule isolated and testable.
 */
export function sandboxRepositoryRoot(repositoryUrl, sandboxRoot = SANDBOX_ROOT) {
  const normalized = String(repositoryUrl || "")
    .trim()
    .replace(/[?#].*$/, "")
    .replace(/\/+$/, "");
  const separator = Math.max(normalized.lastIndexOf("/"), normalized.lastIndexOf(":"));
  const repositoryName = normalized.slice(separator + 1).replace(/\.git$/i, "");

  if (
    !repositoryName
    || repositoryName === "."
    || repositoryName === ".."
    || !/^[A-Za-z0-9._-]+$/.test(repositoryName)
  ) {
    throw new Error("EIDOS_SOURCE_REPOSITORY_INVALID");
  }

  return `${String(sandboxRoot).replace(/\/+$/, "")}/${repositoryName}`;
}
