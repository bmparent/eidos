import { createClient, type Client, type InArgs, type InStatement } from '@libsql/client';
import type { Database, Statement } from './vendor/functions/_shared/platform/core';

class LibsqlStatement implements Statement {
  constructor(readonly client: Client, readonly sql: string, readonly args: InArgs = []) {}
  bind(...args: unknown[]) { return new LibsqlStatement(this.client, this.sql, args as InArgs); }
  async first<T>() {
    const result = await this.client.execute({ sql: this.sql, args: this.args });
    return result.rows[0] ? { ...result.rows[0] } as T : null;
  }
  async all<T>() {
    const result = await this.client.execute({ sql: this.sql, args: this.args });
    return { results: result.rows.map(row => ({ ...row })) as T[] };
  }
  async run() {
    const result = await this.client.execute({ sql: this.sql, args: this.args });
    return { meta: { changes: result.rowsAffected } };
  }
}

export function adaptDatabase(client: Client): Database {
  return {
    prepare: sql => new LibsqlStatement(client, sql),
    async batch(statements) {
      const queries: InStatement[] = statements.map(statement => {
        if (!(statement instanceof LibsqlStatement) || statement.client !== client)
          throw Error('Mismatched database statement');
        return { sql: statement.sql, args: statement.args };
      });
      // libSQL batches are one write transaction: payment and webhook records commit together.
      return client.batch(queries, 'write');
    },
  };
}

let database: Database | undefined;
export function platformDatabase(env: Record<string, string | undefined> = process.env) {
  if (!env.EIDOS_DATABASE_URL || !env.EIDOS_DATABASE_AUTH_TOKEN) return undefined;
  const url = new URL(env.EIDOS_DATABASE_URL);
  // Hosted execution must never use ephemeral /tmp or an embedded file for quotas/entitlements.
  if (!['libsql:', 'https:'].includes(url.protocol) || url.username || url.password) return undefined;
  return database ??= adaptDatabase(createClient({
    url: env.EIDOS_DATABASE_URL, authToken: env.EIDOS_DATABASE_AUTH_TOKEN,
  }));
}
