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

let database: {key:string;value:Database} | undefined;
export function databaseConfiguration(env: Record<string, string | undefined>) {
  const prefix = env.EIDOS_DATABASE_BINDING_PREFIX;
  if (prefix && !/^EIDOS_[A-Z0-9_]{1,48}$/.test(prefix)) throw Error('Invalid Works database binding prefix');
  // A dedicated preview can select separately provisioned sensitive bindings.
  // Missing prefixed credentials never fall back to the shared/production database.
  return { url: prefix ? env[prefix + '_TURSO_DATABASE_URL'] : env.EIDOS_DATABASE_URL,
    authToken: prefix ? env[prefix + '_TURSO_AUTH_TOKEN'] : env.EIDOS_DATABASE_AUTH_TOKEN };
}
export function platformDatabase(env: Record<string, string | undefined> = process.env) {
  const selected = databaseConfiguration(env);
  if (!selected.url || !selected.authToken) return undefined;
  const url = new URL(selected.url);
  // Hosted execution must never use ephemeral /tmp or an embedded file for quotas/entitlements.
  if (!['libsql:', 'https:'].includes(url.protocol) || url.username || url.password) return undefined;
  const key=JSON.stringify([selected.url,selected.authToken]);
  if(database?.key===key)return database.value;
  const value=adaptDatabase(createClient({
    url: selected.url, authToken: selected.authToken,
  }));
  database={key,value};return value;
}
