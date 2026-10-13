import { randomBytes, scrypt, timingSafeEqual } from 'node:crypto';

// OWASP scrypt profile: N=2^17, r=8, p=1, 128 MiB working memory.
// Asynchronous built-in Node KDF; no blocking JS hashing and no plaintext persistence.
const params = { N: 131072, r: 8, p: 1, maxmem: 256 * 1024 * 1024 };
function derive(password: string, salt: Buffer): Promise<Buffer> {
  return new Promise((resolve, reject) => scrypt(password, salt, 64, params, (error, key) => error ? reject(error) : resolve(key)));
}
const dummy = 'scrypt$131072$8$1$' + '00'.repeat(32) + '$' + '00'.repeat(64);
export const passwordService = {
  async hash(password: string) {
    const salt = randomBytes(32), key = await derive(password, salt);
    return `scrypt$131072$8$1$${salt.toString('hex')}$${key.toString('hex')}`;
  },
  async verify(password: string, encoded: string | null) {
    const match = (encoded || dummy).match(/^scrypt\$131072\$8\$1\$([a-f0-9]{64})\$([a-f0-9]{128})$/);
    const parts = match || dummy.match(/^scrypt\$131072\$8\$1\$([a-f0-9]{64})\$([a-f0-9]{128})$/)!;
    const key = await derive(password, Buffer.from(parts[1], 'hex'));
    return timingSafeEqual(key, Buffer.from(parts[2], 'hex')) && Boolean(encoded && match);
  },
};
