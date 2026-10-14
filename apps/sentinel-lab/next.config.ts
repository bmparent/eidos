import type { NextConfig } from "next";
import { fileURLToPath } from "node:url";

const workspaceRoot = fileURLToPath(new URL("../../", import.meta.url));

const nextConfig: NextConfig = {
  poweredByHeader: false,
  reactStrictMode: true,
  // The Observatory collector imports the same reviewed public-target boundary
  // as its Worker, rather than maintaining a second SSRF policy implementation.
  turbopack: { root: workspaceRoot },
  outputFileTracingRoot: workspaceRoot,
  async headers() {
    return [{ source: "/:path*", headers: [
      { key: "X-Content-Type-Options", value: "nosniff" },
      { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
    ] }];
  },
};

export default nextConfig;
