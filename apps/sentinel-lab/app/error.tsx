"use client";

export default function GlobalError({ reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return (
    <main className="fatal-state">
      <p className="eyebrow">OPERATOR SURFACE HALTED</p>
      <h1>The lab could not render this state.</h1>
      <p>No proof gate or artifact was changed.</p>
      <button className="outline-button" onClick={reset}>Retry render</button>
    </main>
  );
}
