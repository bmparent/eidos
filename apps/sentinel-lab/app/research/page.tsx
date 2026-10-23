import { SentinelLab } from "@/components/sentinel-lab";
import { simulateSmoke } from "@/lib/sentinel/simulate";
import type { SmokeResult } from "@/lib/sentinel/types";
import Link from "next/link";

export default function Research() {
  const initialRun = simulateSmoke({ scenario: "S1_hidden_backdoor", seed: 0, frames: 240, system: "eidos_ms_v1_observer" }) as SmokeResult;
  return <><div className="gl-research-banner"><Link href="/">← Guided analysis</Link><span>Research console · initial observatory is a recorded synthetic example, separate from your uploaded runs</span></div><SentinelLab initialRun={initialRun} /></>;
}
