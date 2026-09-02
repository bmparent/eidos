import { SentinelLab } from "@/components/sentinel-lab";
import { simulateSmoke } from "@/lib/sentinel/simulate";
import type { SmokeResult } from "@/lib/sentinel/types";

export default function Home() {
  const initialRun = simulateSmoke({
    scenario: "S1_hidden_backdoor",
    seed: 0,
    frames: 240,
    system: "eidos_ms_v1_observer",
  }) as SmokeResult;

  return <SentinelLab initialRun={initialRun} />;
}
