import type { SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement>;

export function EidosMark(props: IconProps) {
  return (
    <svg viewBox="0 0 32 32" aria-hidden="true" {...props}>
      <circle cx="16" cy="16" r="13.5" fill="none" stroke="currentColor" strokeWidth="1" />
      <path d="M8 16h16M16 8v16" fill="none" stroke="currentColor" strokeWidth="1" />
      <circle cx="16" cy="16" r="3.5" fill="currentColor" />
      <path d="m10.3 10.3 11.4 11.4m0-11.4L10.3 21.7" fill="none" stroke="currentColor" strokeWidth=".75" />
    </svg>
  );
}

export function PlayIcon(props: IconProps) {
  return <svg viewBox="0 0 20 20" aria-hidden="true" {...props}><path d="m7 4 8 6-8 6V4Z" fill="currentColor" /></svg>;
}

export function MenuIcon({ open = false, ...props }: IconProps & { open?: boolean }) {
  return (
    <svg viewBox="0 0 20 20" aria-hidden="true" {...props}>
      {open ? <path d="m5 5 10 10m0-10L5 15" fill="none" stroke="currentColor" strokeWidth="1.5" /> : <path d="M3 5.5h14M3 10h14M3 14.5h14" fill="none" stroke="currentColor" strokeWidth="1.5" />}
    </svg>
  );
}

export function ArrowIcon(props: IconProps) {
  return <svg viewBox="0 0 20 20" aria-hidden="true" {...props}><path d="M4 10h11m-4-4 4 4-4 4" fill="none" stroke="currentColor" strokeWidth="1.4" /></svg>;
}

export function UploadIcon(props: IconProps) {
  return <svg viewBox="0 0 20 20" aria-hidden="true" {...props}><path d="M10 13V3m0 0L6.5 6.5M10 3l3.5 3.5M4 12.5v3.5h12v-3.5" fill="none" stroke="currentColor" strokeWidth="1.35" /></svg>;
}

export function DownloadIcon(props: IconProps) {
  return <svg viewBox="0 0 20 20" aria-hidden="true" {...props}><path d="M10 3v10m0 0 3.5-3.5M10 13 6.5 9.5M4 14.5V17h12v-2.5" fill="none" stroke="currentColor" strokeWidth="1.35" /></svg>;
}

export function CloseIcon(props: IconProps) {
  return <svg viewBox="0 0 20 20" aria-hidden="true" {...props}><path d="m5 5 10 10m0-10L5 15" fill="none" stroke="currentColor" strokeWidth="1.4" /></svg>;
}
