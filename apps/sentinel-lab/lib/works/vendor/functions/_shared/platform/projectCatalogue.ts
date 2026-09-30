/** Public claims reviewed against repository evidence; never infer live acceptance from source. */
export const catalogueRevision = '2026-09-21.1';
export interface VerifiedProject {
  id: string;
  title: string;
  kind: 'client implementation' | 'internal tool' | 'fictional demonstration' | 'product';
  availability: 'live' | 'preview' | 'coming soon';
  aliases: string[];
  businessTasks: string[];
  supportedClaims: string[];
  limitations: string[];
  sourceUrls: string[];
  demoUrl?: string;
  assetRefs: string[];
  lastVerifiedAt: string;
}
export const projectCatalogue: VerifiedProject[] = [
  {
    id: 'cinematic-starter', title: 'Cinematic Starter Kit', kind: 'product', availability: 'live',
    aliases: ['starter kit', 'cinematic starter', 'cinematic kit'], businessTasks: ['website starter', 'download kit', 'recover purchase'],
    supportedClaims: ['The Cinematic Starter Kit is a downloadable website starting point. The shop explains included files, license and support scope before checkout.'],
    limitations: ['A checkout link is not proof of payment or successful fulfillment. Custom installation is not included unless separately agreed.'],
    sourceUrls: ['/shop/cinematic-starter/'], assetRefs: [], lastVerifiedAt: '2026-09-21',
  },
  {
    id: 'promo-organizer', title: 'Promo File Organizer / DG Promo Photos', kind: 'internal tool', availability: 'preview',
    aliases: ['promo file organizer', 'promo organizer', 'promo photos', 'dg promo', 'photo organizer'],
    businessTasks: ['job photos', 'organize photos', 'print shop', 'photo intake', 'upload queue', 'sort by job'],
    supportedClaims: ['DG Promo Photos is a private photo-intake companion to the separate Promo File Organizer. Enter a job and company, choose photos, label Front/Back/Detail, and upload the queue to PromoUploads through Cloudflare email-code access.', 'The intake preserves supplied file bytes and reports upload quality checks.'],
    limitations: ['Automatic sorting into job folders by the separate organizer remains unverified; the latest recorded acceptance check did not observe sorting. Physical iPhone acceptance remains pending. This is not a general production-tracking system.'],
    sourceUrls: ['/services/business-systems/'], assetRefs: [], lastVerifiedAt: '2026-09-21',
  },
  {
    id: 'wellway', title: 'Wellway', kind: 'fictional demonstration', availability: 'live',
    aliases: ['wellway', 'well way', 'welway'], businessTasks: ['wellness', 'check in', 'supporting records'],
    supportedClaims: ['Wellway is a fictional wellness demonstration with check-ins, browser-local saves, recovery exports, editable plans and source-linked assistance when its bounded hosted allowance is available.'],
    limitations: ['Missing measurements stay unknown. It is not a clinical deployment, diagnosis service or compliance certification.'],
    sourceUrls: ['/demos/wellway/'], demoUrl: '/demos/wellway/', assetRefs: ['/images/site-gallery/wellway-journey.webp'], lastVerifiedAt: '2026-09-21',
  },
  {
    id: 'embroidery', title: 'EmbroideryCalc', kind: 'internal tool', availability: 'live',
    aliases: ['embroiderycalc', 'embroidery calculator', 'embroidery calc', 'embrodiery calculator'], businessTasks: ['stitch count', 'embroidery', 'dst files', 'production time'],
    supportedClaims: ['EmbroideryCalc estimates embroidery production time, inspects DST files and matches artwork colors to Madeira threads. Files, history and calibration stay in the browser.'],
    limitations: ['Estimates are planning aids, not an official commercial quote. Physical-device and App Store acceptance are separate from the web tool.'],
    sourceUrls: ['/work/'], demoUrl: 'https://embroiderycalc-public.pages.dev/', assetRefs: ['/images/site-gallery/embroiderycalc-pro.webp'], lastVerifiedAt: '2026-09-21',
  },
  {
    id: 'pernr', title: 'PERNR employee access gate', kind: 'client implementation', availability: 'live',
    aliases: ['pernr', 'employee gate', 'employee access', 'roster gate'], businessTasks: ['eligibility', 'private store', 'employee id', 'roster'],
    supportedClaims: ['The employee gate checks a PERNR or approved name against a controlled Google Sheet through Apps Script, normalizing spacing and capitalization. The public case study uses fictional credentials and keeps the employee roster private.'],
    limitations: ['Completed within Data Graphics client-services work, not a direct Disney engagement or endorsement. The gate is not enterprise SSO or high-assurance identity verification.'],
    sourceUrls: ['/work/pernr-access-gate/'], demoUrl: '/work/pernr-access-gate/', assetRefs: ['/images/case-studies/pernr-access-gate.png'], lastVerifiedAt: '2026-09-21',
  },
  {
    id: 'storefront-projects', title: 'Nighttime Spectaculars and Holidays in Hollywood', kind: 'client implementation', availability: 'live',
    aliases: ['nighttime spectaculars', 'holidays in hollywood', 'storefronts'], businessTasks: ['storefront', 'product paths', 'branded entrance'],
    supportedClaims: ['Branded storefront entrances and product navigation were delivered within Data Graphics client-services work. Hosted commerce retains products, accounts, cart and checkout. Eidos Works scopes interfaces around the chosen platform.'],
    limitations: ['No Disney endorsement, direct Disney client relationship or platform license entitlement is claimed.'],
    sourceUrls: ['/work/'], assetRefs: [], lastVerifiedAt: '2026-09-21',
  },
  {
    id: 'playground', title: 'Eidos Playground', kind: 'product', availability: 'preview',
    aliases: ['playground'], businessTasks: ['page editor', 'page composition'],
    supportedClaims: ['Playground supports local page composition and reversible edits. Legacy projects have an owned cloud and test-export path.'],
    limitations: ['New structured blocks and media remain local-only at the current release boundary. AI editing is not yet active. Paid export is test-only.'],
    sourceUrls: ['/playground/'], demoUrl: '/playground/', assetRefs: [], lastVerifiedAt: '2026-09-21',
  },
  {
    id: 'snapshot', title: 'Eidos Snapshot', kind: 'product', availability: 'coming soon',
    aliases: ['snapshot'], businessTasks: ['website report', 'concept image'],
    supportedClaims: ['Snapshot currently offers a launch notice for a planned website report and concept image.'],
    limitations: ['Paid generation and durable delivery have not passed launch acceptance. A launch notice is not a purchase or delivered report.'],
    sourceUrls: ['/snapshot/'], assetRefs: [], lastVerifiedAt: '2026-09-21',
  },
];
