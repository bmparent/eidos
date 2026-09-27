export const knowledge = [
  {
    id: 'work',
    title: 'Explore the work',
    href: '/work',
    match: /\b(work|portfolio|example|built|showcase)\b/i,
    text: 'Eidos Works creates custom storefront experiences, websites, dashboards, and useful workflow tools. The portfolio includes Nighttime Spectaculars, Holidays in Hollywood, a roster-backed store access system, and Sentinel Lab. Each project explains the role and the platform constraints.',
  },
  {
    id: 'storefront',
    title: 'Storefront design and access',
    href: '/services/storefront-access-systems',
    match:
      /\b(inksoft|store|storefront|commerce|disney|hollywood|nighttime|shop|cart)\b/i,
    text: 'Eidos can design a branded entrance, improve product paths, and integrate eligibility checks within a hosted storefront. InkSoft continues to handle products, accounts, cart, and checkout. Portfolio examples include work completed within Data Graphics’ client-services workflow; no Disney endorsement is claimed.',
  },
  {
    id: 'lab',
    title: 'Eidos / Sentinel Lab',
    href: '/lab',
    match: /\b(lab|sentinel|research|brain|experiment)\b/i,
    text: 'Sentinel Lab runs separately at https://eidos-sentinel-lab.vercel.app/. It explores streaming prediction, surprise, and human review. Inspect the live lab for current evidence. Synthetic engineering tests do not establish research proof. The public assistant cannot operate lab experiments.',
  },
  {
    id: 'studio',
    title: 'Meet Brent Parent',
    href: '/about',
    match: /\b(brent|founder|who|about|location|florida|team)\b/i,
    text: 'Brent Parent founded Eidos Works in Central Florida. The studio brings together design, storefront development, production operations, reporting, and AI-assisted implementation, and collaborates with client teams.',
  },
  {
    id: 'build',
    title: 'Start a project',
    href: '/contact',
    match:
      /\b(build|website|project|quote|cost|price|budget|timeline|contact|hire|custom)\b/i,
    text: 'Bring a brief description of what you want to build, who will use it, your current platform, and the main constraint. Eidos scopes websites, storefronts, dashboards, and focused automations around the actual problem. Custom pricing and delivery dates require a conversation with Brent; the assistant cannot commit to a quote.',
  },
  {
    id: 'kit',
    title: 'Cinematic Starter',
    href: '/shop/cinematic-starter',
    match: /\b(kit|starter|download|template|glass|header|hero)\b/i,
    text: 'Cinematic Starter is an original responsive glass header and hero package with plain HTML, CSS, JavaScript, setup notes, and a commercial license for one finished website. The intended one-time price is $29 USD. The product page shows whether checkout is available. Client brand assets and the Eidos homepage illustration are not included.',
  },
  {
    id: 'community',
    title: 'Community and Agent Exchange',
    href: '/community',
    match: /\b(question|community|forum|agent|assistant|token|bot)\b/i,
    text: 'People can submit questions and replies for review. Mention @eidos in a question to request a public source-based answer after approval. Optional follow-up allows one helpful source suggestion on an unanswered thread after 24 hours. Agent Exchange accepts operator-registered agents through a rate-limited API. The assistant never starts agent-to-agent reply loops.',
  },
  {
    id: 'automation',
    title: 'Dashboards and workflow tools',
    href: '/services/dashboards-workflow-tools',
    match: /\b(dashboard|automat|workflow|report|data|integrat)/i,
    text: 'Eidos builds operational views and focused automations around an existing source of data. Work includes production reporting with department and date filters and searchable work orders. Access, data shape, and the actual daily decision determine the implementation.',
  },
];
export function selectKnowledge(question: string) {
  const selected = knowledge.filter((item) => item.match.test(question));
  return (selected.length ? selected : [knowledge[0], knowledge[4]]).slice(
    0,
    3,
  );
}
export function sourceAnswer(question: string) {
  const sources = selectKnowledge(question);
  return {
    answer: sources
      .slice(0, 2)
      .map((s) => s.text)
      .join('\n\n'),
    sources: sources.map(({ title, href }) => ({ title, href })),
    mode: 'sources' as const,
  };
}
